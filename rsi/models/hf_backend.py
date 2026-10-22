"""本地 Hugging Face 单卡适配器。重型依赖在执行时加载。"""

from collections.abc import Callable
import gc
from pathlib import Path
import random
import time
from typing import Any

from rsi.storage.artifacts import file_sha256, write_json
from rsi.models.backends import ArtifactRef, TrainingRequest, TrainingResult
from rsi.configuration.contracts import ModelRef
from rsi.storage.checkpoints import StepJournal, TrainingPaused, request_identity
from rsi.datasets.data import TrajectoryRecord
from rsi.models.scoring import SpanMetric, TokenMetrics, TokenScore
from rsi.configuration.validation import fail, digest


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers
        import peft
    except ImportError as exc:
        fail("dependencies", "需要可用的 torch、transformers 和 peft 环境：" + str(exc))
    return torch, transformers, peft


def load_tokenizer(model_path: str | Path) -> Any:
    _, transformers, _ = load_dependencies()
    return transformers.AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True,
    )


def masked_targets(record: TrajectoryRecord) -> tuple[list[int], list[int]]:
    """返回预测位置与目标 token。目标位置只向左移一次。"""
    raw = record.raw
    ids = raw["tokenization"]["token_ids"]
    positions = [i for i, enabled in enumerate(raw["supervision"]["loss_mask"]) if enabled]
    if not positions or positions[0] == 0:
        fail(record.data_id, "需要具有前序位置的有效监督 token。")
    return [i - 1 for i in positions], [ids[i] for i in positions]


class HfRuntime:
    """持有一个已加载模型。使用上下文管理器释放设备资源。"""

    def __init__(
        self, model_path: str | Path, device: str, precision: str, adapter_path: str | Path | None = None,
        *, adapter_trainable: bool = False, attention_implementation: str = "eager",
    ) -> None:
        self.torch, transformers, peft = load_dependencies()
        if device != "cpu" and not device.startswith("cuda:"):
            fail("device", "只支持 cpu 或显式 cuda 设备。")
        if device.startswith("cuda:") and not self.torch.cuda.is_available():
            fail("device", "当前进程没有可用 CUDA 设备。")
        if precision not in {"bf16", "fp32"}:
            fail("precision", "只支持 bf16 或 fp32。")
        if device == "cpu" and precision != "fp32":
            fail("precision", "CPU 工程检查使用 fp32。")
        if device.startswith("cuda:"):
            self.torch.cuda.set_device(device)
            if precision == "bf16" and not self.torch.cuda.is_bf16_supported():
                fail("precision", "当前设备不支持 bf16。")
        self.device = device
        dtype = self.torch.bfloat16 if precision == "bf16" else self.torch.float32
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=dtype, attn_implementation=attention_implementation,
        ).to(device)
        self.model.config.use_cache = False
        if adapter_path is not None:
            try:
                self.model = peft.PeftModel.from_pretrained(
                    self.model, str(adapter_path), is_trainable=adapter_trainable, local_files_only=True,
                )
            except BaseException:
                self.__exit__()
                raise

    def __enter__(self) -> "HfRuntime":
        return self

    def __exit__(self, *args: Any) -> None:
        del self.model
        gc.collect()
        if self.device.startswith("cuda:"):
            self.torch.cuda.empty_cache()

    def score_tokens(self, record: TrajectoryRecord) -> tuple[TokenScore, ...]:
        """有效监督位置的逐 token CE 和完整词表熵。仅评估，不修改掩码。"""
        if record.role == "final_test":
            fail(record.data_id, "最终测试数据不能进入筛选评分。")
        torch = self.torch
        raw = record.raw
        ids = torch.tensor([raw["tokenization"]["token_ids"]], dtype=torch.long, device=self.device)
        prediction_positions, targets = masked_targets(record)
        self.model.eval()
        ce_values, entropy_values = [], []
        with torch.no_grad():
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
            # 逐小块计算完整词表，避免额外保留整个序列的 fp32 概率矩阵。
            for left in range(0, len(targets), 8):
                rows = logits[prediction_positions[left:left + 8]].float()
                log_prob = torch.log_softmax(rows, dim=-1)
                target = torch.tensor(targets[left:left + 8], device=self.device)
                ce = -log_prob.gather(1, target[:, None]).squeeze(1)
                entropy = -(log_prob.exp() * log_prob).sum(dim=-1)
                ce_values.extend(ce.cpu().tolist())
                entropy_values.extend(entropy.cpu().tolist())
        return tuple(TokenScore(position + 1, ce, entropy) for position, ce, entropy in
                     zip(prediction_positions, ce_values, entropy_values))

    def score(self, record: TrajectoryRecord) -> TokenMetrics:
        """复用逐 token 计算，保持旧评分接口和诊断语义。"""
        values = self.score_tokens(record)
        raw = record.raw
        ce_values = [v.ce for v in values]
        entropy_values = [v.entropy for v in values]
        ce_by_position = {v.position: v.ce for v in values}
        spans = []
        for span in raw["diagnostic_spans"]:
            span_values = [ce_by_position[i] for i in range(span["token_start"], span["token_end"])
                      if i in ce_by_position]
            spans.append(SpanMetric(span["skill"], len(span_values), sum(span_values), sum(span_values) / len(span_values)))
        return TokenMetrics(record.data_id, len(values), record.token_count,
                            sum(ce_values) / len(values), sum(entropy_values) / len(values), tuple(spans),
                            sum(ce_values), sum(entropy_values))

    def generate_code(self, tokenizer: Any, prompt: str, max_new_tokens: int, max_length: int) -> str:
        """执行一次贪心生成。测试答案不会传入提示。拒绝截断任务提示。"""
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        length = inputs["input_ids"].shape[1]
        model_limit = getattr(self.model.config, "max_position_embeddings", max_length)
        if length + max_new_tokens > min(max_length, model_limit):
            fail("evaluation.length", "提示与最大生成长度之和超过配置限制。")
        self.model.eval()
        with self.torch.no_grad():
            result = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                         pad_token_id=tokenizer.eos_token_id, use_cache=True)
        return tokenizer.decode(result[0, length:], skip_special_tokens=True)


class HfLoraBackend:
    """执行单卡 LoRA 更新。每步保存检查点，只恢复完整步骤边界。"""

    def __init__(
        self, model_path: str | Path, model_ref: ModelRef, records: tuple[TrajectoryRecord, ...],
        output_root: str | Path, device: str,
        on_step: Callable[[int, TrajectoryRecord, int], None],
        stop_after_step: int | None = None,
        parent_adapter: ArtifactRef | None = None,
    ) -> None:
        self.model_path, self.model_ref = Path(model_path), model_ref
        self.records = {r.data_id: r for r in records}
        self.output_root, self.device, self.on_step = Path(output_root), device, on_step
        self.stop_after_step = stop_after_step
        self.parent_adapter = parent_adapter

    def train(self, request: TrainingRequest) -> TrainingResult:
        config = request.training_config
        if request.starting_model != self.model_ref or request.batch.model != self.model_ref:
            fail("model", "请求与批次必须使用相同的已核验模型起点。")
        from rsi.storage.artifacts import verify_training_artifact
        from rsi.models.model_resolution import LoraConfig
        from rsi.configuration.validation import read_json
        lora_config = request.lora_config or LoraConfig(alpha=2 * (config.lora_rank or 0))
        parent_path = None
        if self.model_ref.adapter_revision != "none":
            if self.parent_adapter is None or self.parent_adapter.sha256 != self.model_ref.adapter_revision:
                fail("adapter_revision", "已有父 adapter 必须提供匹配的内容清单。")
            verify_training_artifact(self.parent_adapter)
            parent_path = Path(self.parent_adapter.path).parent / "adapter"
            adapter_config = read_json(parent_path / "adapter_config.json")
            if (adapter_config.get("r"), set(adapter_config.get("target_modules", [])),
                adapter_config.get("lora_alpha"), adapter_config.get("lora_dropout")) != (
                config.lora_rank, set(lora_config.targets), lora_config.alpha, lora_config.dropout):
                fail("lora", "继续训练配置与父 adapter 不兼容。")
        elif self.parent_adapter is not None:
            fail("adapter_revision", "无 adapter 的起点不能注入其他父 adapter。")
        if config.backend != "hf_lora" or (config.world_size, config.per_device_batch_size,
                                           config.gradient_accumulation_steps) != (1, 1, 1):
            fail("training", "当前适配器只支持 hf_lora、单卡、单样本、无梯度累积。")
        if config.learning_rate is None or config.lora_rank is None or config.max_sequence_length is None:
            fail("training", "学习率、LoRA 秩和最大长度必须填写。")
        selected = []
        for item in request.batch.items:
            record = self.records.get(item.data_id)
            if record is None or not record.is_eligible or item.repetitions != 1:
                fail("batch", "批次必须来自可选候选，且每条轨迹只训练一遍。")
            if (item.version, item.mask_version, item.content_hash, item.supervised_tokens) != (
                record.version, record.train_mask_version, record.content_hash, record.supervised_tokens
            ):
                fail("batch", "批次与轨迹内容不一致。")
            if record.token_count > config.max_sequence_length:
                fail(record.data_id, "轨迹超过训练最大长度。")
            selected.append(record)
        if not selected or sum(r.supervised_tokens for r in selected) != request.batch.supervised_tokens:
            fail("batch", "批次监督量不一致。")
        if len({r.data_id for r in selected}) != len(selected):
            fail("batch", "批次不能包含重复轨迹。")
        output = self.output_root
        journal = StepJournal(output, request_identity(request, selected, self.device))
        resume_path, completed = None, 0
        if request.resume_artifact is None:
            output.mkdir(parents=True, exist_ok=False)
        else:
            resume_path, completed = journal.validate_resume(request.resume_artifact)
            if completed > len(selected):
                fail("resume", "检查点步骤数超过冻结批次。")
        torch, _, peft = load_dependencies()
        random.seed(request.batch.seed)
        torch.manual_seed(request.batch.seed)
        with HfRuntime(self.model_path, self.device, config.precision, parent_path,
                       adapter_trainable=parent_path is not None,
                       attention_implementation=request.load_config.attention_implementation if request.load_config else "eager") as runtime:
            matches = {target: [name for name, _ in runtime.model.named_modules()
                       if name == target or name.endswith("." + target)] for target in lora_config.targets}
            if any(not names for names in matches.values()):
                fail("lora.targets", "部分目标模块未匹配实际模型。")
            lora = peft.LoraConfig(
                task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout, target_modules=list(lora_config.targets), bias="none",
            )
            if parent_path is None:
                runtime.model = peft.get_peft_model(runtime.model, lora)
            model = runtime.model
            trainable = [p for p in model.parameters() if p.requires_grad]
            if not trainable:
                fail("lora", "可训练参数为空。")
            initial = [p.detach().float().cpu().clone() for p in trainable]
            optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            model.train()
            losses, actual = [], 0
            if resume_path is not None:
                saved = torch.load(resume_path, map_location="cpu", weights_only=True)
                if saved["completed_steps"] != completed:
                    fail("resume", "检查点内外步骤数不一致。")
                parameters = dict(model.named_parameters())
                with torch.no_grad():
                    for name, value in saved["adapter"].items():
                        parameters[name].copy_(value)
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                initial, losses, actual = saved["initial"], saved["losses"], saved["actual_supervised_tokens"]
                if actual != sum(r.supervised_tokens for r in selected[:completed]) or len(losses) != completed:
                    fail("resume", "检查点监督量或步骤记录不一致。")
                # 账本回调必须幂等。恢复不重复反向传播已完成数据。
                for step, record in enumerate(selected[:completed], 1):
                    self.on_step(step, record, record.supervised_tokens)
                torch.set_rng_state(saved["torch_rng"])
                random.setstate(saved["python_rng"])
                if self.device.startswith("cuda:"):
                    torch.cuda.set_rng_state_all(saved["cuda_rng"])
            for step, record in enumerate(selected, 1):
                if step <= completed:
                    continue
                journal.begin(step)
                started_at = time.monotonic()
                raw = record.raw
                ids = torch.tensor([raw["tokenization"]["token_ids"]], device=self.device, dtype=torch.long)
                positions, targets = masked_targets(record)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
                target = torch.tensor(targets, device=self.device)
                token_ce = torch.nn.functional.cross_entropy(logits[positions].float(), target, reduction="none")
                loss = token_ce.mean()
                if not torch.isfinite(loss):
                    fail("loss", "训练损失不是有限值。")
                loss.backward()
                # 已完成反向传播即产生真实训练消耗，即使随后更新或保存失败也不能清除。
                self.on_step(step, record, len(targets))
                actual += len(targets)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                ce_by_position = dict(zip((p + 1 for p in positions), token_ce.detach().cpu().tolist()))
                spans = []
                for span in raw["diagnostic_spans"]:
                    values = [ce_by_position[i] for i in range(span["token_start"], span["token_end"]) if i in ce_by_position]
                    spans.append({"skill": span["skill"], "token_count": len(values), "ce_sum": sum(values),
                                  "mean_ce": sum(values) / len(values)})
                losses.append({"step": step, "data_id": record.data_id, "supervised_tokens": len(targets),
                               "context_tokens": record.token_count, "mean_ce": loss.detach().item(), "spans": spans,
                               "update_seconds": time.monotonic() - started_at})
                checkpoint = {"adapter": {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad},
                              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                              "initial": initial, "losses": losses, "completed_steps": step,
                              "actual_supervised_tokens": actual, "python_rng": random.getstate(),
                              "torch_rng": torch.get_rng_state(),
                              "cuda_rng": torch.cuda.get_rng_state_all() if self.device.startswith("cuda:") else []}
                artifact = journal.publish(step, lambda path: torch.save(checkpoint, path))
                if self.stop_after_step == step:
                    raise TrainingPaused(artifact)
            delta_l2 = sum((p.detach().float().cpu() - old).square().sum().item()
                           for p, old in zip(trainable, initial)) ** 0.5
            model.save_pretrained(output / "adapter", safe_serialization=True)
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if self.device.startswith("cuda:") else [],
                        "completed_steps": len(losses), "actual_supervised_tokens": actual,
                        "request_id": request.attempt_id, "contract_hash": request.contract_hash,
                        "batch_fingerprint": request.batch.fingerprint}, output / "optimizer_state.pt")
            diagnostics = write_json(output / "training.json", {
                "steps": losses, "actual_supervised_tokens": actual, "adapter_delta_l2": delta_l2,
                "trainable_parameters": sum(p.numel() for p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "training_status": "COMPLETED", "resume_supported": "complete_step_boundary",
                "resumed_from_step": completed,
                "loss_reduction": "mean_per_trajectory_one_optimizer_update",
                "context_tokens": sum(r.token_count for r in selected),
                "update_seconds": sum(row["update_seconds"] for row in losses),
                "diagnostics_affect_loss": False,
                "lora": {"rank": config.lora_rank, "alpha": lora_config.alpha,
                         "dropout": lora_config.dropout, "targets": list(lora_config.targets)},
                "parent_model": self.model_ref.__dict__,
            })
        if self.parent_adapter is not None:
            verify_training_artifact(self.parent_adapter)
        files = [{"name": str(p.relative_to(output)), "sha256": file_sha256(p)}
                 for p in sorted(output.rglob("*")) if p.is_file() and
                 (request.model_location_hash is None or p.is_relative_to(output / "adapter"))]
        # 独立训练的模型引用只绑定 adapter 文件。步骤耗时和优化器不改变模型内容身份。
        manifest = {"files": files, "parent_model": self.model_ref.__dict__} if request.model_location_hash else {
                    "attempt_id": request.attempt_id, "contract_hash": request.contract_hash, "files": files}
        artifact = write_json(output / "manifest.json", manifest)
        model_ref = ModelRef("candidate_" + artifact.sha256, self.model_ref.base_revision,
                             artifact.sha256, self.model_ref.tokenizer_revision, self.model_ref.template_revision)
        return TrainingResult(model_ref, artifact, actual, diagnostics)
