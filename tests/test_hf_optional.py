"""可选依赖的离线 CPU 集成检查。无依赖时明确跳过。"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from rsi.backends import TrainingRequest
from rsi.checkpoints import TrainingPaused
from rsi.hf_backend import HfLoraBackend, HfRuntime
from rsi.selection import FixedSelector, build_batch
from rsi.validation import ValidationError, read_json
from tests.helpers import Fixture, ROOT


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','peft'))


@unittest.skipUnless(AVAILABLE, '未安装可选模型依赖；本地基础测试不要求模型环境。')
class HfCpuTests(Fixture):
    def test_diagnostic_spans_do_not_change_variable_length_training(self) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from rsi.data import validate_dataset
        from rsi.selection import BatchItem

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            model_path = root / 'model'
            Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).save_pretrained(model_path)
            context = self.context()
            model_ref = replace(context.contract.model, adapter_revision='none')
            training = replace(context.contract.training, backend='hf_lora', world_size=1, precision='fp32',
                learning_rate=0.001, lora_rank=2, max_sequence_length=64, per_device_batch_size=1, gradient_accumulation_steps=1)
            contract = replace(context.contract, model=model_ref, training=training)
            # 两个完整轨迹使用不同监督量。诊断开关不改变任一训练掩码。
            rows = [r.raw for r in context.records]
            row = next(row for row in rows if row['data_id'] == 'b')
            enabled = [i for i, bit in enumerate(row['supervision']['loss_mask']) if bit]
            row['supervision']['loss_mask'][enabled[-1]] = False
            row['diagnostic_spans'] = []
            records = validate_dataset(rows, contract)
            chosen = [next(r for r in records if r.data_id == key) for key in ('c', 'b')]
            batch = build_batch(context, FixedSelector())
            batch = replace(batch, model=model_ref, supervised_tokens=7,
                items=tuple(BatchItem(r.data_id,r.version,r.train_mask_version,r.content_hash,1,r.supervised_tokens) for r in chosen))
            request = TrainingRequest('run','attempt',context.contract_hash,model_ref,batch,training)
            HfLoraBackend(model_path,model_ref,records,root/'marked','cpu',lambda *args: None).train(request)
            for row in rows:
                row['diagnostic_spans'] = []
            plain = validate_dataset(rows, contract)
            HfLoraBackend(model_path,model_ref,plain,root/'plain','cpu',lambda *args: None).train(request)
            left = load_file(root/'marked/adapter/adapter_model.safetensors')
            right = load_file(root/'plain/adapter/adapter_model.safetensors')
            self.assertTrue(all(torch.equal(left[key],right[key]) for key in left))
            log = read_json(root/'marked/training.json')
            self.assertEqual([step['supervised_tokens'] for step in log['steps']], [4,3])
            self.assertEqual(log['actual_supervised_tokens'], 7)
            self.assertFalse(log['diagnostics_affect_loss'])

    def test_tiny_local_model_trains_saves_and_reloads(self) -> None:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            model_path = Path(folder) / 'model'
            config = Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,
                                num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
            Qwen2ForCausalLM(config).save_pretrained(model_path)
            context = self.context()
            model_ref = replace(context.contract.model,adapter_revision='none')
            training = replace(context.contract.training,backend='hf_lora',world_size=1,precision='fp32',
                               learning_rate=0.001,lora_rank=2,max_sequence_length=64,
                               per_device_batch_size=1,gradient_accumulation_steps=1)
            context = replace(context,contract=replace(context.contract,model=model_ref,training=training))
            batch = build_batch(context,FixedSelector())
            uses = []
            backend = HfLoraBackend(model_path,model_ref,context.records,Path(folder)/'attempt','cpu',
                                    lambda step,record,count: uses.append((step,record.data_id,count)))
            result = backend.train(TrainingRequest('run','attempt',context.contract_hash,model_ref,batch,training))
            self.assertEqual(result.actual_supervised_tokens,8)
            self.assertEqual(sum(item[2] for item in uses),8)
            self.assertGreater(read_json(result.diagnostic_artifact.path)['adapter_delta_l2'],0)
            self.assertTrue(Path(result.artifact.path).is_file())
            with HfRuntime(model_path,'cpu','fp32',Path(folder)/'attempt/adapter') as runtime:
                metric = runtime.score(context.records[0])
                self.assertEqual(metric.supervised_tokens,4)
                self.assertTrue(0 < metric.nll < 100)

    def test_resumed_training_matches_continuous_parameters_and_rng(self) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            model_path = root / 'model'
            Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).save_pretrained(model_path)
            context = self.context()
            model_ref = replace(context.contract.model, adapter_revision='none')
            training = replace(context.contract.training, backend='hf_lora', world_size=1, precision='fp32',
                               learning_rate=0.001, lora_rank=2, max_sequence_length=64,
                               per_device_batch_size=1, gradient_accumulation_steps=1)
            context = replace(context, contract=replace(context.contract, model=model_ref, training=training))
            batch = build_batch(context, FixedSelector())
            request = TrainingRequest('run', 'attempt', context.contract_hash, model_ref, batch, training)
            counts = {}
            def record_use(step, record, count):
                previous = counts.setdefault(step, (record.data_id, count))
                self.assertEqual(previous, (record.data_id, count))
            continuous = HfLoraBackend(model_path, model_ref, context.records, root / 'continuous', 'cpu', lambda *args: None)
            continuous.train(request)
            paused = HfLoraBackend(model_path, model_ref, context.records, root / 'resumed', 'cpu', record_use, 1)
            with self.assertRaises(TrainingPaused) as caught:
                paused.train(request)
            checkpoint = caught.exception.artifact
            self.assertEqual(sum(value[1] for value in counts.values()), 4)
            resumed = HfLoraBackend(model_path, model_ref, context.records, root / 'resumed', 'cpu', record_use)
            result = resumed.train(replace(request, resume_artifact=checkpoint))
            self.assertEqual(result.actual_supervised_tokens, 8)
            self.assertEqual(sum(value[1] for value in counts.values()), 8)
            first = load_file(root / 'continuous/adapter/adapter_model.safetensors')
            second = load_file(root / 'resumed/adapter/adapter_model.safetensors')
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]), key)
            first_state = torch.load(root / 'continuous/optimizer_state.pt', weights_only=True)
            second_state = torch.load(root / 'resumed/optimizer_state.pt', weights_only=True)
            self.assertTrue(torch.equal(first_state['torch_rng'], second_state['torch_rng']))
            self.assertEqual(first_state['python_rng'], second_state['python_rng'])
            self.assertEqual(first_state['scheduler'], second_state['scheduler'])
            self.assertEqual(read_json(root / 'resumed/training.json')['resumed_from_step'], 1)

    def test_failure_after_backward_preserves_cost_and_blocks_resume(self) -> None:
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            model_path = root / 'model'
            Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)).save_pretrained(model_path)
            context = self.context()
            model_ref = replace(context.contract.model, adapter_revision='none')
            training = replace(context.contract.training, backend='hf_lora', world_size=1, precision='fp32',
                               learning_rate=0.001, lora_rank=2, max_sequence_length=64,
                               per_device_batch_size=1, gradient_accumulation_steps=1)
            context = replace(context, contract=replace(context.contract, model=model_ref, training=training))
            batch = build_batch(context, FixedSelector())
            request = TrainingRequest('run', 'attempt', context.contract_hash, model_ref, batch, training)
            uses = {}
            def record_use(step, record, count):
                uses.setdefault(step, count)
                if step == 2:
                    raise RuntimeError('injected after backward')
            paused = HfLoraBackend(model_path, model_ref, context.records, root / 'attempt', 'cpu', record_use, 1)
            with self.assertRaises(TrainingPaused) as caught:
                paused.train(request)
            request = replace(request, resume_artifact=caught.exception.artifact)
            backend = HfLoraBackend(model_path, model_ref, context.records, root / 'attempt', 'cpu', record_use)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                backend.train(request)
            self.assertEqual(sum(uses.values()), 8)
            with self.assertRaisesRegex(ValidationError, '成本可能未知'):
                backend.train(request)
            self.assertEqual(sum(uses.values()), 8)
