"""离线构造 tiny 模型。缺依赖时明确跳过，不下载任何模型。"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

from rsi.artifacts import model_manifest, write_json
from rsi.attempt_run import load_attempt
from rsi.backends import TrainingRequest
from rsi.checkpoints import TrainingPaused
from rsi.hf_backend import HfLoraBackend, load_tokenizer
from rsi.model_inputs import bound_model
from rsi.model_resolution import LoadConfig, ModelLocation
from rsi.training_demo import run_demo
from rsi.validation import ValidationError, read_json, to_json_value


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "缺少 transformers、peft 或 tokenizers；真实 tiny 参数更新与恢复未执行。")
class TrainingHfOptionalTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "tiny"
        vocabulary = {word: index for index, word in enumerate(["[UNK]", "[EOS]"] + [chr(i) for i in range(32, 127)] + ["\n", "\t"])}
        core = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        core.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="[EOS]", pad_token="[UNK]")
        tokenizer.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '>' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
        tokenizer.save_pretrained(self.model)
        Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, eos_token_id=1, pad_token_id=0,
            max_position_embeddings=1024)).save_pretrained(self.model)
        self.location = ModelLocation.parse({"schema_version": "model-location-v1",
            "model_ref": bound_model(model_manifest(self.model), load_tokenizer(self.model)), "base_model_path": str(self.model),
            "adapter_manifest": None, "load": to_json_value(LoadConfig())})
        self.location_file = self.root / "location.json"
        write_json(self.location_file, self.location.to_dict())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def first(self):
        run_demo(self.root / "a", mode="hf_local", location_path=self.location_file)
        from rsi.validation_run import completed_candidate
        return completed_candidate(self.root / "a/attempt")

    def test_tiny_full_measurements_are_measured_not_fixture(self) -> None:
        self.first()
        report = read_json(self.root / "a/demo_report.json")
        self.assertEqual(report["source"], "measured")
        self.assertEqual(report["data_kind"], "synthetic")
        log = read_json(self.root / "a/attempt/training/training.json")
        self.assertGreater(log["adapter_delta_l2"], 0)
        self.assertGreater(log["trainable_parameters"], 0)
        from rsi.round_feedback import parse_round_feedback
        feedback = parse_round_feedback(read_json(self.root / "a/attempt/validation/round_feedback.json"))
        self.assertEqual(feedback.source, "measured")

    def test_parent_adapter_advances_and_resume_matches_continuous(self) -> None:
        import torch
        from safetensors.torch import load_file
        from rsi.artifacts import file_sha256, verify_training_artifact
        first = self.first()
        parent_file = Path(first.artifact.path).parent / "adapter/adapter_model.safetensors"
        parent_sha = file_sha256(parent_file)
        parent_location = replace(self.location, model_ref=first.candidate_model, adapter_manifest=first.artifact)
        parent_path = self.root / "parent.json"
        write_json(parent_path, parent_location.to_dict())
        run_demo(self.root / "b", mode="hf_local", location_path=parent_path)
        plan, inputs, _ = load_attempt(self.root / "b/attempt")
        updated = load_file(self.root / "b/attempt/training/adapter/adapter_model.safetensors")
        parent = load_file(parent_file)
        self.assertTrue(any(not torch.equal(updated[k], parent[k]) for k in parent))
        self.assertEqual(file_sha256(parent_file), parent_sha)
        verify_training_artifact(first.artifact)
        request = TrainingRequest(plan.run_id, plan.attempt_id, plan.input_hashes["contract_hash"], plan.batch.model,
                    plan.batch, plan.training, None, plan.lora, plan.load, parent_location.identity)
        counts = {}
        def count(step, record, amount):
            counts.setdefault(step, amount)
        backend = HfLoraBackend(self.model, plan.batch.model, inputs.records, self.root / "resumed", "cpu", count, 1, first.artifact)
        with self.assertRaises(TrainingPaused) as caught:
            backend.train(request)
        HfLoraBackend(self.model, plan.batch.model, inputs.records, self.root / "resumed", "cpu", count,
                      parent_adapter=first.artifact).train(replace(request, resume_artifact=caught.exception.artifact))
        resumed = load_file(self.root / "resumed/adapter/adapter_model.safetensors")
        self.assertTrue(all(torch.equal(updated[k], resumed[k]) for k in updated))
        self.assertEqual(sum(counts.values()), plan.batch.supervised_tokens)

    def test_parent_adapter_config_and_content_tamper_rejected(self) -> None:
        first = self.first()
        plan, inputs, _ = load_attempt(self.root / "a/attempt")
        batch = replace(plan.batch, model=first.candidate_model)
        request = TrainingRequest("r", "bad", plan.input_hashes["contract_hash"], first.candidate_model,
                    batch, plan.training, None, replace(plan.lora, alpha=8), plan.load, "identity")
        backend = HfLoraBackend(self.model, first.candidate_model, inputs.records, self.root / "bad", "cpu", lambda *args: None,
                                parent_adapter=first.artifact)
        with self.assertRaises(ValidationError):
            backend.train(request)
        adapter_file = Path(first.artifact.path).parent / "adapter/adapter_config.json"
        with adapter_file.open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaises(ValidationError):
            backend.train(replace(request, lora_config=plan.lora))
