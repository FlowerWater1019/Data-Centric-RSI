"""不依赖模型的消息位置与监督映射检查。"""

from copy import deepcopy
from typing import Any

from rsi.tokenization import compile_trajectory
from rsi.validation import ValidationError, read_jsonl
from tests.helpers import Fixture, ROOT


class CharacterTokenizer:
    is_fast = True
    chat_template = "test_only"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "".join(f'<{m["role"]}>{m["content"]}</end>' for m in messages)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in ids)


class TokenizationTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        self.source = read_jsonl(ROOT / 'examples/pilot_messages.jsonl')[0]

    def compile(self, source: dict | None = None, tokenizer: CharacterTokenizer | None = None, limit: int = 2048) -> dict:
        return compile_trajectory(source or self.source, tokenizer or CharacterTokenizer(), self.contract, limit)

    def test_failed_action_and_template_are_not_supervised(self) -> None:
        record = self.compile()
        supervised = ''.join(c for c, enabled in zip(record['text'], record['supervision']['loss_mask']) if enabled)
        self.assertEqual(supervised, self.source['messages'][-1]['content'])
        self.assertIn('return None', record['text'])
        self.assertEqual(record['tokenization']['status'], 'verified')

    def test_overlength_fails_without_truncation(self) -> None:
        with self.assertRaisesRegex(ValidationError, '截断'):
            self.compile(limit=10)

    def test_wrong_action_cannot_request_supervision(self) -> None:
        self.source['messages'][1]['supervise'] = True
        with self.assertRaisesRegex(ValidationError, '有效 assistant'):
            self.compile()

    def test_template_content_transformation_is_rejected(self) -> None:
        class TransformingTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
                chat = deepcopy(messages)
                for message in chat:
                    message['content'] = message['content'].strip()
                return super().apply_chat_template(chat, **kwargs)

        with self.assertRaisesRegex(ValidationError, '按内容改变'):
            self.compile(tokenizer=TransformingTokenizer())

    def test_roundtrip_mismatch_is_rejected(self) -> None:
        class LossyTokenizer(CharacterTokenizer):
            def decode(self, ids: list[int], **kwargs: Any) -> str:
                return super().decode(ids, **kwargs).strip('</end>')

        with self.assertRaisesRegex(ValidationError, '往返'):
            self.compile(tokenizer=LossyTokenizer())

    def test_diagnostic_range_stays_within_valid_answer(self) -> None:
        record = self.compile()
        span = record['diagnostic_spans'][0]
        self.assertTrue(all(record['supervision']['loss_mask'][span['token_start']:span['token_end']]))
        self.assertEqual(record['text'][span['char_start']:span['char_end']], self.source['messages'][-1]['content'])
