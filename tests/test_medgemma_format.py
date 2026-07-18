from training.medgemma_format import (
    as_medgemma_messages,
    clean_generated_text,
    generation_stop_token_ids,
    validate_sft_record,
)


class FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<end_of_turn>": 106}.get(token, self.unk_token_id)


def test_training_messages_include_the_same_system_context_as_inference() -> None:
    rendered = as_medgemma_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "Safety policy",
    )

    assert [message["role"] for message in rendered] == ["system", "user", "assistant"]
    assert rendered[0]["content"][0]["text"] == "Safety policy"
    assert rendered[-1]["content"][0]["text"] == "Hello!"


def test_generation_stops_on_gemma_turn_boundary_and_generic_eos() -> None:
    assert generation_stop_token_ids(FakeTokenizer()) == [106, 1]


def test_clean_generated_text_keeps_only_the_first_assistant_turn() -> None:
    generated = (
        "Contact urgent care today.<end_of_turn>\n"
        "<start_of_turn>model\nThis second turn must never be shown."
    )

    assert clean_generated_text(generated) == "Contact urgent care today."


def test_clean_generated_text_removes_private_reasoning_segment() -> None:
    generated = (
        "<unused94>thought\nInternal analysis.<unused95>model\n"
        "Contact urgent care today.<end_of_turn>"
    )

    assert clean_generated_text(generated) == "Contact urgent care today."


def test_sft_format_gate_rejects_meta_language_and_reserved_tokens() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "Can you answer?"},
            {
                "role": "assistant",
                "content": "I should select an approved topic.<end_of_turn>",
            },
        ]
    }

    issues = validate_sft_record(record)

    assert "assistant target contains a reserved chat token" in issues
    assert "assistant target contains internal or meta-instruction language" in issues


def test_sft_format_gate_accepts_direct_user_facing_answer() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "Can I combine these medicines?"},
            {
                "role": "assistant",
                "content": "Ask a pharmacist to review the exact products before combining them.",
            },
        ]
    }

    assert validate_sft_record(record) == []
