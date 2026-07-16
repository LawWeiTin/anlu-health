from pathlib import Path

SCRIPT = Path("training/medgemma_kaggle_baseline.py")


def test_kaggle_baseline_is_syntax_valid() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    compile(source, str(SCRIPT), "exec")


def test_kaggle_baseline_keeps_tokens_and_large_cache_private() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in source
    assert 'Path("/kaggle/temp/anlu-health")' in source
    assert 'Path("/kaggle/working/anlu-health")' in source
    assert "login(" not in source
    assert "print(hf_token" not in source
    assert "google.colab" not in source
    assert "BitsAndBytesConfig" not in source
    assert 'torch_dtype=torch.float32' in source
    assert 'device_map="auto"' in source
    assert 'max_memory={0: "14GiB", 1: "14GiB"}' in source
    assert 'text.rsplit("<unused95>", 1)[-1]' in source


def test_kaggle_baseline_cannot_auto_promote() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"promotion_allowed": False' in source
    assert '"requires_human_review": True' in source
    assert "push_to_hub" not in source
