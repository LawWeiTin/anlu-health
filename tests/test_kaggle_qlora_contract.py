from pathlib import Path

SCRIPT = Path("training/medgemma_kaggle_qlora.py")


def test_kaggle_qlora_is_syntax_valid() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    compile(source, str(SCRIPT), "exec")


def test_kaggle_qlora_keeps_secrets_and_heavy_artifacts_private() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'get_secret("HF_TOKEN")' in source
    assert 'optional_secret("GH_TOKEN")' in source
    assert 'os.environ.get("ANLU_SNAPSHOT_DIR")' in source
    assert 'Path("/kaggle/temp/anlu-health-qlora")' in source
    assert 'Path("/kaggle/working/anlu-health/medgemma-qlora")' in source
    assert 'os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"' in source
    assert 'os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"' in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in source
    assert "torch.cuda.device_count() == 1" in source
    assert "for access_attempt in range(1, 21)" in source
    assert 'progress_path = RUN_DIR / "progress.log"' in source
    assert "print(hf_token" not in source
    assert "print(gh_token" not in source
    assert "push_to_hub" not in source
    assert "login(" not in source


def test_kaggle_qlora_has_numerical_data_and_release_gates() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "bnb_4bit_compute_dtype=torch.float32" in source
    assert "torch.isfinite(probe_logits).all()" in source
    assert "model.enable_input_require_grads()" in source
    assert "gradient_probe_loss.requires_grad" in source
    assert 'progress("gradient_flow_gate_passed")' in source
    assert '"anlu-authored-safety"' in source
    assert "BEHAVIOR_WEIGHT = 4" in source
    assert 'candidate_evaluation["pass_rate"] == 1.0' in source
    assert '"human_review_required": True' in source
    assert '"promotion_allowed": False' in source


def test_kaggle_qlora_reserves_supervised_completion_tokens() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "MIN_PROMPT_TOKENS = 128" in source
    assert "completion_ids = full_ids[len(prompt_ids) :]" in source
    assert "completion_ids[: MAX_LENGTH - MIN_PROMPT_TOKENS]" in source
    assert "prompt_ids[:header_tokens] + prompt_ids[-tail_tokens:]" in source
    assert "assert any(label != -100 for label in labels)" in source
