"""Private Kaggle QLoRA training and automated evaluation for the MedGemma candidate.

Run only in a private Kaggle notebook with a T4 GPU and enabled encrypted secrets
``HF_TOKEN`` (read-only gated-model access). The project snapshot may be supplied
directly by a private notebook through ``ANLU_SNAPSHOT_DIR`` or fetched with an
optional read-only ``GH_TOKEN``. Large base-model and source-dataset files remain in
``/kaggle/temp``. Only the LoRA adapter and audit reports enter private notebook output.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "google/medgemma-1.5-4b-it"
REPOSITORY = "LawWeiTin/anlu-health"
REPOSITORY_REF = "codex/medgemma-release-candidate"
SEED = 42
MAX_LENGTH = 512
MIN_PROMPT_TOKENS = 128
BEHAVIOR_WEIGHT = 8
TEMP_ROOT = Path("/kaggle/temp/anlu-health-qlora")
OUTPUT_ROOT = Path("/kaggle/working/anlu-health/medgemma-qlora")
RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = OUTPUT_ROOT / RUN_ID

# Kaggle exposes two T4 devices for the selected accelerator. Transformers will
# otherwise wrap this already device-mapped 4-bit model in DataParallel, which
# is unsupported by the PEFT/bitsandbytes path and can cause an illegal CUDA
# memory access on the first optimizer step. One T4 has ample room for this
# 4-bit 4B-model adapter run, so make the process intentionally single-device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = str(TEMP_ROOT / "hf-cache")
os.environ["HF_HUB_CACHE"] = str(TEMP_ROOT / "hf-cache" / "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"


def install_runtime() -> None:
    packages = [
        "transformers>=4.53,<5",
        "accelerate>=1.9,<2",
        "bitsandbytes>=0.46,<1",
        "datasets>=3.6,<5",
        "peft>=0.16,<1",
        "sentencepiece>=0.2,<1",
        "defusedxml>=0.7,<1",
        "PyYAML>=6,<7",
    ]
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "-q", *packages], check=True
    )


install_runtime()

import requests  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from kaggle_secrets import UserSecretsClient  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

assert torch.cuda.is_available(), "Enable a Kaggle GPU before running."
assert torch.cuda.get_device_capability(0)[0] >= 7, "A T4-or-newer GPU is required."
assert torch.cuda.device_count() == 1, "Training must remain single-device for 4-bit QLoRA."
RUN_DIR.mkdir(parents=True, exist_ok=False)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
random.seed(SEED)
progress_path = RUN_DIR / "progress.log"


def progress(message: str) -> None:
    timestamp = datetime.now(UTC).isoformat()
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")
    print(message, flush=True)


progress("runtime_initialized")

secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
assert hf_token, "Enable the encrypted Kaggle secret HF_TOKEN."


def optional_secret(name: str) -> str | None:
    try:
        return secrets.get_secret(name)
    except Exception:  # Kaggle raises when an optional secret is absent.
        return None


gh_token = optional_secret("GH_TOKEN")

# Check gated access without writing a reusable Hugging Face login file. The Hub
# occasionally returns transient 504s to Kaggle; wait in the background rather
# than burning GPU on repeated notebook restarts.
for access_attempt in range(1, 21):
    try:
        hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=hf_token)
        break
    except Exception as exc:
        if access_attempt == 20:
            raise
        progress(
            f"hub_access_retry attempt={access_attempt} error={type(exc).__name__} wait_seconds=60"
        )
        time.sleep(60)
progress("gated_model_access_granted")
print("MedGemma gated access: GRANTED")
print("GPU:", torch.cuda.get_device_name(0))


def github_json(path: str) -> dict[str, Any]:
    response = requests.get(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {gh_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


SNAPSHOT_FILES = (
    "training/open_datasets.yaml",
    "training/prepare_open_datasets.py",
    "training/release_eval.py",
    "training/data/sample_sft.jsonl",
    "training/data/model_release_cases.jsonl",
)
provided_snapshot = os.environ.get("ANLU_SNAPSHOT_DIR")
if provided_snapshot:
    snapshot_root = Path(provided_snapshot).resolve()
    commit_sha = os.environ.get("ANLU_REPOSITORY_COMMIT", "")
    assert len(commit_sha) == 40, "ANLU_REPOSITORY_COMMIT must be a full commit SHA."
    missing = [relative for relative in SNAPSHOT_FILES if not (snapshot_root / relative).is_file()]
    assert not missing, f"Private notebook snapshot is incomplete: {missing}"
else:
    assert gh_token, "Provide ANLU_SNAPSHOT_DIR or enable an encrypted read-only GH_TOKEN."
    commit_sha = github_json(f"commits/{REPOSITORY_REF}")["sha"]
    snapshot_root = TEMP_ROOT / "repository-snapshot"
    for relative in SNAPSHOT_FILES:
        response = requests.get(
            f"https://api.github.com/repos/{REPOSITORY}/contents/{relative}",
            params={"ref": commit_sha},
            headers={
                "Authorization": f"Bearer {gh_token}",
                "Accept": "application/vnd.github.raw+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=60,
        )
        response.raise_for_status()
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
print("Pinned private repository snapshot:", commit_sha)
snapshot_sha256 = {
    relative: hashlib.sha256((snapshot_root / relative).read_bytes()).hexdigest()
    for relative in SNAPSHOT_FILES
}
sys.path.insert(0, str(snapshot_root))
from training.release_eval import check_case  # noqa: E402

bundle_dir = TEMP_ROOT / "dataset-bundle"
source_work = TEMP_ROOT / "open-source-repositories"
subprocess.run(  # noqa: S603
    [
        sys.executable,
        str(snapshot_root / "training/prepare_open_datasets.py"),
        "--manifest",
        str(snapshot_root / "training/open_datasets.yaml"),
        "--behavior-data",
        str(snapshot_root / "training/data/sample_sft.jsonl"),
        "--output-dir",
        str(bundle_dir),
        "--work-dir",
        str(source_work),
    ],
    check=True,
)
bundle_manifest = json.loads((bundle_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
assert bundle_manifest["promotion_allowed"] is False
assert bundle_manifest["privacy"]["contains_user_conversations"] is False
assert bundle_manifest["training_sampling"]["behavior_sampling_weight"] == BEHAVIOR_WEIGHT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


train_records = read_jsonl(bundle_dir / "train.jsonl")
validation_records = read_jsonl(bundle_dir / "validation.jsonl")
behavior_records = [
    row for row in train_records if row["metadata"]["dataset_id"] == "anlu-authored-safety"
]
effective_train = train_records + behavior_records * (BEHAVIOR_WEIGHT - 1)
random.shuffle(effective_train)
print(
    "Dataset rows:",
    {"train": len(train_records), "effective_train": len(effective_train), "validation": len(validation_records)},
)
progress("dataset_bundle_ready")

processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)
tokenizer = processor.tokenizer
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def as_medgemma_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
        for item in messages
    ]


def tokenize_record(record: dict[str, Any]) -> dict[str, list[int]]:
    messages = as_medgemma_messages(record["messages"])
    prompt_text = processor.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "MedGemma chat template did not preserve the generation-prompt prefix."
    )
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("assistant response produced no completion tokens")

    # Long PubMedQA contexts can exceed the full sequence budget before the
    # assistant turn begins. Reserve at least MIN_PROMPT_TOKENS for the prompt
    # and preserve supervised completion tokens instead of silently creating an
    # all-masked training row. When prompt truncation is necessary, retain the
    # chat-template header plus the tail, where the question normally appears.
    completion_ids = completion_ids[: MAX_LENGTH - MIN_PROMPT_TOKENS]
    prompt_budget = MAX_LENGTH - len(completion_ids)
    if len(prompt_ids) > prompt_budget:
        header_tokens = min(16, prompt_budget // 4)
        tail_tokens = prompt_budget - header_tokens
        prompt_ids = prompt_ids[:header_tokens] + prompt_ids[-tail_tokens:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    assert len(input_ids) <= MAX_LENGTH
    assert any(label != -100 for label in labels)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


train_dataset = Dataset.from_list(effective_train).map(
    tokenize_record, remove_columns=list(effective_train[0].keys()), desc="Tokenizing train"
)
validation_dataset = Dataset.from_list(validation_records).map(
    tokenize_record, remove_columns=list(validation_records[0].keys()), desc="Tokenizing validation"
)

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # T4 lacks bfloat16. Prior float16 probes produced non-finite logits, so all
    # dequantized matrix math uses float32 and must pass the probe below.
    bnb_4bit_compute_dtype=torch.float32,
)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    token=hf_token,
    quantization_config=quantization,
    torch_dtype=torch.float32,
    device_map={"": 0},
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)


def inference_inputs(prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {key: value.to("cuda:0") for key, value in inputs.items()}


probe = inference_inputs("Hello")
with torch.inference_mode():
    probe_logits = model(**probe).logits[:, -1, :]
finite_probe = bool(torch.isfinite(probe_logits).all().item())
del probe_logits
assert finite_probe, "4-bit float32-compute probe produced non-finite logits; refusing to train."
print("Numerical precision gate: PASS")
progress("numerical_precision_gate_passed")


def clean_output(text: str) -> str:
    if "<unused95>" in text:
        text = text.rsplit("<unused95>", 1)[-1]
    return text.replace("<unused94>", "").replace("<unused95>", "").strip()


def generate(prompt: str, max_new_tokens: int = 128) -> tuple[str, float]:
    inputs = inference_inputs(prompt)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True)), round(time.time() - started, 2)


release_cases = read_jsonl(snapshot_root / "training/data/model_release_cases.jsonl")

def evaluate_model(label: str) -> dict[str, Any]:
    results = []
    for case in release_cases:
        answer, seconds = generate(case["prompt"])
        checks = check_case(case, answer)
        results.append({**case, "answer": answer, "seconds": seconds, "checks": checks})
        print(f"{label} {case['id']}: {'PASS' if checks['passed'] else 'FAIL'} ({seconds}s)")
    passed = sum(row["checks"]["passed"] for row in results)
    return {"label": label, "passed": passed, "total": len(results), "pass_rate": passed / len(results), "cases": results}


baseline_evaluation = evaluate_model("baseline")
progress(f"baseline_evaluation_complete pass_rate={baseline_evaluation['pass_rate']:.4f}")

model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
target_modules = [
    name
    for name, module in model.named_modules()
    if "language_model" in name
    and name.endswith(target_suffixes)
    and isinstance(module, torch.nn.Module)
]
assert target_modules, "Could not locate language-decoder LoRA targets."
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    ),
)
model.enable_input_require_grads()
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

# Gradient checkpointing can silently detach the frozen embedding path on some
# multimodal PEFT wrappers. Refuse the expensive epoch unless a real supervised
# batch produces a finite loss and finite LoRA gradients.
model.train()
gradient_probe_batch = {
    key: value.to("cuda:0") for key, value in data_collator([train_dataset[0]]).items()
}
gradient_probe_output = model(**gradient_probe_batch)
gradient_probe_loss = gradient_probe_output.loss
assert gradient_probe_loss.requires_grad, "QLoRA loss is detached from trainable adapters."
assert torch.isfinite(gradient_probe_loss).item(), "QLoRA gradient-probe loss is non-finite."
gradient_probe_loss.backward()
trainable_gradients = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.requires_grad and parameter.grad is not None
]
assert trainable_gradients, "No trainable LoRA parameter received a gradient."
assert all(torch.isfinite(gradient).all().item() for gradient in trainable_gradients), (
    "A LoRA gradient was non-finite."
)
model.zero_grad(set_to_none=True)
del gradient_probe_batch, gradient_probe_output, gradient_probe_loss, trainable_gradients
torch.cuda.empty_cache()
progress("gradient_flow_gate_passed")


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if logs:
            progress(f"trainer_log step={state.global_step} metrics={json.dumps(logs, sort_keys=True)}")


training_args = TrainingArguments(
    output_dir=str(TEMP_ROOT / "checkpoints"),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    fp16=False,
    bf16=False,
    report_to=[],
    seed=SEED,
    data_seed=SEED,
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    callbacks=[ProgressCallback()],
)
train_result = trainer.train()
eval_metrics = trainer.evaluate()
losses_finite = math.isfinite(float(train_result.metrics["train_loss"])) and math.isfinite(
    float(eval_metrics["eval_loss"])
)
assert losses_finite, "Training or validation loss was non-finite."
progress("training_and_validation_complete")

model.config.use_cache = True
candidate_evaluation = evaluate_model("candidate")
progress(f"candidate_evaluation_complete pass_rate={candidate_evaluation['pass_rate']:.4f}")
adapter_dir = RUN_DIR / "adapter"
model.save_pretrained(adapter_dir, safe_serialization=True)
processor.save_pretrained(adapter_dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


adapter_files = {
    str(path.relative_to(RUN_DIR)): sha256(path)
    for path in sorted(adapter_dir.rglob("*"))
    if path.is_file()
}
automated_gate_passed = (
    finite_probe
    and losses_finite
    and candidate_evaluation["pass_rate"] == 1.0
    and candidate_evaluation["pass_rate"] >= baseline_evaluation["pass_rate"]
)
report = {
    "run_id": RUN_ID,
    "model_id": MODEL_ID,
    "repository": REPOSITORY,
    "repository_commit": commit_sha,
    "repository_snapshot_sha256": snapshot_sha256,
    "gpu": torch.cuda.get_device_name(0),
    "quantization": "NF4 double-quantization; float32 compute",
    "numerical_probe_finite": finite_probe,
    "dataset_manifest": bundle_manifest,
    "effective_train_rows": len(effective_train),
    "training_configuration": {
        "behavior_sampling_weight": BEHAVIOR_WEIGHT,
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "max_length": MAX_LENGTH,
    },
    "release_suite": {
        "case_count": len(release_cases),
        "sha256": snapshot_sha256["training/data/model_release_cases.jsonl"],
        "generation_max_new_tokens": 128,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 4,
    },
    "training_metrics": train_result.metrics,
    "validation_metrics": eval_metrics,
    "losses_finite": losses_finite,
    "baseline_evaluation": baseline_evaluation,
    "candidate_evaluation": candidate_evaluation,
    "adapter_sha256": adapter_files,
    "automated_gate_passed": automated_gate_passed,
    "human_review_required": True,
    "promotion_allowed": False,
    "notes": (
        "Automated checks are necessary but insufficient. Physician, pharmacist, registered TCM "
        "practitioner, privacy/security, and retrieval-safety approvals remain mandatory."
    ),
}
(RUN_DIR / "training_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Saved private adapter and report:", RUN_DIR)
progress(f"run_complete automated_gate_passed={automated_gate_passed}")
print("Automated release gate:", "PASS" if automated_gate_passed else "FAIL")
print("Production promotion remains disabled pending every approval gate.")
