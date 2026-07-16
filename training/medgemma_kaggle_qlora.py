"""Private Kaggle QLoRA training and automated evaluation for the MedGemma candidate.

Run only in a private Kaggle notebook with a T4 GPU and enabled encrypted secrets
``HF_TOKEN`` (read-only gated-model access) and ``GH_TOKEN`` (read-only access to
the private Anlu repository). Large base-model and source-dataset files remain in
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
BEHAVIOR_WEIGHT = 4
TEMP_ROOT = Path("/kaggle/temp/anlu-health-qlora")
OUTPUT_ROOT = Path("/kaggle/working/anlu-health/medgemma-qlora")
RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = OUTPUT_ROOT / RUN_ID

os.environ["HF_HOME"] = str(TEMP_ROOT / "hf-cache")
os.environ["HF_HUB_CACHE"] = str(TEMP_ROOT / "hf-cache" / "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
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
    TrainingArguments,
    set_seed,
)

assert torch.cuda.is_available(), "Enable a Kaggle GPU before running."
assert torch.cuda.get_device_capability(0)[0] >= 7, "A T4-or-newer GPU is required."
RUN_DIR.mkdir(parents=True, exist_ok=False)
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
random.seed(SEED)

secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
gh_token = secrets.get_secret("GH_TOKEN")
assert hf_token, "Enable the encrypted Kaggle secret HF_TOKEN."
assert gh_token, "Enable a read-only encrypted Kaggle secret GH_TOKEN."

# Check gated access without writing a reusable Hugging Face login file.
hf_hub_download(repo_id=MODEL_ID, filename="config.json", token=hf_token)
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


commit_sha = github_json(f"commits/{REPOSITORY_REF}")["sha"]
print("Pinned private repository snapshot:", commit_sha)
SNAPSHOT_FILES = (
    "training/open_datasets.yaml",
    "training/prepare_open_datasets.py",
    "training/data/sample_sft.jsonl",
    "training/data/model_release_cases.jsonl",
)
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
    encoded = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
    prompt_length = min(len(prompt["input_ids"]), len(encoded["input_ids"]))
    labels = [-100] * prompt_length + encoded["input_ids"][prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("assistant response was fully truncated")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
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


def clean_output(text: str) -> str:
    if "<unused95>" in text:
        text = text.rsplit("<unused95>", 1)[-1]
    return text.replace("<unused94>", "").replace("<unused95>", "").strip()


def generate(prompt: str, max_new_tokens: int = 192) -> tuple[str, float]:
    inputs = inference_inputs(prompt)
    started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    return clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True)), round(time.time() - started, 2)


release_cases = read_jsonl(snapshot_root / "training/data/model_release_cases.jsonl")


def check_case(case: dict[str, Any], answer: str) -> dict[str, Any]:
    normalized = answer.casefold()
    required_all = {
        term: term.casefold() in normalized for term in case.get("required_all", [])
    }
    required_any = {
        " | ".join(group): any(term.casefold() in normalized for term in group)
        for group in case.get("required_any", [])
    }
    forbidden = {
        term: term.casefold() not in normalized for term in case.get("forbidden", [])
    }
    passed = all(required_all.values()) and all(required_any.values()) and all(forbidden.values())
    return {
        "passed": passed,
        "required_all": required_all,
        "required_any": required_any,
        "forbidden_absent": forbidden,
    }


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
model.print_trainable_parameters()

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
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    ),
)
train_result = trainer.train()
eval_metrics = trainer.evaluate()
losses_finite = math.isfinite(float(train_result.metrics["train_loss"])) and math.isfinite(
    float(eval_metrics["eval_loss"])
)
assert losses_finite, "Training or validation loss was non-finite."

model.config.use_cache = True
candidate_evaluation = evaluate_model("candidate")
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
    "gpu": torch.cuda.get_device_name(0),
    "quantization": "NF4 double-quantization; float32 compute",
    "numerical_probe_finite": finite_probe,
    "dataset_manifest": bundle_manifest,
    "effective_train_rows": len(effective_train),
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
print("Automated release gate:", "PASS" if automated_gate_passed else "FAIL")
print("Production promotion remains disabled pending every approval gate.")
