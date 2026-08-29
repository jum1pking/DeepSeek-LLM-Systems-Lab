import deepspeed
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

from common import DTYPE, MODEL_NAME

from common import PROJECT_ROOT


# Keep functional DeepSpeed validation results separate from comparable
# 20-step performance benchmarks.
DEEPSPEED_VALIDATION_CSV = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "deepspeed_validation.csv"
)

# Shared ZeRO-3 paths used by initialization and checkpoint-resume tests.

# Use the ZeRO-3 CPU-offload configuration for partitioned model initialization.
ZERO3_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "deepspeed"
    / "zero3_cpu_offload.json"
)

# Store a ZeRO-3 checkpoint to verify sharded training-state save and resume.
ZERO3_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "training"
    / "deepspeed_zero3_checkpoint"
)
def load_zero3_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Keep this object alive while from_pretrained() runs so Transformers
    # can perform ZeRO-3-aware checkpoint loading.
    hf_ds_config = HfDeepSpeedConfig(str(ZERO3_CONFIG_PATH))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer, hf_ds_config

def prepare_smoke_batch(tokenizer, device):
    messages = [
        {
            "role": "user",
            "content": "Explain in one sentence what distributed training is.",
        },
        {
            "role": "assistant",
            "content": (
                "Distributed training uses multiple devices to jointly train "
                "a model by splitting computation or model state."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    return {
        key: value.to(device)
        for key, value in batch.items()
    }