# train_dpo.py
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

def main():

    MODEL_NAME = "Qwen/Qwen2.5-1.5B"
    DATA_PATH = "outputs/cnn_dailymail_dpo_pairs.jsonl"
    OUTPUT_DIR = "dpo_output"


    data = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "prompt": f"Summarize the following news article accurately and concisely.\n\n{item['article']}\n\nSummary:\n",
                "chosen": item["preferred"].strip(),
                "rejected": item["rejected"].strip()
            })
    dataset = Dataset.from_list(data)
    print("Dataset size:", len(dataset))



    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    # Add LoRA
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dpo_config = DPOConfig(
        max_length=1536,              # total input+output length
        # max_prompt_length=768,        # article + prompt
        # max_completion_length=256,        # summary length

        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5.0e-6,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,

        #num_train_epochs=2,
        max_steps=130,

        dataloader_num_workers=2,

        save_strategy="steps",
        save_steps=50,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        # peft_config=peft_config,
    )

    dpo_trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()