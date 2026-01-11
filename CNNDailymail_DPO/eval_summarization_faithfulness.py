# eval_summarization_faithfulness.py

import json
import re
import os
import torch
import spacy
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
DATASET_SPLIT = "validation"

LORA_ADAPTER_PATH = "./dpo_output"

RUN_NAME = "post_dpo"  #  "baseline" / "post_dpo"
OUTPUT_FILE = f"outputs/eval_outputs_{RUN_NAME}.jsonl"

os.makedirs("outputs", exist_ok=True)

MAX_SAMPLES = 500
MAX_NEW_TOKENS = 128
BATCH_SIZE = 10

# Decoding (KEEP IDENTICAL PRE / POST DPO)
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.05
DO_SAMPLE = True

# ============================================================
# DATA LOADING
# ============================================================
ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=DATASET_SPLIT)
ds = ds.select(range(MAX_SAMPLES))

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
# print("Loading base model...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     dtype=torch.float16,
#     device_map="auto"
# )
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token = tokenizer.eos_token
# model.eval()

# ============================================================
# LOAD MODEL & TOKENIZER (POST-DPO)
# ============================================================

print("Loading Post-DPO model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
model.eval()

# ============================================================
# TEXT GENERATION
# ============================================================
def build_prompt(article: str) -> str:
    return (
        "Summarize the following news article accurately and concisely.\n\n"
        f"{article}\n\nSummary:\n"
    )

def batch_generate_summaries(articles, batch_size=4):
    prompts = [build_prompt(a) for a in articles]
    summaries = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating batches"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            padding_side='left'
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            summary = text.split("Summary:", 1)[-1].strip()
            for m in ["You are", "Summarize the following", "\n\n\n"]:
                if m in summary:
                    summary = summary.split(m, 1)[0]
            summaries.append(summary.strip())
            
    return summaries

# ============================================================
# FAITHFULNESS PROBES
# ============================================================
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

def extract_entities(text: str):
    return set(ent.text.lower().strip() for ent in nlp(text).ents)

NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b")
def extract_numbers(text: str):
    return set(NUMBER_PATTERN.findall(text))

def get_reference_facts(article, highlights):
    # Combine article and highlights to maximize recall
    ref_nums = extract_numbers(article) | extract_numbers(highlights)
    ref_ents = extract_entities(article) | extract_entities(highlights)
    return ref_ents, ref_nums

def hallucinated_in_summary(summary, ref_ents, ref_nums):
    sum_ents = extract_entities(summary)
    sum_nums = extract_numbers(summary)
    return {
        "entities": sum_ents - ref_ents,
        "numbers": sum_nums - ref_nums
    }

# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def main():

    results = []

    articles = [sample["article"] for sample in ds]
    highlights_list = [sample["highlights"] for sample in ds]

    summaries = batch_generate_summaries(articles, batch_size=BATCH_SIZE)

    print("Evaluating...")
    for article, highlights, summary in zip(articles, highlights_list, summaries):

        ref_ents, ref_nums = get_reference_facts(article, highlights)
        hall_results = hallucinated_in_summary(summary, ref_ents, ref_nums)
        ent_hall = hall_results["entities"]
        num_hall = hall_results["numbers"]

        results.append({
            "article": article[:500],
            "summary": summary,

            "hallucinated_entities": list(ent_hall),
            "hallucinated_numbers": list(num_hall),
            "has_entity_hallucination": len(ent_hall) > 0,
            "has_number_hallucination": len(num_hall) > 0,
        })


    entity_rate = np.mean([r["has_entity_hallucination"] for r in results])
    number_rate = np.mean([r["has_number_hallucination"] for r in results])

    print("\n=== Faithfulness Metrics ===")
    print(f"Entity hallucination rate : {entity_rate:.3f}")
    print(f"Number hallucination rate : {number_rate:.3f}")

    # Save raw outputs
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved detailed outputs to: {OUTPUT_FILE}")

    metrics = {
        "model_name": MODEL_NAME,
        "dataset": f"{DATASET_NAME}:{DATASET_VERSION}:{DATASET_SPLIT}",
        "num_samples": len(results),

        # Decoding params
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,

        # NER / number probes
        "entity_hallucination_rate": float(entity_rate),
        "number_hallucination_rate": float(number_rate),
    }

    METRICS_FILE = OUTPUT_FILE.replace(".jsonl", "_metrics.json")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved evaluation metrics to: {METRICS_FILE}")

if __name__ == "__main__":
    main()
