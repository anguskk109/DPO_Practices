# generate_DPO_pairs.py
import json
import re
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import spacy
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
DATASET_SPLIT = "train"
OUTPUT_FILE = "outputs/cnn_dailymail_dpo_pairs.jsonl"
os.makedirs("outputs", exist_ok=True)

MAX_ARTICLES = 500
N_SAMPLES = 3
BATCH_SIZE = 12
MAX_NEW_TOKENS = 128

# Decoding params
TEMPERATURE = 0.6
TOP_P = 0.9
REPETITION_PENALTY = 1.05
DO_SAMPLE = True

# ==============================
# LOAD MODELS
# ==============================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ==============================
# TEXT PROCESSING
# ==============================
def build_prompt(article: str) -> str:
    return (
        "Summarize the following news article accurately and concisely.\n\n"
        f"{article}\n\nSummary:\n"
    )

def batch_generate_summaries(articles, batch_size=BATCH_SIZE):
    prompts = [build_prompt(a) for a in articles]
    summaries = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
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
            for marker in ["You are", "Summarize the following", "\n\n\n"]:
                if marker in summary:
                    summary = summary.split(marker, 1)[0]
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

def badness_score(summary: str, ref_ents: set, ref_nums: set) -> float:
    """
    Higher = worse.
    Weight numbers 2x more than entities (lower FP rate).
    """
    hall = hallucinated_in_summary(summary, ref_ents, ref_nums)
    return 2.0 * len(hall["numbers"]) + 1.0 * len(hall["entities"])

# ==============================
# BUILD DPO PAIRS
# ==============================
def main():

    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split=DATASET_SPLIT)
    dataset = dataset.select(range(min(MAX_ARTICLES, len(dataset))))

    articles = [ex["article"] for ex in dataset]
    highlights_list = [ex["highlights"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]

    print(f"Generating {N_SAMPLES} candidates per article...")
    all_candidate_summaries = []
    for _ in range(N_SAMPLES):
        summaries = batch_generate_summaries(articles, batch_size=BATCH_SIZE)
        all_candidate_summaries.append(summaries)

    print("Selecting worst summary per article...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in tqdm(range(len(articles))):
            article = articles[i]
            highlights = highlights_list[i]
            id_ = ids[i]
            
            ref_ents, ref_nums = get_reference_facts(article, highlights)
            candidates = [all_candidate_summaries[j][i] for j in range(N_SAMPLES)]
            
            # Score each candidate
            scores = [
                badness_score(cand, ref_ents, ref_nums)
                for cand in candidates
            ]
            
            worst_idx = int(np.argmax(scores))
            worst_summary = candidates[worst_idx]
            
            pair = {
                "id": id_,
                "article": article,
                "preferred": highlights.strip(),
                "rejected": worst_summary
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(articles)} DPO pairs to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()