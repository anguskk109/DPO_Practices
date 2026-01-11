# DPO Practices
This repository documents my hands-on exploration of **Direct Preference Optimization (DPO)** for aligning language models with human preferences. Each practice focuses on a specific task, with emphasis on **realistic challenges**, **transparent results**, and **lessons learned**.

---

## 1st DPO Practice: Faithful Summarization  
**Goal:** Reduce hallucination (entities & numbers) in CNN/DailyMail summarization via DPO.

### 🔧 Project Details

#### **Dataset**
- **Source**: [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail) v3.0.0  
  - DPO pairs: 500 samples from `train` split  
  - Evaluation: 500 samples from `validation` split
- **Preference Pairs**:
  - ✅ **Chosen**: Human-written `highlights` (faithful by editorial design)
  - ❌ **Rejected**: Model-generated summaries with hallucinated facts
  - **Selection Logic**:  
    `badness = 2 × num_hallucinations + 1 × ent_hallucinations`  
    (Numbers weighted higher due to lower false-positive rate)

#### **Faithfulness Evaluation**
- **Metrics**:
  - **Entity Hallucination Rate**: % of summaries with entities absent from article + highlights
  - **Number Hallucination Rate**: % of summaries with numbers absent from article + highlights
- **Tools**:
  - **spaCy** (`en_core_web_sm`) for NER (with pipes disabled for speed)
  - **Regex** for number extraction: `\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b`
- **Robustness**:  
  Used **article + highlights** as reference to reduce false positives from NER misses in long articles.

#### **Training Setup**
- **Base Model**: [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B)
- **Key Configs**:
  - **Quantization**: 4-bit (`bitsandbytes`, NF4)
  - **LoRA**: `r=32`, `lora_alpha=64`, target modules = `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - **Batch**: `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`
  - **Context**: `max_length=1536` (preserves full article context)
  - **Optimizer**: AdamW, LR=5e-6, cosine decay, 2 epochs

### 📊 Results Comparison

| Metric | Base Model | Post-DPO | Δ |
|--------|------------|----------|-----|
| **Entity Hallucination Rate** | 50.2% | 48.8% | ↓1.4% |
| **Number Hallucination Rate** | 3.2% | 3.8% | ↑0.6% |
| **DPO Training Loss** | — | 0.135 | ↓80% |
| **Reward Accuracy** | — | 100% | — |

> 💡 **Note**: Despite perfect reward accuracy and low loss, **faithfulness did not improve**.

### 🔍 Key Insight
✅ **Training pipeline is flawless**  
❌ **Faithfulness alignment failed**  

**Root Cause**: **Preference pair quality**, not training mechanics.  
- DPO successfully learned to distinguish `chosen` vs `rejected`  
- But synthetic rejections introduced **confounding biases** (e.g., length, fluency, style)  
- Model optimized for **proxy signal**, not true faithfulness

> *"In real-world alignment, dataset curation is 80% of the battle."*

### 🧠 Lessons Learned
1. **Loss ≠ Target Metric**: Always validate on task-specific probes.
2. **Synthetic Pairs Are Risky**: Model-generated rejections may teach unintended behaviors.
3. **Precision > Recall in Signals**: A few high-quality pairs beat many noisy ones.
4. **Faithfulness Requires Causal Alignment**: Preference signal must isolate the target behavior.


---
## More Practices Coming Up...
