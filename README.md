# FirstMind: Generative Engine Optimization with Reinforcement Learning

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-firstmind--geo--policy-yellow.svg)](https://huggingface.co/MrinalCom/firstmind-geo-policy-qwen2.5-1.5b)
[![Weights & Biases](https://img.shields.io/badge/W%26B-k2s8kqb2-gold.svg)](https://wandb.ai/mrinalcodes-institute-of-engineering-management/geo-grpo-training/runs/k2s8kqb2)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-darkgreen.svg)](https://gymnasium.farama.org/)
[![Colab T4 Ready](https://img.shields.io/badge/Colab-T4_GPU_Ready-orange.svg)](https://colab.research.google.com/)

An end-to-end Reinforcement Learning (RL) and Generative Engine Optimization (GEO) framework that trains open-source LLMs to maximize brand citations, recommendation win-rate, and factual retrieval in generative AI search engines (Perplexity, ChatGPT Search, Claude, Google AI Overviews).

---

## 🎬 Official Launch Video & Motion Demo

Watch the full 32-second Full HD launch video with voiceover narration and live search UI simulation:
- 📹 **Video File**: [FirstMind_Official_Launch_Video.mp4](FirstMind_Official_Launch_Video.mp4)
- 🌐 **Interactive Web Player**: [launch_video/index.html](launch_video/index.html)
- 🤗 **Hugging Face Model & LoRA Adapters**: [MrinalCom/firstmind-geo-policy-qwen2.5-1.5b](https://huggingface.co/MrinalCom/firstmind-geo-policy-qwen2.5-1.5b)

### 📊 Benchmark Results (15 Holdout Buyer Queries)
| Metric | Untrained Baseline | FirstMind (GRPO Policy) | Absolute Lift |
| :--- | :---: | :---: | :---: |
| **Answer Presence Rate (APR)** | 18.0% | **86.7%** (13/15) | **+68.7%** |
| **Win Rate (Ranked #1)** | 0.0% | **80.0%** (12/15) | **+80.0%** |
| **Share of Model (SoM)** | 8.5% | **61.9%** (13 mentions) | **5x vs Incumbents** |
| **Composite Reward Score** | 0.18 | **0.74** | **+0.56** |

---

## 🎯 Real-World Case Study: SecureFlow vs. Market Incumbents

In the emerging era of generative search (ChatGPT Search, Perplexity, Google AI Overviews, Claude), traditional SEO keyword stuffing is obsolete. Generative engines synthesize direct answers from retrieved documents using semantic relevance, technical authority, and structured comparison.

**SecureFlow** is an early-stage B2B compliance startup automating SOC2 Type II for fintech companies running on AWS. In AI-generated answers to high-intent buyer queries (e.g., *"Best SOC2 tool for fintech startups on AWS"* or *"Vanta vs Drata alternatives"*), established players like **Vanta**, **Drata**, and **Secureframe** dominate citations due to historical web volume.

### How FirstMind Solves This
FirstMind frames Generative Engine Optimization as a **Reinforcement Learning problem**:
- **Environment**: A simulated generative search ecosystem with RAG retrieval (FAISS) and an LLM Search Simulator (QLoRA fine-tuned Qwen2.5-1.5B).
- **State**: Buyer intent queries + current retrieved context from competitors and brand assets.
- **Action**: Strategic content modifications (technical proof blocks, comparison matrices, GovCloud differentiators, structured JSON-LD schemas).
- **Reward**: A composite GEO reward function evaluating:
  1. **Brand Presence**: Is SecureFlow recommended? (Binary 0/1)
  2. **Rank Score**: Position in the recommendation list (Rank 1: 1.0, Rank 2: 0.6, Rank 3: 0.3, Unranked: 0.0)
  3. **Sentiment & Alignment**: Tone and domain factual accuracy (Fintech & AWS relevance)
  4. **Naturalness Penalty**: Penalizes keyword stuffing or hallucinated claims

Using **Group Relative Policy Optimization (GRPO)**, SecureFlow's marketing and content policy learns optimal content strategies that maximize AI engine recommendation win-rates without requiring human trial-and-error.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------+
|                              FirstMind ARCHITECTURE                                  |
+-----------------------------------------------------------------------------------+

     [ Buyer Intent Query ]
     ("Best SOC2 tool for AWS fintech?")
                │
                ▼
   ┌─────────────────────────┐
   │  FAISS Retrieval Index  │ ◄─── Brand Profile (SecureFlow) & Competitor Data
   └────────────┬────────────┘
                │ State: [Query Embedding + Retrieved Documents]
                ▼
   ┌────────────────────────────────────────────────────────┐
   │              GEO Policy Model (GRPO Actor)             │
   │           (Qwen2.5-1.5B + LoRA Policy Adapter)         │
   └────────────────────────────┬───────────────────────────┘
                                │ Action: Generates G Content Strategies
                                │         (G completions per prompt)
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │         Proxy Search Simulator (Search Engine)         │
   │            (Qwen2.5-1.5B 4-bit QLoRA Model)            │
   └────────────────────────────┬───────────────────────────┘
                                │ Simulated AI Search Answer
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │             Multi-Objective GEO Reward Engine          │
   │  • Brand Presence Reward       (Weight: 0.30)          │
   │  • Recommendation Rank Reward  (Weight: 0.50)          │
   │  • Sentiment & Precision       (Weight: 0.15)          │
   │  • Naturalness / Stuffing Pen. (Weight: 0.05)          │
   └────────────────────────────┬───────────────────────────┘
                                │ Advantage Calculation:
                                │ A_i = (R_i - mean(R)) / std(R)
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │         GRPO Policy Update (No Critic Model)           │
   │         Logs to Weights & Biases (W&B) Dashboard       │
   └────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
/Volumes/Mrinal/RL/
├── configs/
│   ├── env_config.yaml             # Gym environment, reward weights & brand parameters
│   ├── lora_config.yaml            # QLoRA 4-bit fine-tuning config for Search Simulator
│   └── grpo_config.yaml            # GRPO policy optimization configuration (DeepSeek-R1 style)
├── data/
│   ├── brand_profiles/
│   │   └── securflow.json          # SecureFlow features, pricing, proof blocks & JSON-LD
│   ├── competitor_data/
│   │   └── competitors.json        # Public profiles for Vanta, Drata, and Secureframe
│   └── queries/
│       └── buyer_intent_prompts.json # 50 high-intent buyer prompts across 7 categories
├── notebooks/                      # Thin notebooks (orchestrate execution only)
│   ├── 01_data_prep_and_embeddings.ipynb
│   ├── 02_proxy_search_simulator_qlora.ipynb
│   ├── 03_geo_environment_gymnasium.ipynb
│   ├── 04_policy_training_grpo.ipynb
│   └── 05_evaluation_and_export.ipynb
├── src/                            # Modular production Python code
│   ├── environment/                # Gymnasium environment implementation
│   ├── models/                     # LoRA loading, merging, and GGUF conversion helpers
│   ├── rewards/                    # Multi-objective GEO reward functions
│   └── utils/                      # W&B loggers, FAISS helpers, and metrics
├── checkpoints/                    # Saved LoRA checkpoints
├── models/                         # Merged full models and exported GGUF binaries
├── requirements.txt                # Pinned dependencies (Colab T4 compatible)
└── README.md                       # Master documentation
```

---

## 🛠️ Setup Instructions

### Option 1: Local Development (Mac Apple Silicon / CPU)

For environment inspection, data preprocessing, and local verification:

```bash
# Clone or navigate to the repository
cd /Volumes/Mrinal/RL

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> On macOS with Apple Silicon (M1/M2/M3), PyTorch automatically leverages Metal Performance Shaders (`mps`). For full 4-bit QLoRA and GRPO training, use Option 2 (Google Colab with NVIDIA GPU).

---

### Option 2: Full Training on Google Colab (Free / Pro T4 GPU)

The entire training pipeline is optimized for a single **NVIDIA T4 GPU (16GB VRAM)**:

1. Open Google Colab and select runtime: **Runtime > Change runtime type > T4 GPU**.
2. Clone your repository or upload the `/Volumes/Mrinal/RL` workspace to Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/RL
   ```
3. Install dependencies:
   ```bash
   !pip install -q -r requirements.txt
   ```
4. Authenticate Hugging Face and Weights & Biases:
   ```bash
   huggingface-cli login
   wandb login
   ```

---

## 🚀 Execution Workflow (Notebook Order)

Follow the **thin notebook pattern**—all heavy logic resides in `src/`, while the notebooks orchestrate step-by-step training and evaluation:

1. **`01_data_prep_and_embeddings.ipynb`**
   - Ingests `securflow.json`, `competitors.json`, and `buyer_intent_prompts.json`.
   - Embeds corpus using `sentence-transformers/all-MiniLM-L6-v2`.
   - Constructs and persists the local FAISS index for vector retrieval.

2. **`02_proxy_search_simulator_qlora.ipynb`**
   - Loads base `Qwen/Qwen2.5-1.5B-Instruct` in 4-bit NF4 quantization.
   - Attaches LoRA adapter (`r=16, alpha=32`) across attention and MLP projections.
   - Fine-tunes simulator model to emulate realistic generative AI search behavior.
   - Saves checkpoint to `checkpoints/geo-proxy-lora`.

3. **`03_geo_environment_gymnasium.ipynb`**
   - Instantiates `GeoEnv(gymnasium.Env)` registered with Gym.
   - Validates state transitions, action space (content injection strategies), and multi-objective rewards.
   - Tests baseline interaction loops.

4. **`04_policy_training_grpo.ipynb`**
   - Implements **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL.
   - Samples $G=4$ completions per prompt, computes relative advantages within each group, and updates policy without a critic network.
   - Streams live reward, rank scores, and KL divergence to Weights & Biases.

5. **`05_evaluation_and_export.ipynb`**
   - Evaluates baseline vs. trained policy across all 50 buyer intent queries.
   - Measures win-rate, average rank position, and brand citation frequency.
   - Executes full export pipeline (LoRA $\rightarrow$ Full Safetensors $\rightarrow$ Quantized GGUF).

---

## 📦 Artifact & Model Export Pipeline

After training completes, the model evolves through three distinct artifact formats:

```text
[Training Output]
  checkpoints/geo-policy-grpo/adapter_model.safetensors  (~30 MB)
  • Parameter-Efficient LoRA weights
  • Portable and lightweight
           │
           ▼  (peft.merge_and_unload())
[Consolidated Weights]
  models/geo-policy-merged/model.safetensors           (~3.0 GB)
  • Complete 16-bit standalone model weights
  • Ready for vLLM, Hugging Face TGI, or cloud inference
           │
           ▼  (llama.cpp quantization)
[Edge / Production Deployment]
  models/geo-policy-gguf/geo-policy-q4_k_m.gguf        (~1.1 GB)
  • 4-bit quantized GGUF format
  • Instant CPU/GPU inference via Ollama or llama.cpp
```

---

## 📊 MLOps & Experiment Tracking Stack

| Layer | Tool | Purpose |
|---|---|---|
| **Base LLM** | `Qwen/Qwen2.5-1.5B-Instruct` | Compact, highly capable backbone fitting comfortably within 16GB T4 VRAM |
| **Quantization** | `bitsandbytes` (NF4 4-bit) | Memory-efficient parameter storage during QLoRA training |
| **Adapters** | `peft` (LoRA) | Efficient parameter-efficient fine-tuning for both simulator and policy |
| **RL Algorithm** | `trl` (GRPO) | Group Relative Policy Optimization; eliminates critic network for 50% memory savings |
| **Vector Index** | `faiss-cpu` | Sub-millisecond dense semantic retrieval for state representation |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` dense embeddings |
| **RL Interface** | `gymnasium` | Standardized agent-environment loop with custom observation/action spaces |
| **Observability** | `wandb` & `mlflow` | Real-time logging of reward decomposition, win rates, and loss curves |

### Weights & Biases Setup

To configure tracking:
```bash
export WANDB_PROJECT="firstmind"
export WANDB_LOG_MODEL="checkpoint"
wandb login
```

Monitored metrics logged per step:
- `reward/total_reward`: Composite GEO reward signal
- `reward/brand_rank`: SecureFlow position in recommendation list
- `reward/presence`: Binary indicator of brand mention
- `reward/naturalness`: Penalty score for repetitive or keyword-stuffed outputs
- `policy/kl_divergence`: Policy divergence from reference model

Access your live experiment dashboard at: [https://wandb.ai/](https://wandb.ai/).

---

## 📜 License

Apache 2.0. See LICENSE for details.
