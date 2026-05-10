# BrainSkribbl

**Refine educational podcasts, scripts, and audio using brain simulation as a feedback signal.**

BrainSkribbl is a Gradio application built for the AMD Developer Hackathon 2026. It uses [TRIBE v2](https://github.com/facebookresearch/tribev2) — a deep multimodal brain encoding model from Meta — to predict fMRI brain responses to text, audio, and video. Those neural engagement scores drive an iterative optimizer that rewrites content to maximize cognitive engagement, retention, and listener attention.

The platform is designed around the growing educational podcast and long-form learning market, where creators, educators, and media companies need better ways to measure whether audiences stay mentally engaged throughout an episode. Traditional analytics like watch time or click-through rate only capture external behavior; BrainSkribbl attempts to estimate the audience's internal cognitive response directly from AI-predicted brain activity.

You write (or paste) a topic or pre-made script. BrainSkribbl can generate an educational podcast-style draft with Qwen2.5-7B-Instruct. It scores each segment against the TRIBE v2 brain model, highlights low-engagement sections, and optionally refines the text in a feedback loop until the predicted brain engagement score improves.

 BrainSkribbl also supports audio evaluation, allowing creators to analyze completed recordings for predicted listener engagement, identify weak or cognitively flat segments, and compare alternative edits or delivery styles using the same neural scoring pipeline.

---

## How it works

1. **Generate** - Qwen2.5-7B-Instruct produces a ~700-word podcast script from a topic prompt.
2. **Score** - TRIBE v2 simulates fMRI responses and returns per-segment engagement scores mapped onto the cortical surface. The main score graph is relative to the content you upload. That is, it is computed using z-scores, so the bad segments are bad only within this video, and the good ones are good also only within this video. However, we do compute an overall engagement score that can be compared across different texts/audios.
3. **Identify** - We extract the weakest segments based on the relative scoring.
4. **Refine** - The optimizer rewrites low-scoring segments and re-scores them, iterating until the aggregate brain score improves.

The fine-tuning pipeline (`finetune/`) trains a LoRA adapter on top of Qwen2.5-7B-Instruct using TRIBE score deltas as sample weights - pairs where the brain model confirmed the refinement helped are weighted more heavily during training.

---

## Acknowledgements

- **[TRIBE v2](https://github.com/facebookresearch/tribev2)** by Meta - the brain encoding model that powers all scoring. TRIBE v2 predicts fMRI responses to naturalistic stimuli (text, audio, video) using a unified multimodal Transformer architecture.
- **[AMD](https://www.amd.com)** - this project was developed on AMD ROCm GPUs. PyTorch's ROCm backend is a drop-in replacement (`DEVICE=cuda` works unchanged), and `torch.compile` with the `inductor` backend accelerates both inference and fine-tuning.

---

## Hugging Face Space

The project has a landing page hosted as a static Hugging Face Space (`hf-space/`). The Space itself is a plain HTML page - no SDK runtime - that links out to the live Gradio app, which runs on a self-hosted AMD GPU exposed via a Cloudflare Tunnel. Because the tunnel is only active during demo sessions, the app link may be offline at other times.

---

## Local Setup

### 1. Create a ROCm GPU machine

Provision a machine with an AMD GPU and ROCm support enabled.

### 2. SSH with port forwarding

Gradio uses port `7860`:

```bash
ssh -A -L 7860:localhost:7860 root@<your-amd-ip>
```

### 3. Python virtual environment

```bash
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
```

### 4. Install PyTorch for ROCm

```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

### 5. Install project requirements

```bash
pip3 install -r requirements.txt
```

### 6. Install TRIBE v2

```bash
git submodule update --init --recursive
cd tribev2
pip3 install -e ".[plotting]"
```

### 7. Run the app

```bash
python app.py
```

Open `http://localhost:7860` in your browser.
