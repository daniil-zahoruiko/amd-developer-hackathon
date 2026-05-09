# Local Setup Instructions

## 1. Create a ROCm GPU Droplet

Provision a machine with an AMD GPU and ROCm support enabled.

---

## 2. SSH Into the Machine with Port Forwarding

Gradio uses port `7860`, so forward it to your local machine:

```bash
ssh -A -L 7860:localhost:7860 root@<your-amd-ip>
```

---

## 3. Set Up a Python Virtual Environment

Install the Python venv package:

```bash
sudo apt install python3.12-venv
```

Create and activate a virtual environment inside the cloned repository:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 4. Install PyTorch for ROCm

```bash
pip3 install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

---

## 5. Install Project Requirements

```bash
pip3 install -r requirements.txt
```

---

## 6. Install TribeV2 (with Plotting Support)

Initialize git submodules:

```bash
git submodule update --init --recursive
```

Install TribeV2 in editable mode with plotting dependencies:

```bash
cd tribev2
pip3 install -e ".[plotting]"
```

---

## 7. Run the Application

Example:

```bash
python app.py
```

Once running, open:

```text
http://localhost:7860
```