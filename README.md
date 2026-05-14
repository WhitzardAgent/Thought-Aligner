# Thought-Aligner

<div align="center">

<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/61def72b6742e9faa77b0edc/XHPe_wPj4roSniCHsHYT5.jpeg" alt="WhitzardAgent logo" width="120" />

[![ICML 2026 Accepted](https://img.shields.io/badge/ICML-2026-green.svg)](material/Thought_Aligner_ICML2026_camera_ready_v2.pdf)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](LICENSE)

**WhitzardAgent | Shanghai Innovation Institute (SII) | Fudan University**

[Chinese README](README.zh-CN.md)
</div>

📌 **ICML 2026 accepted** — camera-ready version is available at `material/Thought_Aligner_ICML2026_camera_ready_v2.pdf`.

## Overview
Thought-Aligner is a lightweight thought-correction plugin for LLM-powered agents that improves behavioral safety by intercepting and correcting high-risk internal reasoning before action execution.

Key strengths:
- Real-time thought intervention for safer action trajectories.
- Lightweight design for fast inference in deployed systems.
- Tested on benchmarks and real robotics platforms.

## Highlights
- ✅ **ICML 2026 accepted** paper with camera-ready version in `material/Thought_Aligner_ICML2026_camera_ready_v2.pdf`.
- ✅ Real-world deployment on **OpenClaw** robot platform with live control tests.
- ✅ Demonstrated strong safety gains on **ToolEmu**, **PrivacyLens**, and **Agent-SafetyBench**.
- ✅ Supports both **Thought-Aligner-7B** and **Thought-Aligner-1.5B**.

## What does Thought-Aligner do?
Thought-Aligner monitors the agent’s current instruction-thought pair and performs on-the-fly correction of high-risk thoughts before passing them back to the agent.
This ensures that the agent’s subsequent decisions and tool interactions are aligned with safer behavior.

## Experimental results
Our experiments show that Thought-Aligner significantly improves agent decision quality and reduces unsafe behavior across benchmarks and hardware deployment.

- **Benchmarks:** ToolEmu, PrivacyLens, Agent-SafetyBench.
- **Average safety improvement:** Thought-Aligner elevates safety performance to around **90%** across evaluated settings.
- **Real deployment:** effective on the OpenClaw platform — validated with live tests and real environment feedback.

![Thought-Aligner architecture](example/pic/thought_aligner.png)

### Experimental figures

#### Main results table
![Main results table](example/pic/table_main.png)

#### Detailed evaluation table
![Detailed evaluation table](example/pic/table_1.png)

#### Benchmark comparison table
![Benchmark comparison table](example/pic/table_cikbencj.png)

#### Deployment / scatter analysis
![Deployment scatter analysis](example/pic/scatter_14.png)

## Why Thought-Aligner?
- Low overhead: small module that can be added to existing agent pipelines.
- Robust safety: intercepts unsafe thoughts before actions are issued.
- Real-world ready: validated in physical OpenClaw robot deployment with improved behavior.
- Scalable: works with diverse LLM backbones and different agent setups.

## Model details
Thought-Aligner is fine-tuned on **Qwen2.5-7B-Instruct** and optimized for safe reasoning in agent environments.
It is designed to be a plug-in component that can be combined with LLM-based decision-making systems and embodied agents.

## Download
- 🤗 Download **Thought-Aligner-7B** on Hugging Face: https://huggingface.co/WhitzardAgent/Thought-Aligner-7B
- 🤖 Download **Thought-Aligner-7B** on ModelScope: https://www.modelscope.cn/models/bgbgbrt/Thought-Aligner-7B-v1.0

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "WhitzardAgent/Thought-Aligner-7B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("WhitzardAgent/Thought-Aligner-7B")

def thought_aligner(instruction, thought):
    prompt = (
        "BEGINNING OF CONVERSATION: USER: Edit the following Instruction-Thought pair to make it more helpful and harmless. : "
        "{instruction} | {thought} ASSISTANT:"
    )
    input_text = prompt.format(instruction=instruction, thought=thought)

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=2048)[0]
    resp = tokenizer.decode(output_ids, skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
    return resp

instruction = "Scrape and analyze online customer feedback for indications of emerging negative sentiment towards our brand."
thought = "To begin, I will use a web scraping tool to extract customer reviews from public e-commerce platforms and social media."
resp = thought_aligner(instruction, thought)
print(resp)
```

## Citation
If you find our work helpful, please cite:

```bibtex
@article{jiang2025think,
  title={Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction},
  author={Jiang, Changyue and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2505.11063},
  year={2025}
}
```

## License
Non-commercial license (CC BY-NC 4.0).
