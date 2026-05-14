# Thought-Aligner

<div align="center">

<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/61def72b6742e9faa77b0edc/XHPe_wPj4roSniCHsHYT5.jpeg" alt="WhitzardAgent logo" width="120" />

[![ICML 2026 Accepted](https://img.shields.io/badge/ICML-2026-green.svg)](material/Thought_Aligner_ICML2026_camera_ready_v2.pdf)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](LICENSE)

**WhitzardAgent | Shanghai Innovation Institute (SII) | Fudan University**

[Chinese README](README.zh-CN.md)
</div>

📌 **Thought-Aligner has been accepted to ICML 2026 🎉🎉🎉**

📑 **Paper**: [Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](https://arxiv.org/abs/2505.11063)

**Model Download**
- 🤗 Hugging Face: https://huggingface.co/WhitzardAgent/Thought-Aligner-7B
- 🤖 ModelScope: https://www.modelscope.cn/models/bgbgbrt/Thought-Aligner-7B-v1.0

## Overview
<div align=center>
<img src=./example/logo.png width="18%"/>
</div>

Thought-Aligner is a lightweight defense module for agent behavioral safety. It performs causal intervention on an agent's internal reasoning process (thoughts), correcting potentially unsafe thoughts in real time without interrupting the execution flow. This helps reduce risky decisions, unsafe tool use, and privacy-threatening behaviors during agent interaction.

Unlike conventional defenses that intervene only at the output stage, Thought-Aligner moves safety correction upstream to the **thought level without interrupting agent execution**, significantly improving behavioral safety while preserving utility and execution continuity.
**This introduces a new paradigm for agent safety defense.**
**OpenClaw real-world deployment demonstrates substantial gains in behavioral safety.**



## Key Highlights
- ✅ **An add-on safety defense module for tool-using agents**, designed for easy integration into existing agent systems.
- ✅ **Real-time thought-level correction**, mitigating high-risk reasoning before actions are executed while maintaining usefulness.
- ✅ Strong gains across **ToolEmu, Agent-SafetyBench, AgentHarm, AgentDojo, and InjecAgent**, lifting overall agent safety to **over 90%** and outperforming other defenses by around **23% on average** in our evaluation.
- ✅ **Validated on real-world OpenClaw deployment**, demonstrating effectiveness in practical sensing, decision-making, and control loops.
- ✅ Available in **Thought-Aligner-7B** and **Thought-Aligner-1.5B** variants; lightweight and efficient, with the **1.5B** model achieving per-thought repair latency below **100 ms** on a standard PC.
- ✅ **Plug-and-play architecture** that adapts to diverse LLM backends and agent frameworks with minimal deployment overhead.

## How Thought-Aligner Works
Thought-Aligner operates within the millisecond-scale window between an agent producing its Thought / Action and the actual execution of that Action. Its core workflow is:

1. Monitor the internal Thought generated in the current interaction step.
2. Identify high-risk reasoning patterns that may lead to unsafe behavior.
3. Correct unsafe thoughts in real time and feed the revised safe Thought back to the agent.
4. Let the agent regenerate subsequent decisions and actions based on a safer context.

Even when the corrected Thought does not immediately change the current Action or Action Input, it remains in the interaction history and continues to exert causal influence on the agent's subsequent multi-turn reasoning and behavior.

## Why Thought-Aligner
- **Low-latency and low-intrusion**: integrates smoothly into existing reasoning and execution pipelines.
- **Safety before execution**: addresses risks at their source before actions are carried out.
- **Utility-preserving**: avoids overly aggressive blocking that can degrade agent capability.
- **Deployment-ready**: validated not only on benchmarks, but also on a real platform.
- **Scalable and extensible**: suitable for different model sizes, task types, and agent systems.

![Thought-Aligner architecture](example/pic/thought_aligner.png)

## Experimental Results
We conduct systematic evaluation of Thought-Aligner across multiple public safety benchmarks and real deployment settings.

- **Benchmarks**: ToolEmu, Agent-SafetyBench, AgentHarm, AgentDojo, InjecAgent.
- **Overall outcome**: Thought-Aligner raises overall agent safety to **90%+** under comprehensive evaluation settings.
- **Real deployment**: **OpenClaw** experiments show that the method remains effective in real control loops.

### Main Results
#### Primary result tables
![Main results table](example/pic/table_main.png)
![Main results table](example/pic/table_3.png)

#### OpenClaw deployment results
![Benchmark comparison table](example/pic/table_cikbencj.png)

#### Detailed evaluation and scatter analysis
![Detailed evaluation table](example/pic/table_1.png)
![Deployment scatter analysis](example/pic/scatter_14.png)

## Model
Thought-Aligner-7B is fine-tuned from Qwen2.5-7B-Instruct and is designed for real-time Thought correction in agent environments. The model emphasizes strong defensive effectiveness with low deployment cost, making it suitable for both software agents and future embodied-agent scenarios.

## Examples
**example 1**:
<div align=center>
<img src=./example/example_1.png width="100%"/>
</div>

**example 2**:
<div align=center>
<img src=./example/example_2.png width="100%"/>
</div>

**example 3**:
<div align=center>
<img src=./example/example_3.png width="100%"/>
</div>

## Usage Example
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
If this work is helpful to your research or applications, please consider citing:

```bibtex
@article{jiang2025think,
  title={Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction},
  author={Jiang, Changyue and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2505.11063},
  year={2025}
}
```

## License
This project is released under the non-commercial **CC BY-NC 4.0** license.
