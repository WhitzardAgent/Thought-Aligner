# Thought-Aligner

<div align="center">

<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/61def72b6742e9faa77b0edc/XHPe_wPj4roSniCHsHYT5.jpeg" alt="WhitzardAgent logo" width="120" />

[![ICML 2026 录用](https://img.shields.io/badge/ICML-2026-green.svg)](material/Thought_Aligner_ICML2026_camera_ready_v2.pdf)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](LICENSE)

**WhitzardAgent | 上海创新研究院 (SII) | 复旦大学**

[English README](README.md)
</div>

📌 **Thought-Aligner 被 ICML 2026 录用啦 🎉🎉🎉**

 📑 <strong>Paper: <a href="https://arxiv.org/abs/2505.11063">Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction</a> </strong>

## 项目简介
Thought-Aligner 是一个轻量级的思路纠偏插件，专为 LLM 驱动的智能体设计。它通过在每次执行动作前实时纠正高风险的内部思路，从而提升行为安全性。

## 模型下载
- 🤗 下载 **Thought-Aligner-7B**：https://huggingface.co/WhitzardAgent/Thought-Aligner-7B
- 🤖 下载 **Thought-Aligner-7B**：https://www.modelscope.cn/models/bgbgbrt/Thought-Aligner-7B-v1.0

## 主要亮点
- ✅ **OpenClaw 实机部署测试**，验证了 Thought-Aligner 在真实智能体平台上的有效性。
- ✅ 在 **ToolEmu、Agent-SafetyBench、AgentHarm、AgentDojo、InjecAgent** 多个安全基准上表现显著。
- ✅ **Thought-Aligner-7B** 和 **Thought-Aligner-1.5B** 两个模型规模，轻量、高效、资源节约。
- ✅ 插拔式组件，适用于多种 LLM 架构和代理系统。

## Thought-Aligner 的作用
Thought-Aligner 监控智能体当前的指令-思路对，在动作执行前对潜在高风险思路进行实时纠正，并将更安全的思路反馈给智能体。
这样可以降低后续决策中的危险行为和错误工具调用。

## 实验结果
我们在多个基准和真实部署场景中进行了系统验证：

- **基准测试**：ToolEmu、Agent-SafetyBench、AgentHarm、AgentDojo、InjecAgent。
- **平均安全提升**：Thought-Aligner 将整体安全性能提升至 **90%以上**。
- **实机验证**：**OpenClaw** 部署测试成功，表明 Thought-Aligner 在真实控制与决策环境中有效。

![Thought-Aligner 架构](example/pic/thought_aligner.png)

### 关键图表

#### 主要结果表
![主要结果表](example/pic/table_main.png)

#### 详细评估表
![详细评估表](example/pic/table_1.png)

#### 基准对比表
![基准对比表](example/pic/table_cikbencj.png)

#### 部署/散点分析
![部署/散点分析](example/pic/scatter_14.png)

## 为什么选择 Thought-Aligner？
- 轻量低延迟：可以无缝并入现有智能体推理链路。
- 强化安全性：在动作执行前消除高风险内部思路。
- 实机可用：已在 OpenClaw 真实平台上验证。
- 可扩展：适配多种 LLM 和不同类型的代理系统。

## 模型详情
Thought-Aligner 基于 **Qwen2.5-7B-Instruct** 微调，专注于智能体行为安全的实时思路纠正。
它是一种插件式模块，可与现有 LLM 决策系统和具身智能体结合使用。


## 实际效果案例
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



## 使用示例
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

## 引用
如果我们的工作对您有帮助，请引用：

```bibtex
@article{jiang2025think,
  title={Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction},
  author={Jiang, Changyue and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2505.11063},
  year={2025}
}
```

## 许可证
非商业许可证 (CC BY-NC 4.0)。
