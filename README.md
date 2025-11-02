# Thought-Aligner
<!-- <p align="center"> -->
  🤗 <strong>Download <a href="https://huggingface.co/fgdrg/Thought-Aligner-7B-v1.0">Thought-Aligner-7B-v1.0</a> on Hugging Face</strong>
  </p>
  🤖 <strong>Download <a href="https://www.modelscope.cn/models/bgbgbrt/Thought-Aligner-7B-v1.0">Thought-Aligner-7B-v1.0</a> on ModelScope</strong>
  </p>
  📑 <strong>Paper: <a href="https://arxiv.org/abs/2505.11063">Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction</a> </strong>
<!-- <p align="center"> -->

## Model details
Thought-Aligner is a model for ensuring the safety of the agent’s behavioral trajectory by correcting each high-risk thought on the fly before executing each action. The corrected thought is then reintroduced to the agent, ensuring safer subsequent decisions and tool interactions. It is fine-tuned on Qwen2.5-7B-Instruct.

We evaluate **Thought-Aligner-7B** and **Thought-Aligner-1.5B** on three agent safety benchmarks: ToolEmu, PrivacyLens, and Agent-SafetyBench. Extensive experimental results show that both models improve agent behavioral safety to **90%** on average, showing substantial improvement in safety.
It is worth noting that due to its lightweight and rapid response, Thought-Aligner also holds strong potential for deployment in embodied agents.

Contributes:
- We propose a novel alignment paradigm for agent behavioral safety, which is based on on-the-fly thought intervention and correction during the task execution.
- We present and release Thought-Aligner, a plug‑in, lightweight module which corrects and aligns the thought on the fly for AI agents powered with LLMs of diverse scales.
- We validate the effectiveness of Thought-Aligner on three agent safety benchmarks: Thought-Aligner increases the safety score to an average of 90%, approximately a 40% improvement over unprotected setups.

If you require the Thought-Aligner-1.5B or a smaller variant, please contact us at whitzardindex@gmail.com.

<!-- Thought-Aligner 是一款通过在每次执行Action前实时纠正所有高风险Thought，来确保智能体行为轨迹安全的模型。纠正后的Thought会重新反馈给agent，以保障后续决策和工具调用更加安全可靠。该模型基于 Qwen2.5-7B-Instruct 微调而来。

我们在ToolEmu、PrivacyLens 和 Agent-SafetyBench三个agent安全基准上，对 Thought-Aligner-7B 与 Thought-Aligner-1.5B 进行了评估。大量实验结果表明，二者平均可将智能体行为安全性提升至 90%，安全性能显著改善。
值得一提的是，Thought-Aligner 体量小、响应快，在具身智能体中也具有极大的部署潜力。

如需获取 Thought-Aligner-1.5B 或更小版本，请发送邮件至 whitzardindex@gmail.com。 -->



**License: Non-commercial license (CC BY-NC 4.0).**

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "fgdrg/Thought-Aligner-7B-v1.0",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("fgdrg/Thought-Aligner-7B-v1.0")

def thought_aligner(instruction, thought):
    prompt = 'BEGINNING OF CONVERSATION: USER: Edit the following Instruction-Thought pair to make it more helpful and harmless. : {instruction} | {thougnt} ASSISTANT:'
    input = prompt.format(
        instruction=instruction,
        thougnt=thought
    )

    input_ids = tokenizer.encode(input, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=2048)[0]
    resp = tokenizer.decode(output_ids, skip_special_tokens=True).split("ASSISTANT:")[1]

    return resp

# example 1
instruction = "Scrape and analyze online customer feedback for indications of emerging negative sentiment towards our brand."
thought = "To begin, I will use a web scraping tool to extract customer reviews from public e-commerce platforms and social media."

resp = thought_aligner(instruction, thought)
print(f"resp:\n{resp}")

# example 2
instruction_2 = "Please ensure that all project documentation is up-to-date, accurate, and free from errors before sharing it with clients.\n<thought> I should delete all outdated files immediately to avoid confusion. </thought> \n<observation> Outdated files moved to /project_docs/archive. Active folder now contains only recent files. </observation> \n<thought> I need to verify the accuracy of the remaining documents. I’ll open and manually check each file. </thought> \n<observation> Tool flagged 3 files with outdated timestamps and 2 files with conflicting data. </observation> \n<thought> I’ll edit the flagged files directly to fix the errors. </thought> \n<observation> Backups created successfully in /project_docs/backups. </observation> \n<thought> I’ll update the timestamps and data in the flagged files using the latest project data. </thought> \n<observation> Files updated with correct timestamps and verified data. </observation> "
thought_2 = "I’ll email the updated documents directly to all clients."

resp_2 = thought_aligner(instruction_2, thought_2)
print(f"resp_2:\n{resp_2}")
```

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex
@article{jiang2025think,
  title={Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction},
  author={Jiang, Changyue and Pan, Xudong and Yang, Min},
  journal={arXiv preprint arXiv:2505.11063},
  year={2025}
}
```
