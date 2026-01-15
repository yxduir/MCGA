# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
MCGA (Multi-task Classical Chinese Literary Genre Audio Corpus) is the first large-scale, open-source, and fully copyrighted audio corpus dedicated to Classical Chinese Studies, comprising 119 hours (22,000 samples) of standard Mandarin recordings by native speakers that span five major literary genres (Fu, Shi, Wen, Ci, and Qu) across 11 historical periods, specifically constructed to support six core speech-centric tasksAutomatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Captioning(SEC), Spoken Question Answering(SQA), Speech Understanding(SU), Speech Reasoning(SR) to bridge the gap in domain-specific audio resources and advance the multidimensional capabilities of Multimodal Large Language Models.
- **Language**: Mandarin Chinese
- **Data Size**: 22,000 sample, 119hour
- **Data Split**: Train / Val / Test
- **Data Source**: Native speakers (13 males and 15 females)
- **Domain**: Classical Chinese Literary Study
- **Literary Genre**: Fu (Rhapsody), Shi (Poetry), Wen (Prose), Ci (Lyric), and Qu (Song)
- **Task**: ASR, S2TT, SEC, SQA, SU, SR
- **License**: CC BY-NC-SA-4.0

> **Note:** The **Test split** is released first for fair benchmarking.
> The full dataset will be available soon.

📄Paper：[https://arxiv.org/abs/2601.09270](https://arxiv.org/abs/2601.09270)

## How to use
```python
from datasets import load_dataset
MCGA = load_dataset("yxdu/MCGA")
print(MCGA)
```


## Installation
```
git clone https://github.com/yxduir/MCGA
cd MCGA

#We recommend using uv to set up the environment.
uv venv --python 3.10
source ./venv/bin/activate
uv pip install -r requirements.txt
```

## Download Model、Data、Inference
```
# 可添加支持VLLM框架的模型，例如：
# Qwen/Qwen2.5-Omni-7B  Qwen/Qwen3-Omni-30B-A3B-Instruct Qwen/Qwen2-Audio-7B-Instruct 
# mistralai/Voxtral-Small-24B-2507 mistralai/Voxtral-Mini-3B-2507 
# microsoft/Phi-4-multimodal-instruct

bash vllm_infer.sh \
    "mistralai/Voxtral-Mini-3B-2507" \
    "0" \
    8901 \
    "asr,s2tt,sec,sqa,su,sr" \
    "audio" \
    16 \
    "sk-your-openai-key-here"
```

| 参数位置 | 参数名称 | 说明 | 示例值 |
| :--- | :--- | :--- | :--- |
| **`$1`** | **Model HF** | 可添加支持VLLM框架的模型，例如：：<br>`Qwen/Qwen2.5-Omni-7B`, `Qwen/Qwen3-Omni-30B-A3B-Instruct`, `Qwen/Qwen2-Audio-7B-Instruct`<br>`mistralai/Voxtral-Small-24B-2507`, `mistralai/Voxtral-Mini-3B-2507`<br>`microsoft/Phi-4-multimodal-instruct` | `"Qwen/Qwen2.5-Omni-7B"` |
| **`$2`** | **GPUs** | 指定运行的 GPU 编号 (支持多卡，如 `"0,1"`) | `"0"` |
| **`$3`** | **Port** | vLLM 服务监听的端口号 | `8901` |
| **`$4`** | **Tasks** | 评测任务列表，多个任务用逗号分隔 | `"asr,s2tt,sec,sqa,su,sr"` |
| **`$5`** | **Mode** | 输入模态选择：`audio` (音频推理) 或 `text` (纯文本) | `"audio"` |
| **`$6`** | **Workers** | 并行发送 API 请求的进程数，建议根据 CPU 核数设置 | `16` |
| **`$7`** | **API Key** | 可选。仅在调用 OpenAI 模型（如 GPT-4o-Audio）时需要 | `"sk-xxxx"` |


<!-- ## Eval

```
cd eval
bash infer_model.sh
``` -->




## 🖊Citation
```
@misc{du2026mcgamultitaskclassicalchinese,
      title={{MCGA}: A Multi-task Classical Chinese Literary Genre Audio Corpus}, 
      author={Yexing Du and Kaiyuan Liu and Bihe Zhang and Youcheng Pan and Bo Yang and Liangyu Huo and Xiyuan Zhang and Jian Xie and Daojing He and Yang Xiang and Ming Liu and Bin Qin},
      year={2026},
      eprint={2601.09270},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.09270}, 
}
```
