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
    "Qwen/Qwen2.5-Omni-7B" \             #模型地址
    "0" \                                #GPU编号 0,1 | 0
    8901 \                               #VLLM端口号
    "asr,s2tt,sec,sqa,su,sr" \           #任务列表,适用audio和text
    "audio" \                            #audio ｜ text 
    16 \                                 #多进程并发api请求
    "sk-your-openai-key-here"            #可不填，用于评估GPT-4o-mini-Audio
```


<!-- ## Eval

```
cd eval
bash infer_model.sh
``` -->




# 🖊Citation
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
