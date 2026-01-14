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


📄Paper：[https://arxiv.org/abs/2508.07295](https://arxiv.org/abs/2508.07295)

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

## Download Model 
```
cd models
hf download Qwen/Qwen2.5-Omni-7B --local-dir Qwen2.5-Omni-7B
cd ..
```

## Download Test Split Data
```
hf download yxdu/MCGA MCGA_test.tar.gz --repo-type dataset --local-dir ./
tar -zxvf MCGA_test.tar.gz 
```


## VLLM Inference Server

```
cd eval
bash vllm_server.sh
cd ..
```

## Eval

```
cd eval
bash infer_model.sh
```




# 🖊Citation
```
```
