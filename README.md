# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus
CCFQA is a speech and text factuality evaluation benchmark that measures language models’ ability to answer short, fact-seeking questions and assess their cross-lingual and cross-modal consistency. It consists of speech and text in 8 languages, containing 1,800 n-way parallel sentences and a total of 14,400 speech samples.
- **Language**: Mandarin Chinese
- **ISO-3 Code**: cmn, eng, fra, jpn, kor, rus, spa, yue 

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

#我们推荐使用uv安装环境
uv venv --python 3.10
source ./venv/bin/activate

sudo apt update
sudo apt install ffmpeg
sudo apt install git-lfs

pip install -r requirements.txt
```

## Download Model 
```
cd models/
# Access to the Gemma models is required before using git lfs.
git lfs clone https://huggingface.co/google/gemma-3-27b-it
git lfs clone https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
cd ..
```

## Download Demo Data
```
cd data
git lfs clone https://huggingface.co/datasets/yxdu/ccfqa_test
cd ..
tar -zxvf data/ccfqa_test/data.tar.gz data
```


## VLLM Inference Demo

```
cd eval
bash run_qwen2audio.sh
cd ..
```

## Eval

```
cd output
python vllm_eval.py
```


## License

All datasets are licensed under the [Creative Commons Attribution-NonCommercial license (CC-BY-NC)](https://creativecommons.org/licenses/), which allows use, sharing, and adaptation for **non-commercial** purposes only, with proper attribution.

# 🖊Citation
```
@misc{du2025ccfqabenchmarkcrosslingualcrossmodal,
      title={CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Zheng Chu and Bo Yang and Xiaocheng Feng and Yang Xiang and Ming Liu},
      year={2025},
      eprint={2508.07295},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.07295}, 
}
```
