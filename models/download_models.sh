model_config=Qwen/Qwen3-Omni-30B-A3B-Instruct
model_config=microsoft/Phi-4-multimodal-instruct
model_config=mispeech/midashenglm-7b-1021-bf16
model_config=Qwen/Qwen2-Audio-7B-Instruct
model_config=mistralai/Voxtral-Mini-3B-2507
model_config=mistralai/Voxtral-Small-24B-2507
model_config=stepfun-ai/Step-Audio-2-mini
model_config=Qwen/Qwen2.5-Omni-7B
model_dir=${model_config##*/}
hf download ${model_config} --local-dir ${model_dir}