#!/bin/bash

# --- 1. 基础配置 ---
# 默认全部任务 (asr, s2tt, sec, sqa, su, sr)
TASKS="asr,s2tt,sec,sqa,su,sr"
INPUT_MODE="text"
# INPUT_MODE="audio"

# 如果模式是 text，则只保留后面三个任务 (sqa, su, sr)
if [ "$INPUT_MODE" = "text" ]; then
    TASKS="sqa,su,sr"
fi

# INPUT_MODE="audio"
# OpenAI API Key (如果跑 GPT-4o-mini-Audio 模型需要填)
OPENAI_API_KEY="sk-your-openai-key-here"

# --- 2. 待测试的模型列表 ---
# 你可以根据需要注释掉不想跑的模型
MODELS=(
    "qwen2_5_omni"
    # "qwen2_audio"
    # “qwen3_omni”
    # "voxtral_mini"
    # "voxtral_small"
    # "phi4_multimodal"
    # "qwen2_5_mcga"
    # "step-audio-2-mini" 
    # "midashenglm"  # 这个模型会自动被设置为单进程处理，多进程目前有bug
    # "gpt-4o-mini-audio" # 如需测试 OpenAI 模式，取消注释，并确保 OPENAI_KEY 已设置
)

# --- 3. 运行逻辑 ---
echo "=========================================================="
echo "开始多模型评测任务"
echo "任务列表: $TASKS"
echo "输入模式: $INPUT_MODE"
echo "=========================================================="

for MODEL in "${MODELS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在启动模型: $MODEL ..."

    # 针对特定模型调整并行数 (例如midashenglm建议单进程)
    WORKERS=16
    if [ "$MODEL" == "midashenglm" ]; then
        WORKERS=1
        echo "  -> 并发数已调整为: $WORKERS"
    fi

    # 执行 Python 脚本
    # 结果输出到终端的同时，也保存到 logs 文件夹中
    mkdir -p logs
    python3 api_mcga.py \
        --model "$MODEL" \
        --tasks "$TASKS" \
        --input_mode "$INPUT_MODE" \
        --workers "$WORKERS" \
        --api_key "$OPENAI_API_KEY" \
        2>&1 | tee "logs/${MODEL}_${INPUT_MODE}.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 模型 $MODEL 测试完成。"
    echo "----------------------------------------------------------"
done

echo "所有评测任务执行完毕！"