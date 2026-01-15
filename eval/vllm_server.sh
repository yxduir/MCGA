#!/bin/bash

# --- 1. 参数接收 (严格位置参数) ---
# 脚本 1 调用示例: bash vllm_server.sh "0" "8901" "Qwen2.5-Omni-7B"
GPUS=${1:-"0"}               # 显卡
PORT=${2:-"8901"}            # 端口
NAME=${3:-"Qwen2.5-Omni-7B"}   # 模型名

# 自动生成路径
PATH_MOD="../models/$NAME"

# 强制重启设为 true (自动化脚本建议始终为 true，防止端口占用导致卡死)
FORCE_RESTART=true

MODELS=(
    "true | $GPUS | $PORT | $NAME | $PATH_MOD"
)

# --- 2. 端口检查函数 ---
prepare_port() {
    local port=$1
    local model_name=$2
    local pid=$(lsof -t -i:$port -sTCP:LISTEN)
    
    # 只要端口被占用，就执行清理
    if [ -n "$pid" ]; then
        echo "发现端口 $port 已被 PID: $pid 占用，正在清理..."
        kill -9 $pid 2>/dev/null
        sleep 2
        return 1 # 清理后返回“准备好了”
    fi
    return 1 # 本来就是干净的
}

# --- 3. 循环启动逻辑 ---
for model_info in "${MODELS[@]}"; do
    IFS='|' read -r active gpus port name path <<< "$(echo $model_info | tr -d ' ')"
    
    if [ "$active" != "true" ]; then continue; fi

    # 准备端口
    prepare_port $port "$name"

    # 计算 TP 数量
    tp_size=$(echo $gpus | tr -cd ',' | wc -c)
    tp_size=$((tp_size + 1))

    echo "🚀 正在启动: $name (Port: $port, GPU: $gpus, TP: $tp_size)"
    
    # 使用 if [[ ... ]] 模糊匹配特征词
    if [[ "$name" == *"Qwen3-Omni"* ]]; then
        CUDA_VISIBLE_DEVICES=$gpus vllm serve "$path" --port $port --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 \
            --tensor-parallel-size $tp_size --served-model-name "$name" > "${name}.log" 2>&1 &

    elif [[ "$name" == *"Voxtral"* ]]; then
        CUDA_VISIBLE_DEVICES=$gpus vllm serve "$path" --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
            --tokenizer_mode mistral --config_format mistral --load_format mistral \
            --tensor-parallel-size $tp_size --served-model-name "$name" > "${name}.log" 2>&1 &

    elif [[ "$name" == *"Phi-4"* ]]; then
        CUDA_VISIBLE_DEVICES=$gpus vllm serve "$path" --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
            --max-model-len 131072 --limit-mm-per-prompt '{"audio":3,"image":3}' \
            --enable-lora --max-loras 2 --max-lora-rank 320 \
            --lora-modules speech=../models/Phi-4-multimodal-instruct/speech-lora vision=../models/Phi-4-multimodal-instruct/vision-lora \
            --tensor-parallel-size $tp_size --served-model-name "$name" > "${name}.log" 2>&1 &

    elif [[ "$name" == *"midasheng"* ]]; then
        CUDA_VISIBLE_DEVICES=$gpus python3 -m vllm.entrypoints.openai.api_server --model "$path" \
            --port $port --host 0.0.0.0 --dtype bfloat16 --max_model_len 4096 --trust_remote_code \
            --tensor-parallel-size $tp_size --served-model-name "$name" --enable-chunked-prefill false \ > "${name}.log" 2>&1 &

    else
        # 默认启动逻辑 (Qwen2.5-Omni-7B 会走这里)
        CUDA_VISIBLE_DEVICES=$gpus vllm serve "$path" --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
            --tensor-parallel-size $tp_size --served-model-name "$name" > "${name}.log" 2>&1 &
    fi
done

echo "---------------------------------------"
echo "✅ 启动指令发送完毕。监控日志: eval/${NAME}.log"