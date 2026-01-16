#!/bin/bash
# MODELS=(
#     "true  | 0       | 8900 | Qwen2-Audio-7B-Instruct       | ../models/Qwen2-Audio-7B-Instruct"
#     "true  | 0       | 8901 | Qwen2.5-Omni-7B               | ../models/Qwen2.5-Omni-7B"
#     "true  | 0,1     | 8902 | Qwen3-Omni-30B-A3B-Instruct   | ../models/Qwen3-Omni-30B-A3B-Instruct"
#     "true  | 2,3     | 8903 | Voxtral-Small-24B-2507        | ../models/Voxtral-Small-24B-2507"
#     "true  | 1       | 8904 | Voxtral-Mini-3B-2507          | ../models/Voxtral-Mini-3B-2507"
#     "true  | 1       | 8905 | Phi-4-multimodal-instruct     | ../models/Phi-4-multimodal-instruct"
#     "true  | 2       | 8906 | midashenglm-7b-1021-bf16      | ../models/midashenglm-7b-1021-bf16"
#.    "true  | 0       | 8907 | Step-Audio-2-mini             | ../models/Step-Audio-2-mini"
#     "true  | 3       | 8908 | qwen_omni_mcga                | ../models/qwen_omni_mcga"
# )

MODELS=(
    "true | 0       | 8901 | Qwen2.5-Omni-7B     | ../models/Qwen2.5-Omni-7B"
)

# ------------------------------------------------------------------------------
# 2. 全局参数处理
# ------------------------------------------------------------------------------
FORCE_RESTART=true
if [[ "$1" == "-f" || "$1" == "--force" ]]; then
    FORCE_RESTART=true
    echo "模式: 🚀 强制重启 (将清理已占用的端口)"
else
    echo "模式: 🆗 普通启动 (跳过已占用或未激活的服务)"
fi

prepare_port() {
    local port=$1
    local model_name=$2
    local pid=$(lsof -t -i:$port -sTCP:LISTEN)
    
    if [ -n "$pid" ] || ps -ef | grep "vllm" | grep -q "port $port"; then
        if [ "$FORCE_RESTART" = true ]; then
            echo "正在清理 [$model_name] 端口 $port 相关的进程..."
            [ -n "$pid" ] && kill -9 $pid 2>/dev/null
            ps -ef | grep "vllm" | grep "port $port" | awk '{print $2}' | xargs -r kill -9 2>/dev/null
            sleep 2
            return 1 
        else
            return 0 # 占用且不强制，跳过
        fi
    else
        return 1 # 干净
    fi
}

# ------------------------------------------------------------------------------
# 3. 循环启动逻辑 (保留各模型独立超参数)
# ------------------------------------------------------------------------------
for model_info in "${MODELS[@]}"; do
    # 解析配置
    IFS='|' read -r active gpus port name path <<< "$(echo $model_info | tr -d ' ')"
    
    if [ "$active" != "true" ]; then continue; fi

    if prepare_port $port "$name"; then
        echo "⚠️  端口 $port 已被占用，跳过 $name。"
        continue
    fi

    tp_size=$(echo $gpus | tr -cd ',' | wc -c)
    tp_size=$((tp_size + 1))

    echo "🚀 正在启动 $name (GPU: $gpus, TP: $tp_size, Port: $port)..."
    
    # 根据模型名称匹配特有的超参数
    case $name in
        "qwen2_audio")
            CUDA_VISIBLE_DEVICES=$gpus vllm serve $path --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
                --tensor-parallel-size $tp_size --served-model-name $name > ${name}.log 2>&1 &
            ;;
        "qwen3_omni")
            CUDA_VISIBLE_DEVICES=$gpus vllm serve $path --port $port --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 \
                --tensor-parallel-size $tp_size --served-model-name $name > ${name}.log 2>&1 &
            ;;
        "voxtral_small" | "voxtral_mini")
            CUDA_VISIBLE_DEVICES=$gpus vllm serve $path --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
                --tokenizer_mode mistral --config_format mistral --load_format mistral \
                --tensor-parallel-size $tp_size --served-model-name $name > ${name}.log 2>&1 &
            ;;
        "phi4_multimodal")
            CUDA_VISIBLE_DEVICES=$gpus vllm serve $path --port $port --host 0.0.0.0 --dtype auto --trust-remote-code \
                --max-model-len 131072 --limit-mm-per-prompt '{"audio":3,"image":3}' --enable-lora --max-loras 2 --max-lora-rank 320 \
                --tensor-parallel-size $tp_size --served-model-name $name \
                --lora-modules speech=${path}/speech-lora vision=${path}/vision-lora > ${name}.log 2>&1 &
            ;;
        "midashenglm")
            CUDA_VISIBLE_DEVICES=$gpus python3 -m vllm.entrypoints.openai.api_server --model $path \
                --port $port --host 0.0.0.0 --dtype bfloat16 --max_model_len 4096 --trust_remote_code \
                --enable-chunked-prefill false --max-num-seqs 16 \
                --tensor-parallel-size $tp_size --served-model-name $name > ${name}.log 2>&1 &
            ;;
        *)
            CUDA_VISIBLE_DEVICES=$gpus vllm serve $path --port $port --host 0.0.0.0 --dtype bfloat16 --trust-remote-code \
                --tensor-parallel-size $tp_size --served-model-name $name > ${name}.log 2>&1 &
            ;;
    esac
done

echo "---------------------------------------"
echo "✅ 调度指令发送完毕。"
echo "可以使用 'nvidia-smi' 或 'ps -ef | grep vllm' 确认状态。"