#!/bin/bash


# --- 接收命令行参数并设置默认值 ---
# 语法: ${数字:-默认值}
MODEL_HF=${1:-"Qwen/Qwen2.5-Omni-7B"} # 第1个参数：模型地址
GPUS=${2:-"0"}                                       # 第2个参数：GPU编号
PORT=${3:-8901}                                      # 第3个参数：端口号
TASKS=${4:-"asr,s2tt,sec,sqa,su,sr"}                 # 第4个参数：任务列表
INPUT_MODE=${5:-"audio"}                             # 第5个参数：输入模式
WORKER=${6:-16}                                     # 第6个参数：进程数
OPENAI_API_KEY=${7:-"sk-your-openai-key-here"}       # 第7个参数：API Key
BASE_URL=${8:-"https://api.openai.com/v1"}        # 第8个参数：API 基础 URL

# 派生变量
MODEL_NAME=${MODEL_HF##*/} 

echo "==== 配置信息 ===="
echo "Model: $MODEL_NAME"
echo "GPUs: $GPUS"
echo "Port: $PORT"
echo "Tasks: $TASKS"
echo "=================="

# --- 1. 下载模型 ---
# 检查模型是否为 GPT 系列 (通过 API 调用)
if [[ $MODEL_NAME == *"GPT-4o-mini-Audio"* ]]; then
    echo "检测到模型为 ${MODEL_NAME}，将通过 API 进行请求，无需本地权重。"
    echo "==== 跳过模型下载 ===="
else
    # 针对非 GPT 模型（开源本地模型）的下载逻辑
    if [ -d "models/${MODEL_NAME}" ]; then
        echo "检测到模型目录已存在，跳过下载。"
    else
        echo "==== [1/5] 正在下载模型 ===="
        hf download ${MODEL_HF} --local-dir models/${MODEL_NAME}
    fi
fi


# --- 2. 下载并解压测试集 ---
echo "==== [2/5] 正在处理 MCGA 测试集 ===="
if [ -f "MCGA_test.tar.gz" ]; then
    echo "检测到测试集压缩包已存在，跳过下载。"
else
    echo "正在下载测试集压缩包..."
    hf download yxdu/MCGA MCGA_test.tar.gz --repo-type dataset --local-dir ./
fi


if [ -d "data" ]; then
    echo "检测到 data 目录已存在，跳过解压。"
else
    echo "正在解压测试集..."
    tar -zxvf MCGA_test.tar.gz
fi

# --- 3. 启动 vLLM 服务器并动态等待模型加载 ---
echo "==== [3/5] 正在后台启动 vLLM 服务器 ===="
cd eval
if [[ $MODEL_NAME == *"GPT-4o-mini-Audio"* ]]; then
    echo "检测到模型为 ${MODEL_NAME}，将通过 API 进行请求，无需启动VLLM服务器。"
else
    echo "启动 vLLM 服务器，加载模型 ${MODEL_NAME} ..."
    bash vllm_server.sh ${GPUS} ${PORT} ${MODEL_NAME} &
    SERVER_PID=$!

    echo "正在检测服务器及模型加载状态 (端口: 8901)..."
    TIMEOUT=300 # 模型加载时长上限
    ELAPSED=0

    # 核心逻辑：使用 curl 检查状态码是否为 200
    while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)" != "200" ]; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo -ne "VLLM模型加载中，已等待 ${ELAPSED}s... \r"
        if [ $ELAPSED -ge $TIMEOUT ]; then
            echo -e "\n错误: 模型加载超时。"
            kill $SERVER_PID
            exit 1
        fi
    done
fi

echo -e "\n[OK] 模型已完全加载，服务器就绪！"

# --- 4. 执行推理 ---
echo "==== [4/5] 正在开始推理任务 ===="
python3 api_mcga.py \
    --model ${MODEL_NAME} \
    --tasks ${TASKS} \
    --input_mode ${INPUT_MODE} \
    --workers ${WORKER} \
    --api_key ${OPENAI_API_KEY} \
    --port ${PORT} \
    --base_url ${BASE_URL} 

# --- 5. 收尾 ---
# --- 5. 收尾 ---
echo "==== [5/5] 任务完成！ ===="

# 使用 netstat 查找占用端口的 PID
TARGET_PID=$(netstat -nlp | grep ":${PORT} " | awk '{print $7}' | cut -d'/' -f1)

if [ -z "$TARGET_PID" ]; then
    echo "提示：未发现占用端口 ${PORT} 的进程，服务器可能已自行关闭。"
else
    echo "检测到端口 ${PORT} 被进程 ${TARGET_PID} 占用，正在关闭..."
    
    # 先尝试正常关闭 (SIGTERM)
    kill $TARGET_PID
    
    # 等待 5 秒确认是否关闭成功
    for i in {1..5}; do
        # 再次检查端口是否还在监听
        if ! netstat -nlp | grep -q ":${PORT} "; then
            echo "[OK] 服务器已成功关闭。"
            break
        fi
        
        sleep 1
        
        if [ $i -eq 5 ]; then
            echo "警告：进程未响应，正在强制清理 (SIGKILL)..."
            kill -9 $TARGET_PID
        fi
    done
fi

echo "全部流程执行完毕。"
cd ..