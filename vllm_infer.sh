#!/bin/bash


# --- 接收命令行参数并设置默认值 ---
# 语法: ${数字:-默认值}
MODEL_HF=${1:-"Qwen/Qwen2.5-Omni-7B"} # 第1个参数：模型地址
GPUS=${2:-"0"}                                       # 第2个参数：GPU编号
PORT=${3:-8901}                                      # 第3个参数：端口号
TASKS=${4:-"asr,s2tt,sec,sqa,su,sr"}                 # 第4个参数：任务列表
INPUT_MODE=${5:-"audio"}                             # 第5个参数：输入模式
WORKERS=${6:-16}                                     # 第6个参数：进程数
OPENAI_API_KEY=${7:-"sk-your-openai-key-here"}       # 第7个参数：API Key

# 派生变量
MODEL_NAME=${MODEL_HF##*/} # 自动截取仓库名作为模型名

echo "==== 配置信息 ===="
echo "Model: $MODEL_NAME"
echo "GPUs: $GPUS"
echo "Port: $PORT"
echo "Tasks: $TASKS"
echo "=================="

# --- 1. 下载模型 ---
echo "==== [1/5] 正在下载模型 ===="
hf download ${MODEL_HF} --local-dir models/${MODEL_NAME}

# --- 2. 下载并解压测试集 ---
echo "==== [2/5] 正在处理 MCGA 测试集 ===="
hf download yxdu/MCGA MCGA_test.tar.gz --repo-type dataset --local-dir ./

if [ -d "data" ]; then
    echo "检测到 data 目录已存在，跳过解压。"
else
    echo "正在解压测试集..."
    tar -zxvf MCGA_test.tar.gz
fi

# --- 3. 启动 vLLM 服务器并动态等待模型加载 ---
echo "==== [3/5] 正在后台启动 vLLM 服务器 ===="
cd eval
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

echo -e "\n[OK] 模型已完全加载，服务器就绪！"

# --- 4. 执行推理 ---
echo "==== [4/5] 正在开始推理任务 ===="
python3 api_mcga.py \
    --model $MODEL_NAME \
    --tasks $TASKS \
    --input_mode $INPUT_MODE \
    --workers $WORKERS \
    --api_key $OPENAI_API_KEY \
    --port $PORT \

# --- 5. 收尾 ---
echo "==== [5/5] 任务完成！ ===="
echo "正在关闭服务器 (PID: $SERVER_PID)..."
kill $SERVER_PID

echo "全部流程执行完毕。"