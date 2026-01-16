#!/bin/bash


# --- 接收命令行参数并设置默认值 ---
# 语法: ${数字:-默认值}
MODEL_HF=${1:-"Qwen/Qwen2.5-Omni-7B"} # 第1个参数：模型地址
GPUS=${2:-"0"}                                       # 第2个参数：GPU编号
PORT=${3:-8901}                                      # 第3个参数：端口号
TASKS=${4:-"asr,s2tt,sec,sqa,su,sr"}                 # 第4个参数：任务列表
INPUT_MODE=${5:-"audio"}                             # 第5个参数：输入模式
SPLIT=${6:-"test"}                                 # 第6个参数：数据划分
WORKER=${7:-16}                                     # 第7个参数：进程数
IP=${8:-"localhost"}                             # 第8个参数：IP地址
OPENAI_API_KEY=${9:-"sk-your-openai-key-here"}       # 第9个参数：API Key
BASE_URL=${10:-"https://api.openai.com/v1"}        # 第10个参数：API 基础 URL
KILL_VLLM=${11:-"true"}                                # 第11个参数：是否关闭 vLLM 服务器   
EVAL=${12:-"true"}                                # 第11个参数：是否关闭 vLLM 服务器   


# 派生变量
MODEL_NAME=${MODEL_HF##*/} 

echo "==== 配置信息 ===="
echo "Model: $MODEL_NAME"
echo "GPUs: $GPUS"
echo "Port: $PORT"
echo "Tasks: $TASKS"
echo "=================="

# --- 1. 下载模型 ---
# 逻辑：如果是 GPT 系列API，或者 IP 不是 localhost，则跳过下载
if [[ $MODEL_NAME == *"GPT-4o-mini-Audio"* ]]; then
    echo "检测到模型为 ${MODEL_NAME}，将通过 API 进行请求，无需本地权重。"
    echo "==== 跳过模型下载 ===="
elif [[ "$IP" != "localhost" ]]; then
    echo "检测到 IP 为 ${IP} (非本地)，假定模型已在目标服务器运行，跳过本地下载。"
    echo "==== 跳过模型下载 ===="
else
    # 针对非 GPT 模型且在本地运行的下载逻辑
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
echo "==== [3/5] 服务器启动检查 ===="
cd eval

# 逻辑：如果是 GPT 模型 OR Step-Audio-2-mini OR IP 不等于 localhost，则跳过本地启动
if [[ $MODEL_NAME == *"GPT-4o-mini-Audio"* ]]; then
    echo "检测到模型为 ${MODEL_NAME}，将通过 API 进行请求，无需启动本地 vLLM 服务器。"
elif [[ $MODEL_NAME == *"Step-Audio-2-mini"* ]]; then
    echo "检测到模型为 ${MODEL_NAME}，将通过 Docker 启动，无需启动本地 vLLM 服务器。"
elif [[ "$IP" != "localhost" ]]; then
    echo "检测到目标 IP 为 ${IP}，将连接远程服务器，跳过本地 vLLM 启动步骤。"
else
    # --- 新增逻辑：先尝试检测端口是否已经可用 ---
    echo "检测端口 ${PORT} 是否已有正在运行的服务..."
    if [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)" == "200" ]; then
        echo "[SKIP] 端口 ${PORT} 已经响应 200，无需重复启动 vLLM 服务器。"
    else
        # 仅在端口不通时启动本地 vLLM
        echo "启动本地 vLLM 服务器，加载模型 ${MODEL_NAME} (端口: ${PORT}) ..."
        bash vllm_server.sh ${GPUS} ${PORT} ${MODEL_NAME} &
        SERVER_PID=$!

        echo "正在检测本地服务器及模型加载状态 (http://localhost:${PORT}/v1/models)..."
        TIMEOUT=300 # 模型加载时长上限
        ELAPSED=0

        # 核心逻辑：使用 curl 检查本地状态码是否为 200
        while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)" != "200" ]; do
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            echo -ne "本地 vLLM 模型加载中，已等待 ${ELAPSED}s... \r"
            if [ $ELAPSED -ge $TIMEOUT ]; then
                echo -e "\n错误: 本地模型加载超时。"
                kill $SERVER_PID
                exit 1
            fi
        done
        echo -e "\n[OK] 本地模型已完全加载，服务器就绪！"
    fi
fi


# --- 4. 执行推理 ---
echo "==== [4/5] 正在开始推理任务 ===="
python3 api_mcga.py \
    --model ${MODEL_NAME} \
    --tasks ${TASKS} \
    --input_mode ${INPUT_MODE} \
    --workers ${WORKER} \
    --api_key ${OPENAI_API_KEY} \
    --port ${PORT} \
    --ip ${IP} \
    --base_url ${BASE_URL} \
    --split ${SPLIT}

# --- 5. 收尾 ---
echo "==== [5/5] 任务完成！ ===="

# 逻辑：只有当 KILL_VLLM 为 true 且 IP 为 localhost 时才执行清理
if [[ "$KILL_VLLM" == "true" && "$IP" == "localhost" ]]; then
    
    # 使用 netstat 查找占用端口的 PID
    TARGET_PID=$(netstat -nlp | grep ":${PORT} " | awk '{print $7}' | cut -d'/' -f1)

    if [ -z "$TARGET_PID" ]; then
        echo "提示：未发现占用端口 ${PORT} 的进程，服务器可能已自行关闭。"
    else
        echo "检测到端口 ${PORT} 被进程 ${TARGET_PID} 占用，正在关闭 (KILL_VLLM=true)..."
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
else
    echo "==== 跳过服务器关闭步骤 ===="
    echo "原因: KILL_VLLM=$KILL_VLLM, IP=$IP"
fi


echo "==== [1/1] 评估模型！ ===="

if [ "${EVAL}" = "true" ]; then
    python eval_model.py \
        --model ${MODEL_NAME} \
        --mode ${INPUT_MODE}
    echo "模型评估完成"
else
    echo "跳过模型评估（EVAL设置为false）"
fi

echo "全部流程执行完毕。"
cd ..