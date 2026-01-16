import json
import os
import base64
import time
import argparse
import io
import requests
from multiprocessing import Pool
from tqdm import tqdm
from openai import OpenAI
from pydub import AudioSegment

# --- 配置注册表 ---
MODEL_CONFIGS_REGISTRY = {
    "Qwen2-Audio-7B-Instruct": {"ip": "localhost", "port": 8900},
    "Qwen2.5-Omni-7B": {"ip": "localhost", "port": 8901},
    "Qwen3-Omni-30B-A3B-Instruct": {"ip": "localhost", "port": 8902},
    "Voxtral-Small-24B-2507": {"ip": "localhost", "port": 8903},
    "Voxtral-Mini-3B-2507": {"ip": "localhost", "port": 8904},
    "Phi-4-multimodal-instruct": {"ip": "localhost", "port": 8905},
    "midashenglm-7b-1021-bf16": {"ip": "localhost", "port": 8906},
    "Step-Audio-2-mini": {"ip": "localhost", "port": 8907},
    "qwen_omni_mcga": {"ip": "localhost", "port": 8908},
    "GPT-4o-mini-Audio": {
        "mode": "openai", 
        "model": "gpt-4o-mini-audio-preview", 
        "base_url": "https://api.openai.com/v1"
    }
}

class SafeDict(dict):
    def __missing__(self, key): return f"{{{key}}}"

def encode_audio(path, max_ms=30000):
    if args.model == "Step-Audio-2-mini":
        max_ms = 29900 # Step-Audio-2-mini 对于长度为 30s 的音频会超出范围，因此限制为 29.9s
    audio = AudioSegment.from_file(path)
    if len(audio) > max_ms: audio = audio[:max_ms]
    byte_io = io.BytesIO()
    audio.export(byte_io, format="wav") 
    return base64.b64encode(byte_io.getvalue()).decode('utf-8')

def load_prompts(file_path="mcga_prompt.jsonl"):
    configs = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): configs.update(json.loads(line))
    return configs

def process_line(args):
    data_entry, base_dir, system_p, raw_instr, audio_key, out_key, opt = args
    input_mode = opt['input_mode']
    
    # 构造 Prompt
    try: actual_instr = raw_instr.format_map(SafeDict(data_entry))
    except: actual_instr = raw_instr
    
    combined_prompt = f"文本: {data_entry.get('asr','')}\n任务: {actual_instr}" if input_mode == "text" else actual_instr

    for attempt in range(3):
        try:
            # --- Local 模式 (使用 requests 模拟 OpenAI 协议) ---
            if opt['mode'] == "local":
                if input_mode == "audio":
                    audio_b64 = encode_audio(os.path.join(base_dir, data_entry.get(audio_key)))
                    content = [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
                        {"type": "text", "text": combined_prompt}
                    ]
                else:
                    content = combined_prompt

                payload = {
                    "model": opt['model_name'],
                    "messages": [{"role": "system", "content": system_p}, {"role": "user", "content": content}],
                    "temperature": 0, "max_tokens": 2048,
                }
                # 使用拼接好的完整 URL (包含 /chat/completions)
                resp = requests.post(opt['full_url'], json=payload, headers=opt['headers'], timeout=120)
                answer = resp.json()['choices'][0]['message']['content']

            # --- OpenAI 模式 ---
            else:
                # SDK 只需要 base_url
                client = OpenAI(api_key=opt['api_key'], base_url=opt['base_url'])
                if input_mode == "audio":
                    audio_b64 = encode_audio(os.path.join(base_dir, data_entry.get(audio_key)))
                    user_content = [
                        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                        {"type": "text", "text": combined_prompt}
                    ]
                else:
                    user_content = combined_prompt

                res = client.chat.completions.create(
                    model=opt['model_name'],
                    messages=[{"role": "system", "content": system_p}, {"role": "user", "content": user_content}],
                    temperature=0, max_tokens=2048, timeout=120
                )
                answer = res.choices[0].message.content

            data_entry[out_key] = answer.strip()
            
            # --- 保持原有的打印功能 ---
            if len(data_entry[out_key]) % 10 == 0:
                print(data_entry[out_key], flush=True)
            if len(data_entry[out_key]) < 5:
                print(data_entry[out_key], flush=True)
                
            return {"error": None, "data": data_entry}
        except Exception as e:
            if attempt == 2: return {"error": str(e), "data": data_entry}
            time.sleep(1)

def run_task(task_name, args):
    prompts = load_prompts().get(task_name)
    if not prompts: return

    input_path = f"../data/MCGA_{task_name}_{args.split}.jsonl"
    output_key = f"{task_name}_a_r" if task_name == "sqa" else f"{task_name}_r"
    output_path = f"../eval/{task_name}/{args.model}_{args.input_mode}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = MODEL_CONFIGS_REGISTRY.get(args.model)
    mode = config.get("mode", "local")
    
    # --- 修正后的 URL 处理逻辑 ---
    # 优先级：命令行参数 > 注册表预置 > 默认值
    if mode == "openai":
        final_base_url = args.base_url if args.base_url != "https://api.openai.com/v1" else config.get("base_url", "https://api.openai.com/v1")
    else:
        # Local 模式
        target_port = args.port if args.port is not None else config.get("port")
        target_ip = args.ip if args.ip is not None else config.get("ip")
        
        # 如果命令行手动传了 base_url，则优先使用
        if args.base_url and args.base_url != "https://api.openai.com/v1":
            final_base_url = args.base_url
        else:
            final_base_url = f"http://{target_ip}:{target_port}/v1"

    opt = {
        "mode": mode,
        "model_name": config.get("model", args.model),
        "input_mode": args.input_mode,
        "api_key": args.api_key if mode == "openai" else "EMPTY",
        "base_url": final_base_url, 
    }
    
    # 针对 requests 需要完整路径，SDK 只需要 base_url
    if mode == "local":
        opt["full_url"] = f"{opt['base_url'].rstrip('/')}/chat/completions"
    else:
        opt["full_url"] = opt["base_url"]

    opt["headers"] = {"Authorization": f"Bearer {opt['api_key']}"}

    tasks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") == "test" or args.split == "all":
                tasks.append((item, os.path.dirname(os.path.abspath(input_path)), prompts["system"], prompts["instruction"], "audio", output_key, opt))

    print(f"[*] Task: {task_name} | Model: {args.model} | URL: {opt['base_url']} | Mode: {args.input_mode} | Split: {args.split}| Total: {len(tasks)}")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap(process_line, tasks), total=len(tasks)):
                if res["error"] is None:
                    f_out.write(json.dumps(res["data"], ensure_ascii=False) + "\n")
                else:
                    print(f"Error: {res['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CONFIGS_REGISTRY.keys())
    parser.add_argument("--tasks", type=str, default="asr,s2tt,sec,sqa,su,sr")
    parser.add_argument("--input_mode", type=str, choices=["audio", "text"], default="audio")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--api_key", type=str, default="sk-your-key")
    parser.add_argument("--ip", type=str, default="localhost", help="Manual override for the IP defined in REGISTRY")
    parser.add_argument("--port", type=int, default=None, help="Manual override for the port defined in REGISTRY")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="Manual override for the API base URL")
    parser.add_argument("--split", type=str, default="test")
    
    args = parser.parse_args()

    for task in args.tasks.split(","):
        run_task(task, args)