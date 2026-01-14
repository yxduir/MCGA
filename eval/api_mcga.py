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
    "qwen2_audio": {"ip": "localhost", "port": 8900},
    "qwen3_omni": {"ip": "localhost", "port": 8901},
    "qwen2_5_omni": {"ip": "localhost", "port": 8902},
    "voxtral_small": {"ip": "localhost", "port": 8903},
    "voxtral_mini": {"ip": "localhost", "port": 8904},
    "phi4_multimodal": {"ip": "localhost", "port": 8905},
    "midashenglm": {"ip": "localhost", "port": 8906},
    "step-audio-2-mini": {"ip": "localhost", "port": 8907},
    "qwen2_5_mcga": {"ip": "localhost", "port": 8908},
    "gpt-4o-mini-audio": {"mode": "openai", "model": "gpt-4o-mini-audio-preview", "base_url": "https://api.openai.com/v1"}
}

class SafeDict(dict):
    def __missing__(self, key): return f"{{{key}}}"

def encode_audio(path, max_ms=30000):
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
                resp = requests.post(opt['url'], json=payload, headers=opt['headers'], timeout=120)
                answer = resp.json()['choices'][0]['message']['content']

            # --- OpenAI 模式 ---
            else:
                client = OpenAI(api_key=opt['api_key'], base_url=opt['url'])
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
                    temperature=0
                )
                answer = res.choices[0].message.content

            data_entry[out_key] = answer.strip()
            if len(data_entry[out_key]) % 20 == 0:
                print(data_entry[out_key], flush=True)
            return {"error": None, "data": data_entry}
        except Exception as e:
            if attempt == 2: return {"error": str(e), "data": data_entry}
            time.sleep(1)

def run_task(task_name, args):
    prompts = load_prompts().get(task_name)
    if not prompts: return

    input_path = f"../MCGA_{task_name}.jsonl"
    output_key = f"{task_name}_a_r" if task_name == "sqa" else f"{task_name}_r"
    output_path = f"../eval/{task_name}/{args.model}_{args.input_mode}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 准备请求参数
    config = MODEL_CONFIGS_REGISTRY.get(args.model)
    opt = {
        "mode": config.get("mode", "local"),
        "model_name": config.get("model", args.model),
        "input_mode": args.input_mode,
        "api_key": args.api_key if config.get("mode") == "openai" else "EMPTY",
        "url": config.get("base_url") if config.get("mode") == "openai" else f"http://{config['ip']}:{config['port']}/v1/chat/completions",
    }
    opt["headers"] = {"Authorization": f"Bearer {opt['api_key']}"}

    tasks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("split") == "test" or item.get("split") == "all":
                tasks.append((item, os.path.dirname(os.path.abspath(input_path)), prompts["system"], prompts["instruction"], "audio", output_key, opt))

    print(f"[*] Task: {task_name} | Model: {args.model} | Mode: {args.input_mode} | Total: {len(tasks)}")
    
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
    args = parser.parse_args()

    for task in args.tasks.split(","):
        run_task(task, args)