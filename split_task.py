import json
import re
import os

def process_jsonl_by_task(input_file, output_file, task_type):
    """
    处理 JSONL：
    1. 过滤字段并按任务重命名。
    2. 只保留映射后 split 字段为 'test' 的数据。
    3. 按 id 中的数字 (如 1-10) 进行数值排序。
    """
    # --- 任务配置表 ---
    TASK_CONFIGS = {
        "asr": {
            "keep_keys": ["id", "asr", "audio", "source", "time", "prompt", "check_score", "asr_split", "dynasty", "genre", "sex"],
            "key_mapping": {"asr_split": "split"},
            "skip_if_missing": ["audio", "asr_split"]
        },
        "s2tt": {
            "keep_keys": ["id", "asr", "audio", "s2tt", "s2tt_split"],
            "key_mapping": {"s2tt_split": "split"}, 
            "skip_if_missing": ["s2tt", "s2tt_split"]
        },
        "sec": {
            "keep_keys": ["id", "asr", "audio", "sec_split", "sec_1", "sec_2", "sec_3", "sec"],
            "key_mapping": {"sec_split": "split"},
            "skip_if_missing": ["sec", "sec_split"]
        },
        "sqa": {
            "keep_keys": ["id", "asr", "audio", "sqa", "sqa_a", "sqa_split"],
            "key_mapping": {"sqa_split": "split"},
            "skip_if_missing": ["sqa", "sqa_split"]
        },
        "su": {
            "keep_keys": ["id", "asr", "audio", "su", "su_a", "su_split"],
            "key_mapping": {"su_split": "split"},
            "skip_if_missing": ["su", "su_split"]
        },
        "sr": {
            "keep_keys": ["id", "asr", "audio", "sr", "sr_a", "sr_split"],
            "key_mapping": {"sr_split": "split"},
            "skip_if_missing": ["sr", "sr_split"]
        }
    }

    if task_type not in TASK_CONFIGS:
        print(f"错误: 不支持的任务类型 '{task_type}'")
        return

    config = TASK_CONFIGS[task_type]
    keep_keys = config["keep_keys"]
    mapping = config.get("key_mapping", {})
    skip_if_missing = config["skip_if_missing"]

    results = []
    skipped_count = 0
    filtered_out_count = 0

    # 1. 读取并处理
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num} 行 JSON 格式错误，已跳过。")
                continue
            
            # 检查必要字段是否存在
            if not all(k in data for k in skip_if_missing):
                skipped_count += 1
                continue
            
            # 构造新字典并重命名 Key
            new_data = {}
            for k in keep_keys:
                if k in data:
                    new_key = mapping.get(k, k)
                    new_data[new_key] = data[k]
            
            # --- 核心修改：只保留 split 值为 "test" 的数据 ---
            # 此时的 'split' 是重命名后的键
            if new_data.get("split") == "test":
                results.append(new_data)
            else:
                filtered_out_count += 1

    # 2. 定义排序逻辑 (支持 id="1-1", "1-2" 等)
    def sort_key(item):
        id_val = str(item.get("id", "0-0"))
        parts = re.findall(r'\d+', id_val)
        return [int(p) for p in parts] if parts else [0]

    results.sort(key=sort_key)

    # 3. 写入输出文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in results:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"--- 任务 [{task_type.upper()}] 处理报告 ---")
    print(f"输出路径: {output_file}")
    print(f"成功保留 (split=test): {len(results)} 条")
    print(f"因缺少必要字段跳过: {skipped_count} 条")
    print(f"因非 test 集合过滤: {filtered_out_count} 条")
    print("-" * 30 + "\n")

if __name__ == "__main__":
    # 配置
    name = "MCGA"
    input_jsonl = f"./{name}.jsonl"
    
    # 执行所有子任务
    tasks = ['asr', 's2tt', 'sec', 'sqa', 'su', 'sr']
    
    for t in tasks:
        output_name = f"{name}_{t}.jsonl"
        process_jsonl_by_task(input_jsonl, output_name, t)