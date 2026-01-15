import os
import json
import jiwer
import pandas as pd
import re
import string
import time
import numpy as np
import openai
from collections import Counter
from whisper.normalizers import BasicTextNormalizer
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from opencc import OpenCC  # 需要 pip install opencc-python-reimplemented
import re
import pandas as pd
# 禁用所有来自 httpx 的日志（openai 内部使用 httpx）
logging.getLogger("httpx").setLevel(logging.WARNING)

import re
from opencc import OpenCC  # 需要 pip install opencc-python-reimplemented

class ChineseCERNormalizer:
    def __init__(self):
        self.whisper_norm = BasicTextNormalizer()
        self.cc = OpenCC('t2s')  # 繁体转简体

    def __call__(self, text):
        if not text: return ""
        text = str(text)
        
        # 1. 繁体转简体
        text = self.cc.convert(text)
        
        # 2. 全角转半角 (针对数字和字母)
        text = text.translate(str.maketrans(
            '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ',
            '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        ))
        
        # 3. 执行 Whisper 基础规范化（会自动去掉大部分标点符号）
        text = self.whisper_norm(text)
        
        # 4. 关键修正：删除汉字（及中文符号）之间的所有空格
        # 匹配汉字范围: \u4e00-\u9fa5
        cn_pattern = r'([\u4e00-\u9fa5])\s+([\u4e00-\u9fa5])'
        while re.search(cn_pattern, text):
            text = re.sub(cn_pattern, r'\1\2', text)
            
        # 5. 最后一步：去掉文本中残留的所有空白符（包括换行、制表符、多余空格）
        # 这一步能确保 CER 真正只比对字符
        text = re.sub(r'\s+', '', text)
        
        return text

# --- 1. SEC 情感评估组件 ---
class EmotionEvaluator:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-chat" 
        self.normalizer = ChineseCERNormalizer()

    def _call_llm(self, prompt, retries=3):
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位严谨的语音情感分析助手，请严格按照评分规则进行判定。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    timeout=60
                )
                return response.choices[0].message.content
            except Exception as e:
                if i == retries - 1: return f"请求失败: {str(e)}"
                time.sleep(1)

    def evaluate_voice_persona(self, sec_1, sec_r_1):
        prompt = f"# 任务：语音画像一致性审核\n# 参考答案: \"{sec_1}\"\n# 待评估识别结果: \"{sec_r_1}\"\n\n## 评分算法\n- 满分: 2分\n- 维度1 [性别]: 若识别结果与标准性别一致，得1分，否则0分。\n- 维度2 [年龄]: 若识别结果与标准年龄段一致，得1分，否则0分。\n- 全错或无法识别：0分。\n\n## 输出要求\n请直接给出匹配结果判定（性别、年龄分别说明）及最终得分（格式：[得分/2]）。"
        return self._call_llm(prompt)

    def evaluate_global_emotion(self, sec_2, sec_r_2):
        prompt = f"# 任务：整体情感色彩分析\n# 参考答案: \"{sec_2}\"\n# 待评估识别结果: \"{sec_r_2}\"\n\n## 评分标准\n- 3分 (完全匹配): 识别结果完全捕获了标准中的核心情感词汇及整体意境。\n- 2分 (部分匹配): 识别出标准中一半以上的情感关键词或意境。\n- 1分 (基本不匹配): 情感描述非常微弱或有明显偏差。\n- 0分 (完全错误): 识别成其他无关作品、或描述完全不相干。\n\n## 输出要求\n请给出：匹配程度评价；最终得分（格式：[得分/3]）；"
        return self._call_llm(prompt)

    def evaluate_sentence_emotion(self, sec_3, sec_r_3, asr_ref):
        r3_lines = sec_r_3.strip().split('\n')
        hyp_segments = [line.split('|')[0].strip() if '|' in line else line.strip() for line in r3_lines]
        hyp_text = "".join(hyp_segments)
        cleaned_hyp, cleaned_ref = self.normalizer(hyp_text), self.normalizer(asr_ref)
        try: current_cer = jiwer.cer(cleaned_ref, cleaned_hyp)
        except: current_cer = 1.0 
        prompt = f"# 任务：逐句情感意境精准比对\n# 参考答案: \"{sec_3}\"\n# 待评估识别结果: \"{sec_r_3}\"\n\n## 评分规则\n- 初始满分: 5分\n- 扣分制: \n    - 每一句（或片段）的情感识别错误，扣1分。\n    - 错误超过4句，或者识别为其他文学作品，得0分。\n\n## 输出要求\n请以 Markdown 表格形式列出每句的匹配情况（句子编号、匹配状态、偏差分析），并在末尾给出最终总分（格式：[得分/5]）。"
        llm_result = self._call_llm(prompt)
        return llm_result + f"\n[ASR 准确率校验]\n- CER: {current_cer:.4f}", current_cer

# --- 2. Beauty 评估组件  ---
BEAUTY_PROMPTS = {
    "beauty_of_form": """/* Task prompt */Evaluate the translation of the given Chinese classical poem into English. Focus on whether the translation maintains consistency with the source poem's structure, including the alignment of line numbers and balanced phrasing. 1 point: Poor translation, disregards the poem's structure, and fails to convey its aesthetic qualities. 2 point: Some attempt to maintain structure but lack alignment and aesthetic consistency. 3 point: Basic structural elements are maintained but with noticeable imperfections in alignment and phrasing. 4 point: Good translation, with most structural elements preserved and minor issues in phrasing and alignment. 5 point: Excellent translation, accurately preserving the structure, alignment, and aesthetic qualities of the original poem./* Input Data */: Original Chinese poem: {source} English translation: {translation} Evaluation (score only):/*Output Text */: {score}""",
    "beauty_of_meaning": """/* Task prompt */Evaluate the translation of Chinese classical poetry for the beauty of meaning, focusing on whether the translation effectively conveys the themes, emotions, and messages of the original. This includes the use of concise and precise language to create vivid imagery and a rich atmosphere. 1 point: Poor translation, fails to convey the depth and richness of the original poetry. 2 point: Basic translation with significant shortcomings in capturing themes, emotions, and messages. 3 point: Satisfactory translation, conveys basic themes and emotions but lacks refinement or depth. 4 point: Good translation, effectively captures most themes, emotions, and messages with minor imperfections. 5 point: Excellent translation, accurately conveys the depth, richness, and atmosphere of the original poetry with full thematic and emotional resonance./* Input Data */: Original Chinese poem: {source} English translation: {translation} Evaluation (score only):/*Output Text */: {score}""",
    "beauty_of_sound": """/* Task prompt */Evaluate the beauty of sound in the given Chinese translation of classical poetry. Focus on whether the translation achieves harmonious sound, adherence to strict metrical rules, and a rhythm 1 point: Poor translation, lacks harmony and adherence to metrical rules, and fails to capture the beauty of sound. 2 point: Below average, some rhyme and meter present but with noticeable imperfections and awkwardness. 3 point: Basic translation, captures some aspects of sound beauty but with several imperfections in rhyme, meter, or rhythm. 4 point: Good translation, mostly harmonious with minor imperfections in sound quality or adherence to metrical rules. 5 point: Excellent translation, achieves harmonious sound, precise wording, strict adherence to metrical rules, and a smooth, dynamic rhythm./* Input Data */: Original Chinese poem: {source} English translation: {translation} Evaluation (score only):/*Output Text */: {score}"""
}

# --- 3. 辅助函数 ---
def extract_score(text):
    match = re.search(r'\[(\d+\.?\d*)/\d+\]', text)
    if match: return match.group(1)
    nums = re.findall(r'\d+', text)
    return nums[0] if nums else "0"

def _process_sec_line(args):
    line_data, api_key = args
    evaluator = EmotionEvaluator(api_key=api_key)
    data = line_data
    r_parts = data.get("sec_r", "").replace("\n\n","\n").split('\n', 2)
    
    # 获取三项评分的原始文本结果
    res1 = evaluator.evaluate_voice_persona(data.get("sec_1", ""), r_parts[0] if len(r_parts)>0 else "")
    res2 = evaluator.evaluate_global_emotion(data.get("sec_2", ""), r_parts[1] if len(r_parts)>1 else "")
    res3_text, cer_val = evaluator.evaluate_sentence_emotion(data.get("sec_3", ""), r_parts[2] if len(r_parts)>2 else "", data.get("asr", ""))
    
    # 提取数值分数
    s1 = extract_score(res1)
    s2 = extract_score(res2)
    s3 = extract_score(res3_text)
        
    # 更新数据字典
    data.update({
        "sec_r_c_1": res1, 
        "sec_r_c_2": res2, 
        "sec_r_c_3": res3_text, 
        "sec_score": {
            "cer": round(cer_val*100, 4), 
            "sec_1": s1, 
            "sec_2": s2, 
            "sec_3": s3,
        }
    })
    return data

def _process_beauty_item(args):
    item, metrics, api_key, src_key, trans_key = args

    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    source, translation = item.get(src_key, ''), item.get(trans_key, '')
    translation = translation.split("译文：")[-1].strip()
    scores = {}
    for metric in metrics:
        prompt = BEAUTY_PROMPTS[metric].format(source=source, translation=translation, score="")
        try:
            resp = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10)
            score_text = resp.choices[0].message.content.strip()
            scores[metric] = int(re.search(r'\d', score_text).group()) if re.search(r'\d', score_text) else None
        except: scores[metric] = None
    item['beauty_scores'] = scores
    return item

# --- 4. 主评估类 ---
class ModelEvaluator:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        self.normalizer = ChineseCERNormalizer()
        self.api_key = api_key

    def load_jsonl(self, folder):
        file_path = os.path.join(folder, f"{self.model_name}.jsonl")
        if not os.path.exists(file_path): return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _print_results(self, folder, metric_name, per_split, total):
        print(f"\n--- {folder.upper()} {metric_name} Results ---")
        for k in ['bad', 'other', 'train', 'val', 'test']:
            if k in per_split: print(f"{k}: {per_split[k]:.1f}")
        print(f"TOTAL: {total:.1f}\n" + "&".join([f"{per_split.get(k, 0.0):.1f}" for k in ['bad', 'other','train', 'val', 'test']] + [f"{total:.1f}"]))

    def run_asr_eval(self, folder, ref_key, hyp_key, splits_to_include=None, do_genre_analysis=True, do_dynasty_analysis=True):
        # 1. 加载数据
        data = self.load_jsonl(folder)
        if not data: return
        df = pd.DataFrame(data)
        
        if splits_to_include: 
            df = df[df['split'].isin(splits_to_include)]
            
        print(f" -> Computing {folder} CER...")

        def process_text(text, is_hyp=False):
            if not text or pd.isna(text): return ""
            text_str = str(text)
            if is_hyp:
                for sep in [":\n", ": '", ":'"]:
                    if sep in text_str:
                        text_str = text_str.split(sep)[-1]
                text_str = text_str[:100]
            norm_text = self.normalizer(text_str)
            return " ".join(list(norm_text))

        # --- 逻辑 A：全局与 Split 计算 ---
        r_all = [process_text(r) for r in df[ref_key]]
        h_all = [process_text(h, is_hyp=True) for h in df[hyp_key]]
        total = jiwer.cer(r_all, h_all) * 100

        per_split = {}
        for s, g in df.groupby('split'):
            g_r = [process_text(x) for x in g[ref_key]]
            g_h = [process_text(x, is_hyp=True) for x in g[hyp_key]]
            per_split[s] = jiwer.cer(g_r, g_h) * 100

        self._print_results(folder, "CER", per_split, total)

        # 标准排序定义
        DYNASTY_ORDER = ['先秦', '汉', '魏晋', '南北朝', '隋', '唐', '五代', '宋', '元', '明', '清']
        GENRE_ORDER = ['诗', '词', '曲', '赋', '文']

        # --- 逻辑 B: Genre 分析 (按 Split) ---
        if do_genre_analysis and 'genre' in df.columns:
            self._analyze_and_print_dimension(df, folder, ['split', 'genre'], "Genre", GENRE_ORDER, process_text, ref_key, hyp_key)

        # --- 逻辑 C: Dynasty-Genre 交叉分析 (不区分 Split) ---
        if do_dynasty_analysis:
            if 'dynasty' not in df.columns or 'genre' not in df.columns:
                print(f" [Warning] Missing 'dynasty' or 'genre' column, skipping cross analysis.")
            else:
                # 显式复制一份数据，避免 SettingWithCopyWarning
                df_cross = df.copy()
                df_cross['dynasty_genre'] = df_cross['dynasty'] + df_cross['genre']
                
                # 按照朝代纵向、体裁横向的逻辑生成排序序列
                # 只有当数据中确实存在该组合时才加入排序，保证打印出来的表格紧凑
                existing_combos = set(df_cross['dynasty_genre'].unique())
                combined_order = [
                    f"{d}{g}" 
                    for d in DYNASTY_ORDER 
                    for g in GENRE_ORDER 
                    if f"{d}{g}" in existing_combos
                ]
                
                # 调用时传入包含所有指定 split 的整体 df
                self._analyze_and_print_dimension(
                    df_cross, 
                    folder, 
                    ['dynasty_genre'], # 不传 split，实现“不区分内部切分”
                    "Dynasty-Genre (Total Pool)", 
                    combined_order, 
                    process_text, 
                    ref_key, 
                    hyp_key
                )

    def _analyze_and_print_dimension(self, df, folder, group_cols, label, custom_order, proc_fn, ref_key, hyp_key):
        """通用统计打印逻辑"""
        import jiwer
        import pandas as pd
        
        results = []
        # 1. 强制 group_cols 为列表，确保 groupby 行为一致
        grouped = df.groupby(group_cols)
        
        for keys, group_df in grouped:
            sub_r = [proc_fn(r) for r in group_df[ref_key]]
            sub_h = [proc_fn(h, is_hyp=True) for h in group_df[hyp_key]]
            
            if sub_r:
                sub_cer = jiwer.cer(sub_r, sub_h) * 100
                
                # --- 修正点 A: 改进 keys 的解析逻辑 ---
                if isinstance(keys, (list, tuple)):
                    if len(group_cols) == 1:
                        row_idx, col_val = "Total", str(keys[0]) # 确保是字符串
                    else:
                        row_idx, col_val = str(keys[0]), str(keys[1])
                else:
                    # 如果 keys 只是一个单纯的 string/int
                    row_idx, col_val = "Total", str(keys)
                
                results.append({
                    'Row': row_idx,
                    'Category': col_val,
                    'CER(%)': round(sub_cer, 2),
                    'Count': len(group_df)
                })

        if not results: return
        res_df = pd.DataFrame(results)
        
        # --- 修正点 B: 确保 custom_order 和 res_df['Category'] 格式完全一致 ---
        present_categories = res_df['Category'].unique().tolist()
        
        # 过滤并排序：只保留数据中存在的 custom_order 项
        final_col_order = [c for c in custom_order if c in present_categories]
        # 追加不在预设列表中的项
        final_col_order += [c for c in present_categories if c not in final_col_order]

        # 2. 生成透视表
        pivot_cer = res_df.pivot(index='Row', columns='Category', values='CER(%)')
        pivot_count = res_df.pivot(index='Row', columns='Category', values='Count')
        
        # --- 修正点 C: 重新应用排序 ---
        # 显式 reindex 是纠正字母序的关键
        pivot_cer = pivot_cer.reindex(columns=final_col_order)
        pivot_count = pivot_count.reindex(columns=final_col_order)

        print(f"\n >>> {label} Analysis for {folder} <<<")
        # 使用 to_csv 打印，na_rep 处理缺失组合
        print(f"CER (%) (& 分隔):")
        print(pivot_cer.to_csv(sep='&', index=True, float_format="%.1f", na_rep="-"))
        print(f"Count (条数) (& 分隔):")
        print(pivot_count.fillna(0).astype(int).to_csv(sep='&', index=True))

    def run_sqa_eval(self, folder, ref_key, hyp_key, splits_to_include=None):
        data = self.load_jsonl(folder)
        if not data: return
        df = pd.DataFrame(data)
        if splits_to_include: df = df[df['split'].isin(splits_to_include)]

        def f1(r, h):
            # 1. 清洗并分词（原逻辑）
            r_clean = re.sub(r'[^\w\s]', '', str(r)).lower()
            h_clean = re.sub(r'[^\w\s]', '', str(h)).lower()
            
            r_t, h_t = r_clean.split(), h_clean.split()

            # 2. 空值处理
            if not r_t or not h_t: 
                return 1.0 if r_t == h_t else 0.0

            # --- 新增逻辑：包含即视为完全正确 ---
            # 去掉空格后对比，确保即使空格位置稍有不同也能识别包含关系
            # 如果你希望严格按词匹配，可以改为: if ' '.join(r_t) in ' '.join(h_t):
            if "".join(r_t) in "".join(h_t):
                return 1.0
            # ----------------------------------

            # 3. 原有的词袋 F1 计算逻辑
            common = sum((Counter(r_t) & Counter(h_t)).values())
            p, rec = common/len(h_t), common/len(r_t)
            return (2*p*rec)/(p+rec) if (p+rec)>0 else 0.0

        df['f1'] = df.apply(lambda x: f1(x[ref_key], x[hyp_key]), axis=1)
        self._print_results(folder, "F1", {s: g['f1'].mean()*100 for s, g in df.groupby('split')}, df['f1'].mean()*100)



    def run_acc_eval(self, folder, ref_key, hyp_key, splits_to_include=None):
        data = self.load_jsonl(folder)
        if not data: return
        df = pd.DataFrame(data)
        if splits_to_include: df = df[df['split'].isin(splits_to_include)]

        def is_correct(ref, hyp):
            # 1. 预处理：转字符串、去空格、转大写
            ref_str = str(ref).strip().upper()
            hyp_str = str(hyp).strip().upper()

            # 2. 如果直接相等，返回 True
            if ref_str == hyp_str:
                return True

            # 3. 正则提取：针对 "A. 选项内容" 或 "答案是A" 这种格式
            # 提取第一个出现的 A-G 字母（通常是选项）
            match = re.search(r'\b([A-G])\b', hyp_str)
            if match:
                extracted_choice = match.group(1)
                if extracted_choice == ref_str:
                    return True

            # 4. 包含判定：如果处理后的标准答案在模型回复内
            # 先去掉所有标点符号，避免因为模型多写了句号导致不匹配
            ref_clean = re.sub(r'[^\w\s]', '', ref_str).replace(" ", "")
            hyp_clean = re.sub(r'[^\w\s]', '', hyp_str).replace(" ", "")
            
            if ref_clean and ref_clean in hyp_clean:
                return True

            return False

        # 应用新逻辑
        df['acc'] = df.apply(lambda x: is_correct(x[ref_key], x[hyp_key]), axis=1)
        
        self._print_results(
            folder, 
            "ACC", 
            {s: g['acc'].mean()*100 for s, g in df.groupby('split')}, 
            df['acc'].mean()*100
        )

    def run_sec_eval(self, folder, num_processes=256, force_run=False):
        # 1. 定义输出文件路径
        output_path = os.path.join(folder, f"{self.model_name}_sec.jsonl")
        results = []

        # 2. 逻辑判断：如果不强制推理，且本地文件已存在，则直接加载
        if not force_run and os.path.exists(output_path):
            print(f" -> Found existing results at {output_path}, loading local data...")
            # 这里建议直接手动读取，确保逻辑独立且稳健
            with open(output_path, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
        else:
            # 原始的推理逻辑
            data = self.load_jsonl(folder)
            if not data: return
            print(f" -> Running {folder} SEC Evaluation...")
            
            with Pool(num_processes) as p:
                results = list(tqdm(p.imap(_process_sec_line, [(d, self.api_key) for d in data]), total=len(data)))
            
            # 保存推理结果
            with open(output_path, 'w', encoding='utf-8') as f:
                for l in results: 
                    f.write(json.dumps(l, ensure_ascii=False)+'\n')

        # 3. 统计与打印逻辑（无论数据是读取的还是生成的，都走这里）
        if not results:
            print("No results to process.")
            return

        scores_list = [r['sec_score'] for r in results if 'sec_score' in r]
        if not scores_list:
            print("No valid SEC scores found in results.")
            return
            
        df_scores = pd.DataFrame(scores_list).astype(float)
                
        # 计算所有列的全局平均值
        avg = df_scores.mean()

        print(f"\n--- SEC Evaluation Results ---")
        if 'cer' in avg:
            print(f"CER: {avg['cer']:.2f}%")
        print(f"Persona (sec_1): {avg['sec_1']:.2f}")
        print(f"Global  (sec_2): {avg['sec_2']:.2f}")
        print(f"Sentence(sec_3): {avg['sec_3']:.2f}")
        sec_total = avg['sec_1'] + avg['sec_2'] + avg['sec_3']
        print(f"SEC Total      : {sec_total:.2f}")

        # 按照您的要求：CER 在最前面
        # 拼接顺序：CER & SEC1 & SEC2 & SEC3 & Total
        # print("\n拼接结果 (CER & SEC1 & SEC2 & SEC3 & Total):")
        
        print(f"{avg['cer']:.1f}&{avg['sec_1']:.1f}&{avg['sec_2']:.1f}&{avg['sec_3']:.1f}&{sec_total:.1f}")
        print(f"{avg['cer'] * 1:.1f}&{avg['sec_1'] * 10:.1f}&{avg['sec_2'] * 10:.1f}&{avg['sec_3'] * 10:.1f}&{sec_total * 10:.1f}")
    
    def run_s2tt_beauty_eval(self, folder, src_key, trans_key, num_workers=256, force_run=False):
    # 定义输出文件路径
        output_path = os.path.join(folder, f"{self.model_name}_beauty.jsonl")
        results = []
        metrics = ["beauty_of_form", "beauty_of_meaning", "beauty_of_sound"]

        # 逻辑判断：如果不强制推理，且本地文件已存在
        if not force_run and os.path.exists(output_path):
            print(f" -> Found existing results at {output_path}, loading local data...")
            with open(output_path, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
        else:
            # 原始的推理逻辑
            data = self.load_jsonl(folder)
            if not data: return
            print(f" -> Running {folder} Beauty LLM Evaluation...")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_beauty_item, (d, metrics, self.api_key, src_key, trans_key)) for d in data]
                for f in tqdm(as_completed(futures), total=len(data)): 
                    results.append(f.result())
            
            # 保存推理结果
            with open(output_path, 'w', encoding='utf-8') as f:
                for l in results: 
                    f.write(json.dumps(l, ensure_ascii=False)+'\n')

        # --- 以下是共用的分数计算逻辑，保持不变 ---
        if not results:
            print("No results to process.")
            return

        scores_df = pd.DataFrame([r['beauty_scores'] for r in results if r.get('beauty_scores')])
        if scores_df.empty:
            print("No valid scores found in results.")
            return

        print(f"\n--- S2TT_BEAUTY LLM Results ---")
        m_scores = [scores_df[m].mean() for m in metrics]
        total_avg = sum(m_scores) / len(m_scores)

        for i, m in enumerate(metrics):
            print(f"{m}: Mean {m_scores[i]:.2f}")
        print(f"Total Average: {total_avg:.2f}")
        
        all_scores = m_scores + [total_avg]
        # print("\n拼接结果 (Form & Meaning & Sound & Total):")
        print("&".join([f"{s:.1f}" for s in all_scores]))

        scores_x20 = [f"{s * 20:.1f}" for s in all_scores]

        # 3. 使用 & 拼接打印
        print("\n归一化结果 (x20, 100分制):")
        print("&".join(scores_x20))

        

# --- 执行入口 ---
import argparse
import os

# ... 保持现有的 import 不变 ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型评测脚本")
    
    # 定义命令行参数
    parser.add_argument("--model", nargs="+", default=["Qwen2.5-Omni-7B"], help="要测试的模型列表")
    parser.add_argument("--mode", type=str, default="audio", choices=["audio", "text"], help="模式：audio 或 text")
    parser.add_argument("--tasks", nargs="+", default=["asr", "s2tt", "sec", "sqa", "su", "sr"] ,help="执行的任务列表 (例如: asr s2tt sec sqa su sr)")
    parser.add_argument("--force_run", action="store_true",default=False,help="是否强制重跑")

    args = parser.parse_args()

    # 将参数赋值给原有变量名
    MODELS_TO_TEST = args.model.split("/")[-1]
    mode = args.mode
    force_run = args.force_run

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

    if args.tasks:
        TASKS = args.tasks
    else:
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.strip() == "deepseek-api-for-s2tt-sec-tasks-optional":
            TASKS = ["asr", "sqa", "su", "sr"]
            print("DEEPSEEK_API_KEY 未正确配置，跳过 s2tt 和 sec 任务评估。")
        else:
            TASKS = ["asr", "s2tt", "sec", "sqa", "su", "sr"]

    my_model = f"{MODELS_TO_TEST}_{mode}"

    evaluator = ModelEvaluator(MODELS_TO_TEST, api_key=DEEPSEEK_API_KEY)

    if "asr" in TASKS: evaluator.run_asr_eval('asr', 'asr', 'asr_r', ["train", "val", "test"], do_genre_analysis=True, do_dynasty_analysis=False)
    if "s2tt" in TASKS:evaluator.run_s2tt_beauty_eval(folder='s2tt', src_key='asr', trans_key='s2tt_r', num_workers=64, force_run=force_run)
    if "sec" in TASKS: evaluator.run_sec_eval('sec', num_processes=64, force_run=force_run)
    if "sqa" in TASKS: evaluator.run_sqa_eval('sqa', 'sqa_a', 'sqa_a_r', ["test"])
    if "su" in TASKS: evaluator.run_acc_eval('su', 'su_a', 'su_r', ["test"])
    if "sr" in TASKS: evaluator.run_acc_eval('sr', 'sr_a', 'sr_r', ["test"])