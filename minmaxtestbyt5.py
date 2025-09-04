import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import List, Dict, Any, Tuple
import torch
import time
import sys
import numpy as np

try:
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
except ImportError:
    print("错误：请确保安装了transformers库：pip install transformers")
    print("同时需要安装protobuf和sentencepiece：pip install protobuf sentencepiece")
    exit(1)


def clear_proxies():
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    if sys.platform.startswith('win'):
        try:
            import winreg
            reg_path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, 'ProxyEnable', 0, winreg.REG_DWORD, 0)
        except Exception as e:
            print(f"警告: 无法修改Windows注册表代理设置: {e}")


clear_proxies()


class BF:
    def __init__(self, bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                 max_abs_diff, min_val, max_val, q, weight_file, weight_factor=1.0, weight_range=(0.0, 1.0)):
        self.bf_len = bf_len
        self.bf_num_hash_func = bf_num_hash_func
        self.bf_num_inter = bf_num_inter
        self.bf_step = bf_step
        self.max_abs_diff = max_abs_diff
        self.min_val = min_val
        self.max_val = max_val
        self.q = q
        self.weight_factor = weight_factor
        self.weight_min, self.weight_max = weight_range
        assert max_val > min_val
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
        self.weights = self.load_weights(weight_file)
        self.default_weight = 0.5

    def load_weights(self, weight_file):
        try:
            df = pd.read_csv(weight_file)
            print(f"成功加载权重文件: {weight_file}，包含 {len(df)} 条权重记录")
        except FileNotFoundError:
            print(f"错误: 权重文件 {weight_file} 未找到，程序退出")
            exit(1)

        weights = {}
        for index, row in df.iterrows():
            key = f"{row['original']}-{row['replacement']}"
            weighted_val = max(self.weight_min, min(self.weight_max, row['weight']))
            weights[key] = weighted_val
        return weights

    def _hash_value(self, val):
        hex_str1 = self.h1(val.encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(val.encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    def set_to_bloom_filter(self, val_set, original_set):
        k = self.bf_num_hash_func
        l = self.bf_len
        bloom_set = bitarray(l)
        bloom_set.setall(False)
        extra_positions = []
        extra_positions_with_chars = {}

        for val in val_set:
            int1, int2 = self._hash_value(val)
            for i in range(k):
                gi = (int1 + i * int2) % l
                bloom_set[gi] = True
                if val not in original_set:
                    extra_positions.append(gi)
                    if gi not in extra_positions_with_chars:
                        extra_positions_with_chars[gi] = []
                    extra_positions_with_chars[gi].append(val)

        return bloom_set, extra_positions, extra_positions_with_chars

    def create_AVa(self, original_bf, replaced_bf, original_text, replaced_text):
        AVa = [0.0] * self.bf_len

        for i in range(self.bf_len):
            if original_bf[i] == 1:
                AVa[i] = 1.0
            elif original_bf[i] == 0 and replaced_bf[i] == 1:
                base_weight = self.get_replacement_weight(original_text, replaced_text, i)
                adjusted_weight = base_weight * self.weight_factor
                AVa[i] = max(self.weight_min, min(self.weight_max, adjusted_weight))
            else:
                AVa[i] = 0.0

        return AVa

    def get_replacement_weight(self, original_text, replaced_text, position):
        char_pairs = self.find_char_pairs(original_text, replaced_text, position)

        if not char_pairs:
            return self.default_weight

        weight_sum = 0
        for orig_char, repl_char in char_pairs:
            key = f"{orig_char}-{repl_char}"
            weight_sum += self.weights.get(key, self.default_weight)

        return weight_sum / len(char_pairs)

    def find_char_pairs(self, original_text, replaced_text, position):
        qgrams_original = [original_text[i:i + self.q] for i in range(len(original_text) - self.q + 1)]
        qgrams_replaced = [replaced_text[i:i + self.q] for i in range(len(replaced_text) - self.q + 1)]

        responsible_qgrams_original = []
        responsible_qgrams_replaced = []

        for qg_orig in qgrams_original:
            int1, int2 = self._hash_value(qg_orig)
            for i in range(self.bf_num_hash_func):
                gi = (int1 + i * int2) % self.bf_len
                if gi == position:
                    responsible_qgrams_original.append(qg_orig)
                    break

        for qg_repl in qgrams_replaced:
            int1, int2 = self._hash_value(qg_repl)
            for i in range(self.bf_num_hash_func):
                gi = (int1 + i * int2) % self.bf_len
                if gi == position:
                    responsible_qgrams_replaced.append(qg_repl)
                    break

        char_pairs = []
        for qg_orig in responsible_qgrams_original:
            for qg_repl in responsible_qgrams_replaced:
                for i in range(min(len(qg_orig), len(qg_repl))):
                    if qg_orig[i] != qg_repl[i]:
                        char_pairs.append((qg_orig[i], qg_repl[i]))

        return char_pairs

    def calc_bf_sim(self, AVa, BVb):
        intersection_sum = 0
        union_sum = 0

        for i in range(self.bf_len):
            a_val = AVa[i]
            b_val = BVb[i]
            intersection_sum += min(a_val, b_val)
            union_sum += max(a_val, b_val)

        if union_sum == 0:
            return 0.0

        return (2.0 * intersection_sum) / union_sum


class ByT5QGramGenerator:
    def __init__(self, model_size=r"D:\实习\COMP1\ByT5", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"加载ByT5模型 ({model_size}) 到 {device}...")
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_size)
        self.model = T5ForConditionalGeneration.from_pretrained(model_size).to(device)
        self.device = device
        print("ByT5模型加载完成")

        if torch.cuda.is_available():
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
            self.inference_params = {
                "num_beams": 3,
                "num_return_sequences": 3,
                "max_length": 40,
                "temperature": 0.7
            }
        else:
            print("使用 CPU 进行推理，可能较慢")
            self.inference_params = {
                "num_beams": 2,
                "num_return_sequences": 2,
                "max_length": 30,
                "temperature": 0.8
            }

    def generate_variants(self, text: str, q: int = 2) -> List[str]:
        if not isinstance(text, str) or len(text) < q:
            return [text] if isinstance(text, str) else []

        original_qgrams = [text[i:i + q] for i in range(len(text) - q + 1)]

        input_text = f"generate similar qgrams: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                early_stopping=True, **self.inference_params
            )
        except Exception as e:
            print(f"生成变体时出错: {e}，使用原始文本")
            return original_qgrams

        variants = []
        for output in outputs:
            try:
                variant = self.tokenizer.decode(output, skip_special_tokens=True)
                if variant and variant != input_text:
                    variants.append(variant)
            except Exception as e:
                print(f"警告: 解码失败 - {e}")

        variant_qgrams = []
        for variant in variants:
            variant_qgrams.extend([variant[i:i + q] for i in range(len(variant) - q + 1)])

        all_qgrams = list(set(original_qgrams + variant_qgrams))
        return all_qgrams


def load_dataset(file_path: str, source_name: str) -> pd.DataFrame:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        df = pd.read_csv(file_path)
        df['source_file'] = source_name
        print(f"成功加载 {source_name}: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"错误：读取 {file_path} 时发生异常 - {str(e)}")
        raise


def random_sample_dataset(df: pd.DataFrame, sample_size: int = 1000, random_state: int = 9318) -> pd.DataFrame:
    if len(df) <= sample_size:
        print(f"数据集小于指定样本量，返回全部 {len(df)} 条记录")
        return df

    sampled_df = df.sample(n=sample_size, random_state=random_state)
    print(f"从 {len(df)} 条记录中随机抽取 {sample_size} 条记录")
    return sampled_df


def evaluate_weight_performance(records1, records2, weight_factors, similarity_thresholds,
                                bf_params, weight_file, weight_range=(0.0, 1.0)):
    print("\n开始权重参数量化分析...")
    results = []
    confusion_matrices = []
    all_similarities = []  # 新增：存储所有计算的相似度值用于范围验证

    required_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                        'right_title', 'right_authors', 'right_venue', 'right_year']

    print("预生成所有记录的AVa集合...")
    all_ava = []
    for rec in records1 + records2:
        original_text = " ".join(
            [str(rec['features'][col]) for col in required_columns if pd.notna(rec['features'][col])])
        replaced_text = " ".join(rec['val_set'])

        temp_bf = BF(
            bf_len=bf_params['bf_len'],
            bf_num_hash_func=bf_params['bf_num_hash_func'],
            bf_num_inter=bf_params['bf_num_inter'],
            bf_step=bf_params['bf_step'],
            max_abs_diff=bf_params['max_abs_diff'],
            min_val=bf_params['min_val'],
            max_val=bf_params['max_val'],
            q=bf_params['q'],
            weight_file=weight_file,
            weight_factor=1.0,
            weight_range=weight_range
        )

        ava = temp_bf.create_AVa(
            rec['original_bf'],
            rec['replaced_bf'],
            original_text,
            replaced_text
        )
        all_ava.append(ava)

    ava1 = all_ava[:len(records1)]
    ava2 = all_ava[len(records1):]

    total_combinations = len(weight_factors) * len(similarity_thresholds)
    combo_count = 0

    for weight_factor in weight_factors:
        adjusted_ava1 = [
            [min(weight_range[1], max(weight_range[0], val * weight_factor)) for val in ava]
            for ava in ava1
        ]
        adjusted_ava2 = [
            [min(weight_range[1], max(weight_range[0], val * weight_factor)) for val in ava]
            for ava in ava2
        ]

        for threshold in similarity_thresholds:
            combo_count += 1
            print(f"评估组合 {combo_count}/{total_combinations}: 权重因子={weight_factor}, 阈值={threshold}")

            y_true = []
            y_pred = []
            similarities = []  # 记录当前组合的所有相似度值

            for i, rec1 in enumerate(records1):
                for j, rec2 in enumerate(records2):
                    a_val = adjusted_ava1[i]
                    b_val = adjusted_ava2[j]

                    intersection_sum = sum(min(a, b) for a, b in zip(a_val, b_val))
                    union_sum = sum(max(a, b) for a, b in zip(a_val, b_val))
                    sim = (2.0 * intersection_sum) / union_sum if union_sum > 0 else 0.0
                    similarities.append(sim)

                    true_label = 1 if (rec1['id'] == rec2['id'] or
                                       (rec1['label'] == 1 and rec2['label'] == 1)) else 0
                    pred_label = 1 if sim >= threshold else 0

                    y_true.append(true_label)
                    y_pred.append(pred_label)

            # 保存当前组合的所有相似度值
            all_similarities.extend([(weight_factor, threshold, sim) for sim in similarities])

            try:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = f1_score(y_true, y_pred) if (precision + recall) > 0 else 0

                results.append({
                    'weight_factor': weight_factor,
                    'threshold': threshold,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                confusion_matrices.append({
                    'weight_factor': weight_factor,
                    'threshold': threshold,
                    'confusion_matrix': cm
                })

            except Exception as e:
                print(f"评估出错: {e}")
                continue

    return pd.DataFrame(results), confusion_matrices, all_similarities


def main():
    start_time = time.time()

    random.seed(9318)
    torch.manual_seed(9318)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9318)

    # 布隆过滤器参数
    bf_params = {
        'bf_len': 400,
        'bf_num_hash_func': 8,
        'bf_num_inter': 6,
        'bf_step': 1,
        'max_abs_diff': 20,
        'min_val': 0,
        'max_val': 100,
        'q': 2
    }

    # 权重参数设置
    weight_factor = 1.0
    weight_range = (0.0, 1.0)  # 严格限制权重范围

    # 仅测试极端权重因子（最小值和最大值）
    weight_factors_to_test = [0.2, 2.0]
    # 保留关键阈值进行验证
    similarity_thresholds = [0.8, 0.9]

    # 减小样本量加速验证
    sample_size = 200

    # 文件路径设置
    data_dir = r"D:\实习\COMP1\icip2"
    acm_test_path = os.path.join(data_dir, "dirty_dblp_acm", "test.csv")
    scholar_test_path = os.path.join(data_dir, "dirty_dblp_scholar", "test.csv")
    weight_file = r"D:\实习\COMP1\icip2\results\byt5_adapted_weights_from_test_unlimited.csv"
    result_dir = os.path.join(data_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # 初始化ByT5生成器
    byt5_generator = ByT5QGramGenerator(model_size=r"D:\实习\COMP1\ByT5")

    try:
        print("\n加载原始测试数据集...")
        acm_test_full = load_dataset(acm_test_path, "acm_test_full")
        scholar_test_full = load_dataset(scholar_test_path, "scholar_test_full")

        print(f"\n从测试集中随机抽取 {sample_size} 条记录（验证模式）...")
        acm_test_sampled = random_sample_dataset(acm_test_full, sample_size)
        scholar_test_sampled = random_sample_dataset(scholar_test_full, sample_size)

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 检查必要的列
    required_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                        'right_title', 'right_authors', 'right_venue', 'right_year']
    for df in [acm_test_sampled, scholar_test_sampled]:
        for col in required_columns:
            if col not in df.columns:
                print(f"错误：数据缺少必要的列 '{col}'")
                return

    # 初始化布隆过滤器
    new_bf = BF(
        **bf_params,
        weight_file=weight_file,
        weight_factor=weight_factor,
        weight_range=weight_range
    )

    print("\n正在生成布隆过滤器和AVa集合...")

    # 处理数据集的函数
    def process_file(data: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        processed_records = []
        start_time = time.time()
        total_records = len(data)

        for idx in range(total_records):
            try:
                row = data.iloc[idx]
                id_value = row['id']

                # 提取特征
                features = {col: row[col] for col in required_columns + ['source_file']}

                # 生成原始布隆过滤器
                all_features = []
                for col in required_columns:
                    value = row[col]
                    if pd.notna(value):
                        all_features.append(str(value))

                def _hash_value(val: Any) -> Tuple[int, int]:
                    h1 = hashlib.sha1(str(val).encode('utf-8')).hexdigest()
                    h2 = hashlib.md5(str(val).encode('utf-8')).hexdigest()
                    return int(h1, 16), int(h2, 16)

                def features_to_bloom_filter(features: List[Any]) -> bitarray:
                    bloom = bitarray(bf_params['bf_len'])
                    bloom.setall(False)
                    for feature in features:
                        int1, int2 = _hash_value(feature)
                        for i in range(bf_params['bf_num_hash_func']):
                            gi = (int1 + i * int2) % bf_params['bf_len']
                            bloom[gi] = True
                    return bloom

                original_bf = features_to_bloom_filter(all_features)

                # 生成替换后的布隆过滤器
                val_set = []
                original_set = []
                for col in required_columns:
                    value = row[col]
                    if pd.notna(value) and isinstance(value, str) and len(value) >= bf_params['q']:
                        qgrams = byt5_generator.generate_variants(value, q=bf_params['q'])
                        val_set.extend(qgrams)
                        original_qgrams = [value[i:i + bf_params['q']] for i in range(len(value) - bf_params['q'] + 1)]
                        original_set.extend(original_qgrams)

                val_set = list(set(val_set))
                original_set = list(set(original_set))

                replaced_bf, extra_positions, extra_positions_with_chars = new_bf.set_to_bloom_filter(
                    val_set, original_set)

                # 生成AVa集合
                original_text = " ".join([str(row[col]) for col in required_columns if pd.notna(row[col])])
                replaced_text = " ".join(val_set)

                AVa = new_bf.create_AVa(
                    original_bf,
                    replaced_bf,
                    original_text,
                    replaced_text
                )

                processed_records.append({
                    'id': id_value,
                    'row_index': idx,
                    'original_bf': original_bf,
                    'replaced_bf': replaced_bf,
                    'AVa': AVa,
                    'extra_positions': extra_positions,
                    'extra_positions_with_chars': extra_positions_with_chars,
                    'features': features,
                    'val_set': val_set,
                    'original_set': original_set,
                    'label': row.get('label', 0)
                })

            except Exception as e:
                print(f"警告: 处理记录 {idx} 时出错 - {e}")
                continue

            # 显示进度
            if (idx + 1) % max(1, total_records // 10) == 0:
                progress = (idx + 1) / total_records * 100
                elapsed = time.time() - start_time
                eta = elapsed * (total_records / (idx + 1) - 1)
                print(f"已处理 {source} 中的 {idx + 1}/{total_records} 条记录 ({progress:.1f}%), "
                      f"耗时: {elapsed:.2f}s, ETA: {eta:.2f}s")

        total_time = time.time() - start_time
        print(f"{source} 处理完成，共耗时: {total_time:.2f}s")
        return processed_records

    # 处理两个测试集
    print("\n处理抽样后的ACM测试集...")
    acm_test_records = process_file(acm_test_sampled, 'acm_test_sampled')

    print("\n处理抽样后的Scholar测试集...")
    scholar_test_records = process_file(scholar_test_sampled, 'scholar_test_sampled')

    # 执行评估，获取结果和所有相似度值
    analysis_results, confusion_matrices, all_similarities = evaluate_weight_performance(
        acm_test_records,
        scholar_test_records,
        weight_factors_to_test,
        similarity_thresholds,
        bf_params,
        weight_file,
        weight_range
    )

    # 关键验证：检查所有相似度值是否在0-1之间
    if all_similarities:
        # 提取所有相似度值
        sim_values = [sim for (wf, th, sim) in all_similarities]
        sim_min = min(sim_values)
        sim_max = max(sim_values)
        out_of_range = [(wf, th, sim) for (wf, th, sim) in all_similarities if sim < 0 or sim > 1]

        print(f"\n=== 相似度范围验证 ===")
        print(f"所有相似度值范围: [{sim_min:.6f}, {sim_max:.6f}]")
        if not out_of_range:
            print("✅ 验证通过：所有相似度值均在0-1之间")
        else:
            print(f"❌ 验证失败：发现 {len(out_of_range)} 个超出0-1范围的相似度值")
            # 输出前5个异常值作为示例
            for i, (wf, th, sim) in enumerate(out_of_range[:5]):
                print(f"  异常值 {i + 1}: 权重因子={wf}, 阈值={th}, 相似度={sim:.6f}")
    else:
        print("\n⚠️ 警告：未计算任何相似度值，无法进行范围验证")

    # 保存验证结果
    analysis_path = os.path.join(result_dir, 'weight_validation_with_cm_sampled.xlsx')
    with pd.ExcelWriter(analysis_path, engine='openpyxl') as writer:
        # 保存性能指标
        analysis_results.to_excel(writer, sheet_name='性能指标', index=False)

        # 保存混淆矩阵
        for idx, cm_data in enumerate(confusion_matrices):
            wf = cm_data['weight_factor']
            th = cm_data['threshold']
            cm = cm_data['confusion_matrix']

            cm_df = pd.DataFrame(
                cm,
                index=['真实-不相似', '真实-相似'],
                columns=['预测-不相似', '预测-相似']
            )
            cm_df['权重因子'] = wf
            cm_df['相似度阈值'] = th

            sheet_name = f'混淆矩阵_{idx + 1}_wf{wf}_th{th}'.replace('.', '_')
            if len(sheet_name) > 31:
                sheet_name = f'CM_{idx + 1}_wf{wf}_th{th}'.replace('.', '_')

            cm_df.to_excel(writer, sheet_name=sheet_name)

    print(f"\n验证结果已保存到: {analysis_path}")

    # 显示最优参数（如果有）
    if not analysis_results.empty:
        best_idx = analysis_results['f1_score'].idxmax()
        best_params = analysis_results.loc[best_idx]
        print("\n最优参数组合:")
        print(f"权重因子: {best_params['weight_factor']}")
        print(f"相似度阈值: {best_params['threshold']}")
        print(f"F1分数: {best_params['f1_score']:.4f}")
        print(f"精确率: {best_params['precision']:.4f}, 召回率: {best_params['recall']:.4f}")
    else:
        print("\n未找到有效评估结果")

    total_time = time.time() - start_time
    print(f"\n验证程序运行完成，总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    main()
