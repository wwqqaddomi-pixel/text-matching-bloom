import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Any, Tuple
import torch
import time
import sys

try:
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
except ImportError:
    print("错误：请确保安装了transformers库：pip install transformers")
    print("同时需要安装protobuf和sentencepiece：pip install protobuf sentencepiece")
    exit(1)


# 清除代理设置
def clear_proxies():
    """清除所有可能的代理设置"""
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    # 对于Windows系统，额外检查系统代理设置
    if sys.platform.startswith('win'):
        try:
            import winreg
            reg_path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE) as key:
                winreg.SetValueEx(key, 'ProxyEnable', 0, winreg.REG_DWORD, 0)
        except Exception as e:
            print(f"警告: 无法修改Windows注册表代理设置: {e}")


# 先清除代理再进行其他操作
clear_proxies()


class BF:
    def __init__(self, bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                 max_abs_diff, min_val, max_val, q, weight_file, weight_coefficient=1.0):
        """初始化布隆过滤器的参数，新增weight_coefficient参数调节权重影响"""
        self.bf_len = bf_len
        self.bf_num_hash_func = bf_num_hash_func
        self.bf_num_inter = bf_num_inter
        self.bf_step = bf_step
        self.max_abs_diff = max_abs_diff
        self.min_val = min_val
        self.max_val = max_val
        self.q = q
        self.weight_coefficient = weight_coefficient  # 权重系数，用于调节权重影响程度
        assert max_val > min_val
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
        self.weights = self.load_weights(weight_file)
        self.default_weight = 0.5  # 未找到替换权重时的默认值

    def load_weights(self, weight_file):
        """加载权重文件"""
        try:
            df = pd.read_csv(weight_file)
        except FileNotFoundError:
            print(f"错误: 权重文件 {weight_file} 未找到，请检查路径是否正确")
            exit(1)

        weights = {}
        for index, row in df.iterrows():
            key = f"{row['original']}-{row['replacement']}"
            weights[key] = row['weight']
        return weights

    def _hash_value(self, val):
        """计算哈希值"""
        hex_str1 = self.h1(val.encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(val.encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    def set_to_bloom_filter(self, val_set, original_set):
        """将输入集合转换为布隆过滤器"""
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
        """创建融合原始BF和替换后BF的AVa集合，应用权重系数"""
        AVa = [0.0] * self.bf_len

        for i in range(self.bf_len):
            if original_bf[i] == 1:
                AVa[i] = 1.0
            elif original_bf[i] == 0 and replaced_bf[i] == 1:
                weight = self.get_replacement_weight(original_text, replaced_text, i)
                # 应用权重系数，确保值在[0,1]范围内
                weighted_value = weight * self.weight_coefficient
                AVa[i] = max(0.0, min(1.0, weighted_value))
            else:
                AVa[i] = 0.0

        return AVa

    def get_replacement_weight(self, original_text, replaced_text, position):
        """获取特定位置的替换权重"""
        char_pairs = self.find_char_pairs(original_text, replaced_text, position)

        if not char_pairs:
            return self.default_weight

        weight_sum = 0
        for orig_char, repl_char in char_pairs:
            key = f"{orig_char}-{repl_char}"
            weight_sum += self.weights.get(key, self.default_weight)

        return weight_sum / len(char_pairs)

    def find_char_pairs(self, original_text, replaced_text, position):
        """根据布隆过滤器位置反推可能的字符替换对"""
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
        """计算基于AVa和BVb集合的新型Dice相似度"""
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
        """初始化ByT5模型用于生成q-gram变体"""
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
        """生成文本的q-gram及其变体"""
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

        return list(set(original_qgrams + variant_qgrams))


def load_dataset(file_path: str, source_name: str) -> pd.DataFrame:
    """加载数据集并添加源标记"""
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


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.2, random_state: int = 9318) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """将数据集按比例划分为训练集和测试集"""
    train_df = df.sample(frac=train_ratio, random_state=random_state)
    test_df = df.drop(train_df.index)

    print(f"数据集划分完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
    return train_df, test_df


def evaluate_with_coefficient(acm_test_records, scholar_test_records, weight_file, coefficient,
                              bf_len, bf_num_hash_func, bf_num_inter, bf_step, max_abs_diff,
                              min_val, max_val, q, similarity_threshold):
    """使用指定的权重系数进行评估并返回性能指标"""
    bf = BF(bf_len, bf_num_hash_func, bf_num_inter, bf_step,
            max_abs_diff, min_val, max_val, q, weight_file, weight_coefficient=coefficient)

    all_pairs = []

    for i, rec1 in enumerate(acm_test_records):
        for j, rec2 in enumerate(scholar_test_records):
            try:
                new_sim = bf.calc_bf_sim(rec1['AVa'], rec2['AVa'])

                true_label = 1 if (rec1['id'] == rec2['id'] or
                                   (rec1['label'] == 1 and rec2['label'] == 1)) else 0
                pred_label = 1 if new_sim >= similarity_threshold else 0

                all_pairs.append({
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'new_similarity': new_sim
                })
            except Exception as e:
                print(f"警告: 计算记录对 ({i},{j}) 相似度时出错 - {e}")
                continue

    if all_pairs:
        y_true = [pair['true_label'] for pair in all_pairs]
        y_pred = [pair['pred_label'] for pair in all_pairs]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'coefficient': coefficient,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    return None


def process_records_with_coefficient(data, source, q, byt5_generator, weight_file, coefficient):
    """使用指定的权重系数处理记录并生成AVa集合"""
    processed_records = []
    start_time = time.time()
    max_records = len(data)

    # 原始哈希函数
    def _hash_value(val: Any) -> Tuple[int, int]:
        h1 = hashlib.sha1(str(val).encode('utf-8')).hexdigest()
        h2 = hashlib.md5(str(val).encode('utf-8')).hexdigest()
        return int(h1, 16), int(h2, 16)

    # 原始布隆过滤器生成
    def features_to_bloom_filter(features: List[Any]) -> bitarray:
        bloom = bitarray(400)
        bloom.setall(False)
        for feature in features:
            int1, int2 = _hash_value(feature)
            for i in range(6):
                gi = (int1 + i * int2) % 400
                bloom[gi] = True
        return bloom

    # 创建临时BF实例用于处理当前系数
    temp_bf = BF(400, 6, 5, 1, 20, 0, 100, q, weight_file, weight_coefficient=coefficient)

    for idx in range(max_records):
        try:
            row = data.iloc[idx]
            id_value = row['id']

            required_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                                'right_title', 'right_authors', 'right_venue', 'right_year']
            features = {col: row[col] for col in required_columns + ['source_file']}

            # 原始布隆过滤器处理
            all_features = []
            for col in required_columns:
                value = row[col]
                if pd.notna(value):
                    all_features.append(str(value))
            original_bf = features_to_bloom_filter(all_features)

            # 生成q-gram变体
            val_set = []
            original_set = []
            for col in required_columns:
                value = row[col]
                if pd.notna(value) and isinstance(value, str) and len(value) >= q:
                    qgrams = byt5_generator.generate_variants(value, q=q)
                    val_set.extend(qgrams)
                    original_qgrams = [value[i:i + q] for i in range(len(value) - q + 1)]
                    original_set.extend(original_qgrams)

            val_set = list(set(val_set))
            original_set = list(set(original_set))

            replaced_bf, _, _ = temp_bf.set_to_bloom_filter(val_set, original_set)

            # 生成AVa集合（已应用当前权重系数）
            original_text = " ".join([str(row[col]) for col in required_columns if pd.notna(row[col])])
            replaced_text = " ".join(val_set)

            AVa = temp_bf.create_AVa(original_bf, replaced_bf, original_text, replaced_text)

            processed_records.append({
                'id': id_value,
                'row_index': idx,
                'original_bf': original_bf,
                'replaced_bf': replaced_bf,
                'AVa': AVa,
                'label': row.get('label', 0)
            })

        except Exception as e:
            print(f"警告: 处理记录 {idx} 时出错 - {e}")
            continue

        # 进度显示
        if (idx + 1) % max(1, max_records // 20) == 0:
            progress = (idx + 1) / max_records * 100
            print(f"系数 {coefficient} - 已处理 {source} 中的 {idx + 1}/{max_records} 条记录 ({progress:.1f}%)")

    return processed_records


def main():
    start_time = time.time()

    # 设置随机数种子
    random.seed(9318)
    torch.manual_seed(9318)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9318)

    # 布隆过滤器参数
    bf_len = 400
    bf_num_hash_func = 6
    similarity_threshold = 0.7

    # 新型相似度参数
    bf_num_inter = 5
    bf_step = 1
    max_abs_diff = 20
    min_val = 0
    max_val = 100
    q = 2

    # 数据集划分比例 (仅使用80%的评估部分)
    train_ratio = 0.2

    # 权重系数测试范围 - 可根据需要调整
    weight_coefficients = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0, 2.5, 3.0]
    # 如需更精细的结果，可使用更小的步长，如0.1为间隔

    # 文件路径设置 - 直接使用已有的权重文件
    data_dir = r"D:\实习\COMP1\icip2"
    acm_test_path = os.path.join(data_dir, "dirty_dblp_acm", "test.csv")
    scholar_test_path = os.path.join(data_dir, "dirty_dblp_scholar", "test.csv")

    # 确保结果目录存在
    result_dir = os.path.join(data_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # 权重文件路径 - 使用指定的现有权重文件
    weight_file = os.path.join(result_dir, 'byt5_adapted_weights_from_test_unlimited.csv')
    if not os.path.exists(weight_file):
        print(f"错误: 权重文件 {weight_file} 不存在，请检查路径是否正确")
        return
    print(f"将使用现有权重文件: {weight_file}")

    # 初始化ByT5 q-gram生成器
    byt5_generator = ByT5QGramGenerator(model_size=r"D:\实习\COMP1\ByT5")

    try:
        # 加载测试数据集
        print("\n加载原始测试数据集...")
        acm_test_full = load_dataset(acm_test_path, "acm_test_full")
        scholar_test_full = load_dataset(scholar_test_path, "scholar_test_full")

        # 将测试集按2:8比例划分（仅使用80%的评估部分）
        print(f"\n将测试集按 {train_ratio * 100}:{(1 - train_ratio) * 100} 比例划分...")
        _, acm_test_eval = split_dataset(acm_test_full, train_ratio)
        _, scholar_test_eval = split_dataset(scholar_test_full, train_ratio)

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 定义需要用于匹配的字段
    required_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                        'right_title', 'right_authors', 'right_venue', 'right_year']

    # 检查必要的列是否存在
    for df in [acm_test_eval, scholar_test_eval]:
        for col in required_columns:
            if col not in df.columns:
                print(f"错误：数据缺少必要的列 '{col}'")
                return

    # 存储不同系数下的评估结果
    evaluation_results = []

    # 遍历所有权重系数，评估性能
    for coeff in weight_coefficients:
        print(f"\n{'=' * 50}")
        print(f"开始评估权重系数: {coeff}")
        coeff_start_time = time.time()

        # 使用当前系数处理记录（生成带权重系数的AVa）
        print(f"\n处理ACM测试集（系数 {coeff}）...")
        acm_records = process_records_with_coefficient(
            acm_test_eval, 'acm_test_eval', q, byt5_generator, weight_file, coeff)

        print(f"\n处理Scholar测试集（系数 {coeff}）...")
        scholar_records = process_records_with_coefficient(
            scholar_test_eval, 'scholar_test_eval', q, byt5_generator, weight_file, coeff)

        # 评估当前系数的性能
        print(f"\n评估系数 {coeff} 的性能...")
        result = evaluate_with_coefficient(
            acm_records, scholar_records, weight_file, coeff,
            bf_len, bf_num_hash_func, bf_num_inter, bf_step,
            max_abs_diff, min_val, max_val, q, similarity_threshold
        )

        if result:
            evaluation_results.append(result)
            print(f"系数 {coeff} 评估完成:")
            print(f"  准确率: {result['accuracy']:.4f}")
            print(f"  精确率: {result['precision']:.4f}")
            print(f"  召回率: {result['recall']:.4f}")
            print(f"  F1分数: {result['f1']:.4f}")
        else:
            print(f"系数 {coeff} 评估失败")

        print(f"系数 {coeff} 处理耗时: {time.time() - coeff_start_time:.2f}s")

    # 保存所有系数的评估结果
    if evaluation_results:
        results_df = pd.DataFrame(evaluation_results)
        results_path = os.path.join(result_dir, 'weight_coefficient_evaluation.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n所有系数的评估结果已保存到: {results_path}")

        # 找到最优系数（基于F1分数，F1综合了精确率和召回率）
        best_result = max(evaluation_results, key=lambda x: x['f1'])
        print(f"\n{'=' * 50}")
        print("最优权重系数分析:")
        print(f"  最优系数值: {best_result['coefficient']}")
        print(f"  对应的F1分数: {best_result['f1']:.4f}")
        print(f"  准确率: {best_result['accuracy']:.4f}")
        print(f"  精确率: {best_result['precision']:.4f}")
        print(f"  召回率: {best_result['recall']:.4f}")

        # 输出结果趋势
        print("\n权重系数与F1分数关系:")
        for res in sorted(evaluation_results, key=lambda x: x['coefficient']):
            print(f"  系数 {res['coefficient']:.1f}: F1 = {res['f1']:.4f}")
    else:
        print("\n没有得到有效的评估结果")

    # 输出总运行时间
    total_time = time.time() - start_time
    print(f"\n程序运行完成，总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    main()
