import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple
import torch
import time
import sys
#28分test.csv

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
            # 打开Internet设置注册表项
            reg_path = r'Software\Microsoft\Windows\CurrentVersion\Internet Settings'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE) as key:
                # 禁用代理
                winreg.SetValueEx(key, 'ProxyEnable', 0, winreg.REG_DWORD, 0)
        except Exception as e:
            print(f"警告: 无法修改Windows注册表代理设置: {e}")


# 先清除代理再进行其他操作
clear_proxies()


class BF:
    def __init__(self, bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                 max_abs_diff, min_val, max_val, q, weight_file):
        """初始化布隆过滤器的参数"""
        self.bf_len = bf_len
        self.bf_num_hash_func = bf_num_hash_func
        self.bf_num_inter = bf_num_inter
        self.bf_step = bf_step
        self.max_abs_diff = max_abs_diff
        self.min_val = min_val
        self.max_val = max_val
        self.q = q
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
            print(f"警告: 权重文件 {weight_file} 未找到，使用默认权重")
            return {}

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
        """创建融合原始BF和替换后BF的AVa集合"""
        # 注意：bitarray只能存储0/1，这里使用列表存储浮点数权重
        AVa = [0.0] * self.bf_len

        # 遍历布隆过滤器的每个位置
        for i in range(self.bf_len):
            if original_bf[i] == 1:
                # 情况1: 原始BF中为1，AVa中也为1
                AVa[i] = 1.0
            elif original_bf[i] == 0 and replaced_bf[i] == 1:
                # 情况2: 原始BF中为0，替换后BF中为1，查找替换权重
                weight = self.get_replacement_weight(original_text, replaced_text, i)
                AVa[i] = weight  # 存储权重值
            else:
                # 情况3: 两者都为0，AVa中也为0
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

        return weight_sum / len(char_pairs)  # 返回平均权重

    def find_char_pairs(self, original_text, replaced_text, position):
        """根据布隆过滤器位置反推可能的字符替换对"""
        # 提取原始文本和替换文本的q-gram
        qgrams_original = [original_text[i:i + self.q] for i in range(len(original_text) - self.q + 1)]
        qgrams_replaced = [replaced_text[i:i + self.q] for i in range(len(replaced_text) - self.q + 1)]

        # 找出导致该位置置1的所有q-gram
        responsible_qgrams_original = []
        responsible_qgrams_replaced = []

        for qg_orig in qgrams_original:
            int1, int2 = self._hash_value(qg_orig)
            for i in range(self.bf_num_hash_func):
                gi = (int1 + i * int2) % self.bf_len
                if gi == position:
                    responsible_qgrams_original.append(qg_orig)
                    break  # 找到一个匹配即可

        for qg_repl in qgrams_replaced:
            int1, int2 = self._hash_value(qg_repl)
            for i in range(self.bf_num_hash_func):
                gi = (int1 + i * int2) % self.bf_len
                if gi == position:
                    responsible_qgrams_replaced.append(qg_repl)
                    break  # 找到一个匹配即可

        # 找出所有可能的字符替换对
        char_pairs = []
        for qg_orig in responsible_qgrams_original:
            for qg_repl in responsible_qgrams_replaced:
                # 找出q-gram中不同的字符对
                for i in range(min(len(qg_orig), len(qg_repl))):
                    if qg_orig[i] != qg_repl[i]:
                        char_pairs.append((qg_orig[i], qg_repl[i]))

        return char_pairs

    def calc_bf_sim(self, AVa, BVb):
        """计算基于AVa和BVb集合的新型Dice相似度"""
        # 计算交集的权重和
        intersection_sum = 0
        # 计算并集的权重和
        union_sum = 0

        for i in range(self.bf_len):
            a_val = AVa[i]
            b_val = BVb[i]

            # 交集：两个集合中对应位置的最小值
            intersection_sum += min(a_val, b_val)

            # 并集：两个集合中对应位置的最大值
            union_sum += max(a_val, b_val)

        # 避免除零错误
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

        # 检查GPU可用性并设置相应的推理参数
        if torch.cuda.is_available():
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
            # GPU模式下可以使用稍高的参数
            self.inference_params = {
                "num_beams": 3,
                "num_return_sequences": 3,
                "max_length": 40,
                "temperature": 0.7
            }
        else:
            print("使用 CPU 进行推理，可能较慢")
            # CPU模式下使用更高效的参数
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

        # 解码生成的变体
        variants = []
        for output in outputs:
            try:
                variant = self.tokenizer.decode(output, skip_special_tokens=True)
                if variant and variant != input_text:  # 确保生成了有效内容
                    variants.append(variant)
            except Exception as e:
                print(f"警告: 解码失败 - {e}")

        # 从变体中提取q-grams
        variant_qgrams = []
        for variant in variants:
            variant_qgrams.extend([variant[i:i + q] for i in range(len(variant) - q + 1)])

        # 合并原始和变体q-grams并去重
        all_qgrams = list(set(original_qgrams + variant_qgrams))
        return all_qgrams

    def generate_weight_file(self, training_texts: List[str], output_file: str, q: int = 2):
        """利用ByT5生成字符替换权重文件，使用所有训练文本"""
        print(f"开始生成权重文件: {output_file}")
        replacement_counts = defaultdict(int)  # 统计替换次数
        total_counts = defaultdict(int)  # 统计原始字符出现总次数

        # 使用所有训练文本，不做数量限制
        print(f"使用所有 {len(training_texts)} 个样本生成权重")

        # 对每个训练文本，生成变体并统计替换
        for i, text in enumerate(training_texts):
            # 增加进度显示频率，因为数据量可能很大
            if i % 10 == 0 and i > 0:
                print(f"已处理 {i}/{len(training_texts)} 个文本")

            if not isinstance(text, str) or len(text) < q:
                continue

            # 生成文本变体
            variants = self.generate_variants(text, q=q)
            if not variants:
                continue

            # 提取原始q-grams
            original_qgrams = [text[i:i + q] for i in range(len(text) - q + 1)]

            # 对比原始文本与每个变体
            for variant in variants:
                variant_qgrams = [variant[i:i + q] for i in range(len(variant) - q + 1)]

                # 对比所有可能的q-gram对
                min_length = min(len(original_qgrams), len(variant_qgrams))
                for orig_qg, var_qg in zip(original_qgrams[:min_length], variant_qgrams[:min_length]):
                    if orig_qg == var_qg:
                        continue

                    # 统计每个位置的字符替换
                    for c1, c2 in zip(orig_qg, var_qg):
                        if c1 != c2:
                            replacement_counts[(c1, c2)] += 1
                            total_counts[c1] += 1

        # 计算替换概率作为权重
        weights = {}
        for (orig, repl), count in replacement_counts.items():
            if total_counts[orig] > 0:
                # 权重 = 替换频率，范围 [0,1]
                weights[f"{orig}-{repl}"] = count / total_counts[orig]

        # 保存到CSV文件
        df = pd.DataFrame({
            'original': [k.split('-')[0] for k in weights.keys()],
            'replacement': [k.split('-')[1] for k in weights.keys()],
            'weight': list(weights.values())
        })
        df.to_csv(output_file, index=False)
        print(f"权重文件已保存至: {output_file}，包含 {len(weights)} 个替换规则")
        return weights


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
    """
    将数据集按比例划分为训练集和测试集
    :param df: 原始数据集
    :param train_ratio: 训练集占比
    :param random_state: 随机种子，确保结果可复现
    :return: 训练集和测试集
    """
    # 随机打乱并按比例划分
    train_df = df.sample(frac=train_ratio, random_state=random_state)
    test_df = df.drop(train_df.index)

    print(f"数据集划分完成: 训练集 {len(train_df)} 条, 测试集 {len(test_df)} 条")
    return train_df, test_df


def main():
    start_time = time.time()  # 记录程序开始时间

    # 设置随机数种子，确保结果可复现
    random.seed(9318)
    torch.manual_seed(9318)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(9318)

    # 布隆过滤器参数设置
    bf_len = 400
    bf_num_hash_func = 6
    similarity_threshold = 0.7  # 降低阈值以提高召回率

    # 新型相似度参数
    bf_num_inter = 5
    bf_step = 1
    max_abs_diff = 20
    min_val = 0
    max_val = 100
    q = 2  # q-gram长度

    # 数据集划分比例 (20%用于生成权重，80%用于测试评估)
    train_ratio = 0.2

    # 文件路径设置
    data_dir = r"D:\实习\COMP1\icip2"
    acm_test_path = os.path.join(data_dir, "dirty_dblp_acm", "test.csv")
    scholar_test_path = os.path.join(data_dir, "dirty_dblp_scholar", "test.csv")

    # 确保结果目录存在
    result_dir = os.path.join(data_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    # 权重文件路径
    weight_file = os.path.join(result_dir, 'byt5_adapted_weights_from_test_unlimited.csv')

    # 初始化ByT5 q-gram生成器
    byt5_generator = ByT5QGramGenerator(model_size=r"D:\实习\COMP1\ByT5")

    try:
        # 加载测试数据集
        print("\n加载原始测试数据集...")
        acm_test_full = load_dataset(acm_test_path, "acm_test_full")
        scholar_test_full = load_dataset(scholar_test_path, "scholar_test_full")

        # 将测试集按2:8比例划分
        print(f"\n将测试集按 {train_ratio * 100}:{(1 - train_ratio) * 100} 比例划分...")
        acm_test_train, acm_test_eval = split_dataset(acm_test_full, train_ratio)
        scholar_test_train, scholar_test_eval = split_dataset(scholar_test_full, train_ratio)

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 定义需要用于匹配的字段
    required_columns = ['left_title', 'left_authors', 'left_venue', 'left_year',
                        'right_title', 'right_authors', 'right_venue', 'right_year']

    # 检查必要的列是否存在
    for df in [acm_test_train, acm_test_eval, scholar_test_train, scholar_test_eval]:
        for col in required_columns:
            if col not in df.columns:
                print(f"错误：数据缺少必要的列 '{col}'")
                return

    # 从测试集的20%部分提取所有文本用于生成权重（无数量限制）
    print("\n准备训练文本数据（使用测试集的20%，无数量限制）...")
    training_texts = []
    # 使用划分后的训练部分生成权重
    for df in [acm_test_train, scholar_test_train]:
        for col in required_columns:
            # 提取非空文本并转换为字符串
            texts = df[col].dropna().astype(str).tolist()
            training_texts.extend(texts)

    print(f"使用所有 {len(training_texts)} 个测试文本样本（20%部分）生成替换权重")

    # 生成并保存权重文件（如果不存在）
    if not os.path.exists(weight_file):
        print("\n生成字符替换权重文件（基于测试集的20%，无数量限制）...")
        weight_gen_start = time.time()
        byt5_generator.generate_weight_file(
            training_texts=training_texts,
            output_file=weight_file,
            q=q
        )
        print(f"权重文件生成耗时: {time.time() - weight_gen_start:.2f}s")
    else:
        print(f"\n权重文件已存在，直接加载: {weight_file}")

    # 原始哈希函数
    def _hash_value(val: Any) -> Tuple[int, int]:
        h1 = hashlib.sha1(str(val).encode('utf-8')).hexdigest()
        h2 = hashlib.md5(str(val).encode('utf-8')).hexdigest()
        return int(h1, 16), int(h2, 16)

    # 原始布隆过滤器生成
    def features_to_bloom_filter(features: List[Any]) -> bitarray:
        bloom = bitarray(bf_len)
        bloom.setall(False)
        for feature in features:
            int1, int2 = _hash_value(feature)
            for i in range(bf_num_hash_func):
                gi = (int1 + i * int2) % bf_len
                bloom[gi] = True
        return bloom

    # 原始相似度计算（Dice系数）
    def calc_old_bf_similarity(bf1: bitarray, bf2: bitarray) -> float:
        count1 = bf1.count()
        count2 = bf2.count()
        common = (bf1 & bf2).count()
        if count1 + count2 == 0:
            return 0.0
        return (2.0 * common) / (count1 + count2)

    # 初始化新型相似度计算器
    new_bf = BF(bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                max_abs_diff, min_val, max_val, q, weight_file)

    print("\n正在生成布隆过滤器和AVa/BVb集合...")

    # 为每个文件的记录计算布隆过滤器和AVa/BVb集合
    def process_file(data: pd.DataFrame, source: str, is_test: bool = False) -> List[Dict[str, Any]]:
        processed_records = []
        start_time = time.time()
        # 处理所有记录，不做数量限制
        max_records = len(data)

        for idx in range(max_records):
            try:
                row = data.iloc[idx]
                id_value = row['id']  # 使用数据集中的id列

                # 提取特征
                features = {col: row[col] for col in required_columns + ['source_file']}

                # 原始布隆过滤器处理方式
                all_features = []
                for col in required_columns:
                    value = row[col]
                    if pd.notna(value):
                        all_features.append(str(value))
                original_bf = features_to_bloom_filter(all_features)

                # 新型布隆过滤器处理方式（使用ByT5生成q-gram变体）
                val_set = []
                original_set = []
                for col in required_columns:
                    value = row[col]
                    if pd.notna(value) and isinstance(value, str) and len(value) >= q:
                        qgrams = byt5_generator.generate_variants(value, q=q)
                        val_set.extend(qgrams)
                        # 原始q-grams作为基准
                        original_qgrams = [value[i:i + q] for i in range(len(value) - q + 1)]
                        original_set.extend(original_qgrams)

                val_set = list(set(val_set))  # 去重
                original_set = list(set(original_set))

                replaced_bf, extra_positions, extra_positions_with_chars = new_bf.set_to_bloom_filter(
                    val_set, original_set)

                # 生成AVa集合
                # 合并所有特征字段为一个完整字符串
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
                    'AVa': AVa,  # 存储AVa集合
                    'extra_positions': extra_positions,
                    'extra_positions_with_chars': extra_positions_with_chars,
                    'features': features,
                    'val_set': val_set,
                    'original_set': original_set,
                    'label': row.get('label', 0)  # 保存标签用于评估
                })

            except Exception as e:
                print(f"警告: 处理记录 {idx} 时出错 - {e}")
                continue  # 继续处理下一条记录

            # 进度显示
            if (idx + 1) % max(1, max_records // 20) == 0:  # 每5%报告一次
                progress = (idx + 1) / max_records * 100
                elapsed = time.time() - start_time
                eta = elapsed * (max_records / (idx + 1) - 1)
                print(f"已处理 {source} 中的 {idx + 1}/{max_records} 条记录 ({progress:.1f}%), "
                      f"耗时: {elapsed:.2f}s, ETA: {eta:.2f}s")

        total_time = time.time() - start_time
        print(f"{source} 处理完成，共耗时: {total_time:.2f}s")
        return processed_records

    # 处理测试集的80%部分（用于评估）
    print("\n处理ACM测试集的80%评估部分...")
    acm_test_records = process_file(acm_test_eval, 'acm_test_eval', is_test=True)

    print("\n处理Scholar测试集的80%评估部分...")
    scholar_test_records = process_file(scholar_test_eval, 'scholar_test_eval', is_test=True)

    print("\n正在计算相似度...")
    similar_pairs = []
    all_pairs = []  # 存储所有记录对的真实标签和预测结果

    # 初始化相似度矩阵
    num_records1 = len(acm_test_records)
    num_records2 = len(scholar_test_records)
    new_similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]
    old_similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]

    # 计算相似度（带进度显示）
    total_pairs = num_records1 * num_records2
    processed_pairs = 0
    report_interval = max(total_pairs // 20, 1)  # 每5%报告一次进度
    similarity_start_time = time.time()

    for i, rec1 in enumerate(acm_test_records):
        for j, rec2 in enumerate(scholar_test_records):
            try:
                # 计算原始相似度
                old_sim = calc_old_bf_similarity(
                    rec1['original_bf'],
                    rec2['original_bf']
                )

                # 计算新型相似度（基于AVa和BVb）
                new_sim = new_bf.calc_bf_sim(
                    rec1['AVa'],
                    rec2['AVa']
                )

                # 填充相似度矩阵
                new_similarity_matrix[i][j] = new_sim
                old_similarity_matrix[i][j] = old_sim

                # 真实标签：使用数据集中的label或通过id匹配判断
                true_label = 1 if (rec1['id'] == rec2['id'] or
                                   (rec1['label'] == 1 and rec2['label'] == 1)) else 0
                # 预测标签：新型相似度是否≥阈值
                pred_label = 1 if new_sim >= similarity_threshold else 0

                all_pairs.append({
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'new_similarity': new_sim,
                    'old_similarity': old_sim
                })

                # 保存相似对
                if new_sim >= similarity_threshold:
                    pair_record = {
                        'id_1': rec1['id'],
                        'id_2': rec2['id'],
                        'old_similarity': old_sim,
                        'new_similarity': new_sim,
                        'same_id': 1 if rec1['id'] == rec2['id'] else 0,
                        'true_label': true_label
                    }

                    # 添加两个记录的详细特征
                    for attr in required_columns + ['source_file']:
                        pair_record[f'rec1_{attr}'] = rec1['features'][attr]
                        pair_record[f'rec2_{attr}'] = rec2['features'][attr]

                    similar_pairs.append(pair_record)

            except Exception as e:
                print(f"警告: 计算记录对 ({i},{j}) 相似度时出错 - {e}")

            # 更新进度
            processed_pairs += 1
            if processed_pairs % report_interval == 0:
                progress = processed_pairs / total_pairs * 100
                elapsed = time.time() - similarity_start_time
                eta = elapsed * (total_pairs / processed_pairs - 1)
                print(f"相似度计算进度: {progress:.1f}%, "
                      f"已处理 {processed_pairs}/{total_pairs} 对, "
                      f"耗时: {elapsed:.2f}s, ETA: {eta:.2f}s")

    similarity_time = time.time() - similarity_start_time
    print(f"相似度计算完成，共耗时: {similarity_time:.2f}s")
    print(f"找到 {len(similar_pairs)} 对相似记录")

    # 保存相似度矩阵
    print("\n正在保存相似度矩阵...")

    # 保存新型相似度矩阵
    new_matrix_df = pd.DataFrame(new_similarity_matrix)
    new_matrix_path = os.path.join(result_dir, 'new_similarity_matrix.csv')
    new_matrix_df.to_csv(new_matrix_path, index=False, header=False)
    print(f"新型相似度矩阵已保存到：{new_matrix_path}")

    # 保存原始相似度矩阵
    old_matrix_df = pd.DataFrame(old_similarity_matrix)
    old_matrix_path = os.path.join(result_dir, 'old_similarity_matrix.csv')
    old_matrix_df.to_csv(old_matrix_path, index=False, header=False)
    print(f"原始相似度矩阵已保存到：{old_matrix_path}")

    # 保存结果
    print("\n正在保存结果...")
    try:
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            required_columns_output = [
                'id_1', 'id_2', 'old_similarity', 'new_similarity', 'same_id', 'true_label'
            ]
            for attr in required_columns + ['source_file']:
                required_columns_output.extend([f'rec1_{attr}', f'rec2_{attr}'])

            # 确保DataFrame包含所有需要的列
            for col in required_columns_output:
                if col not in pairs_df.columns:
                    pairs_df[col] = None

            # 按指定顺序排列列
            pairs_df = pairs_df[required_columns_output]

            pairs_path = os.path.join(result_dir, 'acm_scholar_similar_records.xlsx')
            pairs_df.to_excel(pairs_path, index=False)
            print(f"相似记录对已保存到 {pairs_path}")
            print(f"其中 {sum(p['same_id'] for p in similar_pairs)} 对记录具有相同的ID")
        else:
            print("未找到相似记录对")

    except Exception as e:
        print(f"错误：保存结果时发生异常 - {str(e)}")

    # 生成评估指标（使用测试集的80%部分）
    print("\n正在生成评估指标...")
    if all_pairs:
        y_true = [pair['true_label'] for pair in all_pairs]
        y_pred = [pair['pred_label'] for pair in all_pairs]

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0:不相似，1:相似
        cm_df = pd.DataFrame(cm, index=['真实-不相似', '真实-相似'], columns=['预测-不相似', '预测-相似'])

        # 生成分类报告（精确率、召回率等）
        class_report = classification_report(y_true, y_pred, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()

        # 保存混淆矩阵和分类报告到Excel
        metrics_path = os.path.join(result_dir, 'similarity_metrics.xlsx')
        with pd.ExcelWriter(metrics_path) as writer:
            cm_df.to_excel(writer, sheet_name='混淆矩阵')
            class_report_df.to_excel(writer, sheet_name='分类报告')

        print(f"评估指标已保存到 {metrics_path}")
        print("混淆矩阵：")
        print(cm_df)
        print("\n分类报告：")
        print(class_report_df)
    else:
        print("没有计算出任何相似度对，无法生成评估指标")

    # 输出总运行时间
    total_time = time.time() - start_time
    print(f"\n程序运行完成，总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    main()
