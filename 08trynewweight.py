import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple
import torch
import time  # 用于计时

try:
    from transformers import T5ForConditionalGeneration, ByT5Tokenizer
except ImportError:
    print("错误：请确保安装了transformers库：pip install transformers")
    print("同时需要安装protobuf和sentencepiece：pip install protobuf sentencepiece")
    exit(1)

import sys
import os


# 彻底清除代理设置
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

    def get_extra_common_1s(self, extra_positions1, extra_positions2, extra_positions_with_chars1,
                            extra_positions_with_chars2):
        """计算转换后集合多出来的交集"""
        extra_common_1s = 0
        for pos in set(extra_positions1).intersection(set(extra_positions2)):
            if pos in extra_positions_with_chars1 and pos in extra_positions_with_chars2:
                for char1 in extra_positions_with_chars1[pos]:
                    for char2 in extra_positions_with_chars2[pos]:
                        if char1 and char2:
                            # 假设 char1 是原始字符，char2 是替换字符
                            key = f"{char1[0]}-{char2[0]}"
                            if key in self.weights:
                                extra_common_1s += self.weights[key]
        return extra_common_1s

    def create_AVa(self, original_bf, replaced_bf, original_text, replaced_text):
        """
        创建融合原始BF和替换后BF的AVa集合
        original_bf: 原始文本的布隆过滤器
        replaced_bf: 替换后文本的布隆过滤器
        original_text: 原始文本
        replaced_text: 替换后文本
        """
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
        """
        获取特定位置的替换权重
        这里需要根据position反推可能的替换字符对
        """
        char_pairs = self.find_char_pairs(original_text, replaced_text, position)

        if not char_pairs:
            return self.default_weight

        weight_sum = 0
        for orig_char, repl_char in char_pairs:
            key = f"{orig_char}-{repl_char}"
            weight_sum += self.weights.get(key, self.default_weight)

        return weight_sum / len(char_pairs)  # 返回平均权重

    def find_char_pairs(self, original_text, replaced_text, position):
        """
        根据布隆过滤器位置反推可能的字符替换对
        这是一个简化实现，实际可能需要更复杂的映射逻辑
        """
        # 提取原始文本和替换文本的q-gram
        qgrams_original = [original_text[i:i + self.q] for i in range(len(original_text) - self.q + 1)]
        qgrams_replaced = [replaced_text[i:i + self.q] for i in range(len(replaced_text) - self.q + 1)]

        # 找出导致该位置置1的所有q-gram
        # 注意：这里的实现是简化的，实际需要根据哈希函数精确计算
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
        """
        计算基于AVa和BVb集合的新型Dice相似度
        公式: (2 * (AVa ∩ BVb)) / (AVa ∪ BVb)
        """
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

        # 检查GPU可用性
        if torch.cuda.is_available():
            print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用 CPU 进行推理，可能较慢")

    def generate_variants(self, text: str, q: int = 2, max_length: int = 50) -> List[str]:
        """生成文本的q-gram及其变体"""
        original_qgrams = [text[i:i + q] for i in range(len(text) - q + 1)]

        input_text = f"generate similar qgrams: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # 修复生成参数冲突，明确使用采样模式
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=3,
            do_sample=True,  # 启用采样
            temperature=0.7,  # 温度参数，控制随机性
            top_p=0.9,  # 核采样，限制概率分布
            early_stopping=True  # 生成完成后立即停止
        )

        # 解码生成的变体
        variants = []
        for output in outputs:
            try:
                variant = self.tokenizer.decode(output, skip_special_tokens=True)
                variants.append(variant)
            except Exception as e:
                print(f"警告: 解码失败 - {e}")
                variants.append("")  # 添加空字符串作为占位符

        # 从变体中提取q-grams
        variant_qgrams = []
        for variant in variants:
            variant_qgrams.extend([variant[i:i + q] for i in range(len(variant) - q + 1)])

        # 合并原始和变体q-grams并去重
        all_qgrams = list(set(original_qgrams + variant_qgrams))
        return all_qgrams

    def generate_weight_file(self, training_texts: List[str], output_file: str, q: int = 2):
        """
        利用ByT5生成字符替换权重文件

        Args:
            training_texts: 训练文本列表，用于生成替换模式
            output_file: 输出权重文件路径
            q: q-gram长度
        """
        print(f"开始生成权重文件: {output_file}")
        replacement_counts = defaultdict(int)  # 统计替换次数
        total_counts = defaultdict(int)  # 统计原始字符出现总次数

        # 对每个训练文本，生成变体并统计替换
        for i, text in enumerate(training_texts):
            if i % 100 == 0 and i > 0:
                print(f"已处理 {i}/{len(training_texts)} 个文本")

            # 生成文本变体
            variants = self.generate_variants(text, q=q)

            # 提取原始q-grams
            original_qgrams = [text[i:i + q] for i in range(len(text) - q + 1)]

            # 对比原始文本与每个变体
            for variant in variants:
                variant_qgrams = [variant[i:i + q] for i in range(len(variant) - q + 1)]

                # 找出原始q-gram与变体q-gram的字符差异
                for orig_qg, var_qg in zip(original_qgrams, variant_qgrams):
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
        print(f"权重文件已保存至: {output_file}")
        return weights


def main():
    start_time = time.time()  # 记录程序开始时间

    # 设置随机数种子，确保结果可复现
    random.seed(9318)

    # 布隆过滤器参数设置
    bf_len = 400
    bf_num_hash_func = 6
    similarity_threshold = 0.9

    # 新型相似度参数
    bf_num_inter = 5
    bf_step = 1
    max_abs_diff = 20
    min_val = 0
    max_val = 100
    q = 2

    # 生成适配ByT5的权重文件
    weight_file = 'byt5_adapted_weights.csv'

    # 初始化ByT5 q-gram生成器
    byt5_generator = ByT5QGramGenerator(model_size=r"D:\实习\COMP1\ByT5")

    # 读取数据文件
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data1_path = os.path.join(script_dir, 'data', r'process_1.csv')
        data2_path = os.path.join(script_dir, 'data', r'process_2.csv')

        for file_path in [data1_path, data2_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

        data1 = pd.read_csv(data1_path)
        data1['source_file'] = 'file1'

        data2 = pd.read_csv(data2_path)
        data2['source_file'] = 'file2'

        print(f"成功加载文件1: {len(data1)} 条记录")
        print(f"成功加载文件2: {len(data2)} 条记录")

    except FileNotFoundError as e:
        print(f"错误：找不到输入文件 - {str(e)}")
        print("请确保数据文件位于脚本同级的data目录下")
        return
    except Exception as e:
        print(f"错误：读取数据时发生异常 - {str(e)}")
        return

    # 确保数据包含必要的列
    required_columns = ['givenname', 'surname', 'suburb', 'postcode']
    for col in required_columns:
        if col not in data1.columns or col not in data2.columns:
            print(f"错误：数据缺少必要的列 '{col}'")
            return

    # 使用训练文本生成权重（从数据集中采样）
    training_texts = []
    for col in required_columns:
        training_texts.extend(data1[col].dropna().tolist())
        training_texts.extend(data2[col].dropna().tolist())

    # 限制样本量以控制时间（可根据需要调整）
    sample_size = min(1000, len(training_texts))
    training_sample = random.sample(training_texts, sample_size)

    # 生成并保存权重文件
    byt5_generator.generate_weight_file(
        training_texts=training_sample,
        output_file=weight_file,
        q=q
    )

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

    # 初始化新型相似度计算器（使用新生成的权重文件）
    new_bf = BF(bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                max_abs_diff, min_val, max_val, q, weight_file)

    print("正在生成布隆过滤器和AVa/BVb集合...")

    # 为每个文件的记录计算布隆过滤器和AVa/BVb集合
    def process_file(data: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        processed_records = []
        start_time = time.time()

        for idx, row in enumerate(data.itertuples()):
            try:
                id_value = row[1]  # 假设第一列是ID

                features = {
                    'givenname': getattr(row, 'givenname'),
                    'surname': getattr(row, 'surname'),
                    'suburb': getattr(row, 'suburb'),
                    'postcode': getattr(row, 'postcode'),
                    'source_file': source
                }

                # 原始布隆过滤器处理方式
                all_features = []
                for col, value in features.items():
                    if col != 'source_file':
                        all_features.append(str(value))  # 简化原始方法，不生成替换
                original_bf = features_to_bloom_filter(all_features)

                # 新型布隆过滤器处理方式（使用ByT5生成q-gram变体）
                val_set = []
                original_set = []
                for col in required_columns:
                    value = getattr(row, col)
                    if isinstance(value, str) and len(value) >= q:
                        # 使用ByT5生成q-gram变体
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
                original_text = " ".join([str(features[col]) for col in required_columns])
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
                    'original_set': original_set
                })

            except Exception as e:
                print(f"警告: 处理记录 {idx} 时出错 - {e}")
                continue  # 继续处理下一条记录

            # 进度显示，每5%报告一次
            if (idx + 1) % max(1, len(data) // 20) == 0:
                progress = (idx + 1) / len(data) * 100
                elapsed = time.time() - start_time
                eta = elapsed * (len(data) / (idx + 1) - 1)
                print(f"已处理 {source} 文件中的 {idx + 1}/{len(data)} 条记录 ({progress:.1f}%), "
                      f"耗时: {elapsed:.2f}s, ETA: {eta:.2f}s")

        total_time = time.time() - start_time
        print(f"{source} 文件处理完成，共耗时: {total_time:.2f}s")
        return processed_records

    # 处理两个文件
    print("\n处理第一个文件...")
    file1_bloom_filters = process_file(data1, 'file1')

    print("\n处理第二个文件...")
    file2_bloom_filters = process_file(data2, 'file2')

    print("\n正在计算相似度...")
    similar_pairs = []
    all_pairs = []  # 存储所有记录对的真实标签和预测结果

    # 初始化相似度矩阵
    num_records1 = len(file1_bloom_filters)
    num_records2 = len(file2_bloom_filters)
    new_similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]
    old_similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]

    # 计算相似度（带进度显示）
    total_pairs = num_records1 * num_records2
    processed_pairs = 0
    report_interval = max(total_pairs // 100, 1)  # 每1%报告一次进度
    similarity_start_time = time.time()

    for i, rec1 in enumerate(file1_bloom_filters):
        for j, rec2 in enumerate(file2_bloom_filters):
            try:
                # 计算原始相似度
                old_sim = calc_old_bf_similarity(
                    rec1['original_bf'],
                    rec2['original_bf']
                )

                # 计算新型相似度（基于AVa和BVb）
                new_sim = new_bf.calc_bf_sim(
                    rec1['AVa'],  # 使用AVa集合
                    rec2['AVa']  # 使用AVa集合（假设BVb同样存储在AVa字段中）
                )

                # 填充相似度矩阵
                new_similarity_matrix[i][j] = new_sim
                old_similarity_matrix[i][j] = old_sim

                # 真实标签：是否为同一ID（1为正例，0为负例）
                true_label = 1 if rec1['id'] == rec2['id'] else 0
                # 预测标签：新型相似度是否≥阈值（1为相似，0为不相似）
                pred_label = 1 if new_sim >= similarity_threshold else 0

                all_pairs.append({
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'new_similarity': new_sim,
                    'old_similarity': old_sim
                })

                # 使用新型相似度作为判断标准
                if new_sim >= similarity_threshold:
                    pair_record = {
                        'id_1': rec1['id'],
                        'id_2': rec2['id'],
                        'old_similarity': old_sim,
                        'new_similarity': new_sim,
                        'same_id': 1 if rec1['id'] == rec2['id'] else 0,
                    }

                    # 添加两个记录的详细特征
                    for attr in ['givenname', 'surname', 'suburb', 'postcode', 'source_file']:
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
                if processed_pairs % (report_interval * 10) == 0:  # 每10%详细报告一次
                    print(f"相似度计算进度: {progress:.1f}%, "
                          f"已处理 {processed_pairs}/{total_pairs} 对, "
                          f"耗时: {elapsed:.2f}s, ETA: {eta:.2f}s")

    similarity_time = time.time() - similarity_start_time
    print(f"相似度计算完成，共耗时: {similarity_time:.2f}s")
    print(f"找到 {len(similar_pairs)} 对相似记录")

    # 保存相似度矩阵
    print("\n正在保存相似度矩阵...")
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

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
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    try:
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            required_columns = [
                'id_1', 'id_2', 'old_similarity', 'new_similarity', 'same_id',
                'rec1_givenname', 'rec1_surname', 'rec1_suburb', 'rec1_postcode', 'rec1_source_file',
                'rec2_givenname', 'rec2_surname', 'rec2_suburb', 'rec2_postcode', 'rec2_source_file'
            ]

            # 确保DataFrame包含所有需要的列
            for col in required_columns:
                if col not in pairs_df.columns:
                    pairs_df[col] = None

            # 按指定顺序排列列
            pairs_df = pairs_df[required_columns]

            pairs_path = os.path.join(result_dir, 'cross_file_similar_records.xlsx')
            pairs_df.to_excel(pairs_path, index=False)
            print(f"相似记录对已保存到 {pairs_path}")
            print(f"其中 {sum(p['same_id'] for p in similar_pairs)} 对记录具有相同的ID")
        else:
            print("未找到相似记录对")

    except Exception as e:
        print(f"错误：保存结果时发生异常 - {str(e)}")

    print("\n正在生成混淆矩阵...")
    y_true = [pair['true_label'] for pair in all_pairs]
    y_pred = [pair['pred_label'] for pair in all_pairs]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0:不相似，1:相似
    cm_df = pd.DataFrame(cm, index=['真实-不相似', '真实-相似'], columns=['预测-不相似', '预测-相似'])

    # 生成分类报告（精确率、召回率等）
    class_report = classification_report(y_true, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    # 保存混淆矩阵和分类报告到Excel
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    with pd.ExcelWriter(os.path.join(result_dir, 'similarity_metrics.xlsx')) as writer:
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
        class_report_df.to_excel(writer, sheet_name='Classification Report')

    print(f"混淆矩阵已保存到 {os.path.join(result_dir, 'similarity_metrics.xlsx')}")
    print("混淆矩阵：")
    print(cm_df)
    print("\n分类报告：")
    print(class_report_df)

    # 输出总运行时间
    total_time = time.time() - start_time
    print(f"\n程序运行完成，总耗时: {total_time:.2f}s")


if __name__ == "__main__":
    main()