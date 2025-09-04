import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer



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

    def calc_bf_sim(self, bf1, bf2, extra_positions1, extra_positions2, extra_positions_with_chars1,
                    extra_positions_with_chars2):
        """计算两个布隆过滤器的新型相似度"""
        bf1_1s = bf1.count()
        bf2_1s = bf2.count()
        common_1s = (bf1 & bf2).count()

        # 计算每个布隆过滤器的额外位置总权重
        def calculate_total_weights(positions, positions_with_chars):
            total_weight = 0
            for pos in set(positions):
                if pos in positions_with_chars:
                    for chars in positions_with_chars[pos]:
                        if chars and len(chars) > 0:
                            # 假设chars是一个字符串列表，取第一个字符作为原始字符
                            original_char = chars[0][0]
                            # 遍历所有可能的替换字符
                            for replacement_char in chars:
                                if replacement_char:
                                    key = f"{original_char}-{replacement_char[0]}"
                                    if key in self.weights:
                                        total_weight += self.weights[key]
            return total_weight

        # 计算两个布隆过滤器的额外位置总权重
        total_weight1 = calculate_total_weights(extra_positions1, extra_positions_with_chars1)
        total_weight2 = calculate_total_weights(extra_positions2, extra_positions_with_chars2)

        # 计算平均权重作为调整因子
        avg_weight_factor = (total_weight1 + total_weight2) / 2.0
        # 确保调整因子在合理范围内
        weight_adjustment = min(0.95, avg_weight_factor)  # 限制最大调整幅度

        # 应用权重调整
        adjusted_common_1s = common_1s * (1 - weight_adjustment)
        adjusted_bf1_1s = bf1_1s * (1 - weight_adjustment)
        adjusted_bf2_1s = bf2_1s * (1 - weight_adjustment)

        extra_common_1s = self.get_extra_common_1s(extra_positions1, extra_positions2,
                                                   extra_positions_with_chars1,
                                                   extra_positions_with_chars2)

        # 使用调整后的值计算Dice相似度
        dice_sim = (2.0 * (adjusted_common_1s + 0.1 * extra_common_1s)) / (
                adjusted_bf1_1s + adjusted_bf2_1s + 0.1 * extra_common_1s)
        return dice_sim


class ByT5QGramGenerator:
    def __init__(self, model_size="google/byt5-base", device="cuda" if torch.cuda.is_available() else "cpu"):
        """初始化ByT5模型用于生成q-gram变体"""
        print(f"加载ByT5模型 ({model_size}) 到 {device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_size)
        self.model = T5ForConditionalGeneration.from_pretrained(model_size).to(device)
        self.device = device
        print("ByT5模型加载完成")

    def generate_variants(self, text: str, q: int = 2, max_length: int = 50) -> List[str]:
        """
        生成文本的q-gram及其变体
        text: 输入文本
        q: q-gram长度
        max_length: 生成文本的最大长度
        """
        # 生成原始q-grams
        original_qgrams = [text[i:i + q] for i in range(len(text) - q + 1)]

        # 使用ByT5生成可能的变体
        input_text = f"generate similar qgrams: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        # 使用beam search生成多个可能的变体
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            num_return_sequences=3,
            temperature=0.7
        )

        # 解码生成的变体
        variants = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # 从变体中提取q-grams
        variant_qgrams = []
        for variant in variants:
            variant_qgrams.extend([variant[i:i + q] for i in range(len(variant) - q + 1)])

        # 合并原始和变体q-grams并去重
        all_qgrams = list(set(original_qgrams + variant_qgrams))
        return all_qgrams


def main():
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
    weight_file = 'error_rate_results_CNN_emnist_emnist-byclass_2025-03-18.csv'

    # 初始化ByT5 q-gram生成器
    byt5_generator = ByT5QGramGenerator(model_size="google/byt5-base")

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

    print("正在生成布隆过滤器...")

    # 为每个文件的记录计算布隆过滤器（同时支持新旧两种方式）
    def process_file(data: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        processed_records = []
        for idx, row in enumerate(data.itertuples()):
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
            old_bloom_filter = features_to_bloom_filter(all_features)

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

            new_bloom_filter, extra_positions, extra_positions_with_chars = new_bf.set_to_bloom_filter(
                val_set, original_set)

            processed_records.append({
                'id': id_value,
                'row_index': idx,
                'old_bloom_filter': old_bloom_filter,
                'new_bloom_filter': new_bloom_filter,
                'extra_positions': extra_positions,
                'extra_positions_with_chars': extra_positions_with_chars,
                'features': features,
                'val_set': val_set,
                'original_set': original_set
            })

            # 进度显示
            if (idx + 1) % 100 == 0:
                print(f"已处理 {source} 文件中的 {idx + 1}/{len(data)} 条记录")

        return processed_records

    file1_bloom_filters = process_file(data1, 'file1')
    file2_bloom_filters = process_file(data2, 'file2')

    print("正在计算相似度...")
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

    for i, rec1 in enumerate(file1_bloom_filters):
        for j, rec2 in enumerate(file2_bloom_filters):
            # 计算原始相似度
            old_sim = calc_old_bf_similarity(
                rec1['old_bloom_filter'],
                rec2['old_bloom_filter']
            )

            # 计算新型相似度
            new_sim = new_bf.calc_bf_sim(
                rec1['new_bloom_filter'],
                rec2['new_bloom_filter'],
                rec1['extra_positions'],
                rec2['extra_positions'],
                rec1['extra_positions_with_chars'],
                rec2['extra_positions_with_chars']
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

            # 更新进度
            processed_pairs += 1
            if processed_pairs % report_interval == 0:
                progress = processed_pairs / total_pairs * 100
                print(f"相似度计算进度: {progress:.1f}%")

    print(f"找到 {len(similar_pairs)} 对相似记录")

    # 保存相似度矩阵
    print("正在保存相似度矩阵...")
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
    print("正在保存结果...")
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

    print("正在生成混淆矩阵...")
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


if __name__ == "__main__":
    main()