import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple


class BF:
    def __init__(self, bf_len, bf_num_hash_func, bf_num_inter, bf_step,
                 max_abs_diff, min_val, max_val, q, weight_file):
        """
        初始化布隆过滤器的参数
        """
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
        """
        加载权重文件
        """
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
        """
        计算哈希值
        """
        hex_str1 = self.h1(val.encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(val.encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    def set_to_bloom_filter(self, val_set, original_set):
        """
        将输入集合转换为布隆过滤器
        """
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
        """
        计算转换后集合多出来的交集
        """
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
        """
        计算两个布隆过滤器的新型相似度
        """
        bf1_1s = bf1.count()
        bf2_1s = bf2.count()
        common_1s = (bf1 & bf2).count()
        extra_common_1s = self.get_extra_common_1s(extra_positions1, extra_positions2,
                                                   extra_positions_with_chars1,
                                                   extra_positions_with_chars2)
        dice_sim = (2.0 * (common_1s + 0.1*extra_common_1s)) / (bf1_1s + bf2_1s + 0.1*extra_common_1s)
        return dice_sim


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

    # 定义形状相似字符的替换映射（仅包含26个英文字母大小写和数字）
    similar_char_mapping = {
        '0': ['O', 'o'],
        '1': ['I', 'i', 'l', 'L'],
        '2': ['Z', 'z'],
        '3': ['E', 'e'],
        '4': ['A', 'a'],
        '5': ['S', 's'],
        '6': ['G', 'g', 'b'],
        '7': ['T', 't'],
        '8': ['B', 'b'],
        '9': ['q', 'g'],
        'A': ['4', 'a'],
        'B': ['8', 'b'],
        'C': ['G', 'g', 'c'],
        'D': ['O', 'o', '0'],
        'E': ['3', 'e'],
        'F': ['f'],
        'G': ['C', 'c', '6', 'g'],
        'H': ['h'],
        'I': ['1', 'i', 'l', 'L'],
        'J': ['j'],
        'K': ['k'],
        'L': ['1', 'I', 'i', 'l'],
        'M': ['m'],
        'N': ['n'],
        'O': ['0', 'o', 'D', 'd'],
        'P': ['p'],
        'Q': ['q', '9'],
        'R': ['r'],
        'S': ['5', 's'],
        'T': ['7', 't'],
        'U': ['u'],
        'V': ['v'],
        'W': ['w'],
        'X': ['x'],
        'Y': ['y'],
        'Z': ['2', 'z'],
        'a': ['A', '4'],
        'b': ['B', '8', '6', 'g'],
        'c': ['C', 'G', 'g'],
        'd': ['D', 'O', 'o', '0'],
        'e': ['E', '3'],
        'f': ['F'],
        'g': ['G', 'C', 'c', '6', '9', 'q', 'b'],
        'h': ['H'],
        'i': ['I', '1', 'l', 'L'],
        'j': ['J'],
        'k': ['K'],
        'l': ['L', '1', 'I', 'i'],
        'm': ['M'],
        'n': ['N'],
        'o': ['O', '0', 'D', 'd'],
        'p': ['P'],
        'q': ['Q', '9', 'g'],
        'r': ['R'],
        's': ['S', '5'],
        't': ['T', '7'],
        'u': ['U'],
        'v': ['V'],
        'w': ['W'],
        'x': ['X'],
        'y': ['Y'],
        'z': ['Z', '2']
    }

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

    # 生成可能的字符替换
    def generate_replacements(value: Any) -> List[str]:
        if not isinstance(value, str):
            return [str(value)]
        replacements = [value]
        for i, char in enumerate(value):
            if char in similar_char_mapping:
                for repl in similar_char_mapping[char]:
                    new_val = value[:i] + repl + value[i + 1:]
                    replacements.append(new_val)
        return replacements

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
                    all_features.extend(generate_replacements(value))
            old_bloom_filter = features_to_bloom_filter(all_features)

            # 新型布隆过滤器处理方式
            val_set = []
            original_set = []
            for col in required_columns:
                value = getattr(row, col)
                if isinstance(value, str):
                    substrings = [value[i:i + q] for i in range(len(value) - (q - 1))]
                    original_set.extend(substrings)

                    # 生成替换后的子字符串
                    for i in range(len(value)):
                        char = value[i]
                        if char in similar_char_mapping:
                            for replacement in similar_char_mapping[char]:
                                new_s = value[:i] + replacement + value[i + 1:]
                                new_substrings = [new_s[i:i + q] for i in range(len(new_s) - (q - 1))]
                                val_set.extend(new_substrings)
            val_set = list(set(val_set + original_set))  # 去重

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
    old_similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]  # 修正：行数应为num_records1

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