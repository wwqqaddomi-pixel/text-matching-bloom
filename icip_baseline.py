import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Tuple


class BF:
    def __init__(self, bf_len, bf_num_hash_func):
        """初始化布隆过滤器的参数"""
        self.bf_len = bf_len
        self.bf_num_hash_func = bf_num_hash_func
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5

    def _hash_value(self, val):
        """计算哈希值"""
        hex_str1 = self.h1(val.encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = self.h2(val.encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    def set_to_bloom_filter(self, val_set):
        """将输入集合转换为布隆过滤器"""
        k = self.bf_num_hash_func
        l = self.bf_len
        bloom_set = bitarray(l)
        bloom_set.setall(False)

        for val in val_set:
            int1, int2 = self._hash_value(val)
            for i in range(k):
                gi = (int1 + i * int2) % l
                bloom_set[gi] = True

        return bloom_set


def main():
    # 设置随机数种子，确保结果可复现
    random.seed(9318)

    # 布隆过滤器参数设置
    bf_len = 400
    bf_num_hash_func = 6
    similarity_threshold = 0.9

    # 定义形状相似字符的替换映射
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
    }

    # 读取数据文件（修改为你的文件路径）
    try:
        # 注意：Windows路径需使用双反斜杠“\\”或在字符串前加r表示原始路径
        data1_path = r"D:\实习\COMP1\icip2\dirty_dblp_acm\test.csv"  # dirty_dblp_acm测试集
        data2_path = r"D:\实习\COMP1\icip2\dirty_dblp_scholar\test.csv"  # 假设dirty_dblp_scholar测试集路径

        # 检查文件是否存在
        for file_path in [data1_path, data2_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取数据
        data1 = pd.read_csv(data1_path)
        data1['source_file'] = 'dirty_dblp_acm'  # 标记来源

        data2 = pd.read_csv(data2_path)
        data2['source_file'] = 'dirty_dblp_scholar'  # 标记来源

        print(f"成功加载dirty_dblp_acm测试集: {len(data1)} 条记录")
        print(f"成功加载dirty_dblp_scholar测试集: {len(data2)} 条记录")

    except FileNotFoundError as e:
        print(f"错误：找不到输入文件 - {str(e)}")
        print("请检查文件路径是否正确")
        return
    except Exception as e:
        print(f"错误：读取数据时发生异常 - {str(e)}")
        return

    # 确保数据包含必要的列（根据你的数据集调整，原数据集中是title/authors/venue/year等）
    required_columns = ['id', 'label', 'left_title', 'left_authors', 'left_venue', 'left_year',
                       'right_title', 'right_authors', 'right_venue', 'right_year']
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

    # 初始化布隆过滤器
    bf = BF(bf_len, bf_num_hash_func)

    print("正在生成布隆过滤器...")

    # 为每个文件的记录计算布隆过滤器（适配新的特征列）
    def process_file(data: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        processed_records = []
        for idx, row in enumerate(data.itertuples()):
            id_value = row.id  # 使用数据集中的id列作为唯一标识

            # 提取需要用于匹配的特征（标题、作者、 venue、年份）
            features = {
                'title': getattr(row, 'left_title' if 'left_title' in data.columns else 'right_title'),
                'authors': getattr(row, 'left_authors' if 'left_authors' in data.columns else 'right_authors'),
                'venue': getattr(row, 'left_venue' if 'left_venue' in data.columns else 'right_venue'),
                'year': getattr(row, 'left_year' if 'left_year' in data.columns else 'right_year'),
                'source_file': source
            }

            # 生成特征的替换变体（处理拼写变体）
            all_features = []
            for col, value in features.items():
                if col != 'source_file' and pd.notna(value):  # 跳过空值
                    all_features.extend(generate_replacements(str(value)))

            # 生成布隆过滤器
            old_bloom_filter = features_to_bloom_filter(all_features)

            processed_records.append({
                'id': id_value,
                'row_index': idx,
                'old_bloom_filter': old_bloom_filter,
                'features': features
            })
        return processed_records

    # 处理两个数据集
    file1_bloom_filters = process_file(data1, 'dirty_dblp_acm')
    file2_bloom_filters = process_file(data2, 'dirty_dblp_scholar')

    print("正在计算相似度...")
    similar_pairs = []
    all_pairs = []  # 存储所有记录对的真实标签和预测结果

    # 初始化相似度矩阵
    num_records1 = len(file1_bloom_filters)
    num_records2 = len(file2_bloom_filters)
    similarity_matrix = [[0.0 for _ in range(num_records2)] for _ in range(num_records1)]

    for i, rec1 in enumerate(file1_bloom_filters):
        for j, rec2 in enumerate(file2_bloom_filters):
            # 计算相似度
            similarity = calc_old_bf_similarity(
                rec1['old_bloom_filter'],
                rec2['old_bloom_filter']
            )
            similarity_matrix[i][j] = similarity

            # 真实标签：根据数据集的label列（需确认你的数据集中是否有该列）
            # 若没有label列，可根据id是否相同判断（假设相同id为匹配）
            true_label = 1 if rec1['id'] == rec2['id'] else 0
            # 预测标签：基于相似度阈值
            pred_label = 1 if similarity >= similarity_threshold else 0

            all_pairs.append({
                'true_label': true_label,
                'pred_label': pred_label,
                'similarity': similarity
            })

            # 保存相似对
            if similarity >= similarity_threshold:
                pair_record = {
                    'id_1': rec1['id'],
                    'id_2': rec2['id'],
                    'similarity': similarity,
                    'same_id': 1 if rec1['id'] == rec2['id'] else 0,
                }

                # 添加特征详情
                for attr in ['title', 'authors', 'venue', 'year', 'source_file']:
                    pair_record[f'rec1_{attr}'] = rec1['features'][attr]
                    pair_record[f'rec2_{attr}'] = rec2['features'][attr]

                similar_pairs.append(pair_record)

    print(f"找到 {len(similar_pairs)} 对相似记录")

    # 保存结果到本地（创建results文件夹）
    result_dir = os.path.join(os.path.dirname(data1_path), 'results')  # 结果保存在数据同级的results目录
    os.makedirs(result_dir, exist_ok=True)

    # 保存相似度矩阵
    matrix_df = pd.DataFrame(similarity_matrix)
    matrix_path = os.path.join(result_dir, 'similarity_matrix.csv')
    matrix_df.to_csv(matrix_path, index=False, header=False)
    print(f"相似度矩阵已保存到：{matrix_path}")

    # 保存相似记录对
    try:
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            pairs_path = os.path.join(result_dir, 'cross_file_similar_records.xlsx')
            pairs_df.to_excel(pairs_path, index=False)
            print(f"相似记录对已保存到 {pairs_path}")
            print(f"其中 {sum(p['same_id'] for p in similar_pairs)} 对记录具有相同的ID")
        else:
            print("未找到相似记录对")
    except Exception as e:
        print(f"错误：保存结果时发生异常 - {str(e)}")

    # 生成混淆矩阵和分类报告
    print("正在生成评估指标...")
    y_true = [pair['true_label'] for pair in all_pairs]
    y_pred = [pair['pred_label'] for pair in all_pairs]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['真实-不相似', '真实-相似'], columns=['预测-不相似', '预测-相似'])

    class_report = classification_report(y_true, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()

    # 保存评估指标
    metrics_path = os.path.join(result_dir, 'similarity_metrics.xlsx')
    with pd.ExcelWriter(metrics_path) as writer:
        cm_df.to_excel(writer, sheet_name='混淆矩阵')
        class_report_df.to_excel(writer, sheet_name='分类报告')

    print(f"评估指标已保存到 {metrics_path}")
    print("混淆矩阵：")
    print(cm_df)
    print("\n分类报告：")
    print(class_report_df)


if __name__ == "__main__":
    main()