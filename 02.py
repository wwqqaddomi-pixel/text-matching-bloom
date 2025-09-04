import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
import numpy as np
from typing import List, Dict, Any, Tuple


def main():
    # 设置随机数种子，确保结果可复现
    random.seed(9318)

    # 布隆过滤器参数设置
    bf_len = 50
    bf_num_hash_func = 2
    similarity_threshold = 0.8

    # 读取数据文件
    try:
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建数据文件路径
        data1_path = os.path.join(script_dir, 'data',
                                  'D:\实习\COMP1\COMP3850_PPRL\\ncvr_numrec_100_modrec_2_ocp_20_myp_0_nump_5.csv')
        data2_path = os.path.join(script_dir, 'data',
                                  'D:\实习\COMP1\COMP3850_PPRL\\ncvr_numrec_100_modrec_2_ocp_20_myp_1_nump_5.csv')

        # 检查文件是否存在
        for file_path in [data1_path, data2_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取CSV文件并添加来源标识
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

    # 定义形状相似字符的替换映射
    similar_char_mapping = {
        '0': ['o', 'O'],
        '1': ['i', 'l', 'I'],
        '2': ['z', 'Z'],
        '5': ['s', 'S'],
        '6': ['b', 'G'],
        '8': ['B'],
        '9': ['q', 'g'],
        'b': ['h', '6'],
        'B': ['8'],
        'c': ['C'],
        'C': ['G'],
        'd': ['D'],
        'D': ['O', '0'],
        'i': ['1', 'l'],
        'I': ['1', 'l'],
        'j': ['i'],
        'J': ['I'],
        'l': ['1', 'i', 'I'],
        'L': ['1', 'I'],
        'o': ['0', 'O'],
        'O': ['0', 'o'],
        'q': ['9'],
        's': ['5'],
        'S': ['5'],
        'z': ['2'],
        'Z': ['2'],
    }

    # 哈希函数
    def _hash_value(val: Any) -> Tuple[int, int]:
        """计算输入值的两个哈希值"""
        h1 = hashlib.sha1
        h2 = hashlib.md5
        hex_str1 = h1(str(val).encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = h2(str(val).encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    # 将特征转换为布隆过滤器
    def features_to_bloom_filter(features: List[Any]) -> bitarray:
        """将特征列表转换为布隆过滤器"""
        bloom = bitarray(bf_len)
        bloom.setall(False)

        for feature in features:
            int1, int2 = _hash_value(feature)
            for i in range(bf_num_hash_func):
                gi = (int1 + i * int2) % bf_len
                bloom[gi] = True

        return bloom

    # 计算两个布隆过滤器的相似度
    def calc_bf_similarity(bf1: bitarray, bf2: bitarray) -> float:
        """计算两个布隆过滤器的Dice系数相似度"""
        count1 = bf1.count()
        count2 = bf2.count()
        common = (bf1 & bf2).count()

        if count1 + count2 == 0:
            return 0.0

        return (2.0 * common) / (count1 + count2)

    # 生成可能的字符替换
    def generate_replacements(value: Any) -> List[str]:
        """生成值的可能替换列表"""
        if not isinstance(value, str):
            return [str(value)]

        replacements = [value]
        for i, char in enumerate(value):
            if char in similar_char_mapping:
                for repl in similar_char_mapping[char]:
                    new_val = value[:i] + repl + value[i + 1:]
                    replacements.append(new_val)
        return replacements

    # 为每个文件的记录计算布隆过滤器
    print("正在生成布隆过滤器...")

    file1_bloom_filters: List[Dict[str, Any]] = []
    for idx, row in enumerate(data1.itertuples()):
        id_value = row[1]  # 假设第一列是ID

        features = {
            'givenname': getattr(row, 'givenname'),
            'surname': getattr(row, 'surname'),
            'suburb': getattr(row, 'suburb'),
            'postcode': getattr(row, 'postcode'),
            'source_file': 'file1'
        }

        all_features = []
        for col, value in features.items():
            if col != 'source_file':
                all_features.extend(generate_replacements(value))

        bloom_filter = features_to_bloom_filter(all_features)

        file1_bloom_filters.append({
            'id': id_value,
            'row_index': idx,
            'bloom_filter': bloom_filter,
            'features': features
        })

    file2_bloom_filters: List[Dict[str, Any]] = []
    for idx, row in enumerate(data2.itertuples()):
        id_value = row[1]  # 假设第一列是ID

        features = {
            'givenname': getattr(row, 'givenname'),
            'surname': getattr(row, 'surname'),
            'suburb': getattr(row, 'suburb'),
            'postcode': getattr(row, 'postcode'),
            'source_file': 'file2'
        }

        all_features = []
        for col, value in features.items():
            if col != 'source_file':
                all_features.extend(generate_replacements(value))

        bloom_filter = features_to_bloom_filter(all_features)

        file2_bloom_filters.append({
            'id': id_value,
            'row_index': idx,
            'bloom_filter': bloom_filter,
            'features': features
        })

    # 创建相似度矩阵 (100x100) - 只比较文件1和文件2的记录
    print("正在计算相似度矩阵...")
    matrix_size = min(100, len(file1_bloom_filters), len(file2_bloom_filters))
    similarity_matrix = np.zeros((matrix_size, matrix_size))

    # 计算相似度矩阵（文件1的记录与文件2的记录比较）
    for i in range(matrix_size):
        for j in range(matrix_size):
            sim = calc_bf_similarity(
                file1_bloom_filters[i]['bloom_filter'],
                file2_bloom_filters[j]['bloom_filter']
            )
            similarity_matrix[i, j] = sim

    # 查找相似的记录对
    print("正在查找相似记录对...")
    similar_pairs = []

    for i in range(matrix_size):
        for j in range(matrix_size):
            if similarity_matrix[i, j] >= similarity_threshold:
                # 获取两个记录的详细信息
                rec1 = file1_bloom_filters[i]
                rec2 = file2_bloom_filters[j]

                # 创建相似对记录
                pair_record = {
                    'id_1': rec1['id'],
                    'id_2': rec2['id'],
                    'similarity': similarity_matrix[i, j]
                }

                # 添加两个记录的详细特征
                for attr in ['givenname', 'surname', 'suburb', 'postcode', 'source_file']:
                    pair_record[f'rec1_{attr}'] = rec1['features'][attr]
                    pair_record[f'rec2_{attr}'] = rec2['features'][attr]

                similar_pairs.append(pair_record)

    # 保存结果
    print("正在保存结果...")
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    try:
        # 保存相似度矩阵
        matrix_df = pd.DataFrame(similarity_matrix)
        matrix_path = os.path.join(result_dir, 'cross_file_similarity_matrix.csv')
        matrix_df.to_csv(matrix_path, index=False)
        print(f"相似度矩阵已保存到 {matrix_path}")

        # 保存相似记录对
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            pairs_path = os.path.join(result_dir, 'cross_file_similar_records.xlsx')
            pairs_df.to_excel(pairs_path, index=False)
            print(f"找到 {len(similar_pairs)} 对相似记录，已保存到 {pairs_path}")
        else:
            print("未找到相似记录对")

    except Exception as e:
        print(f"错误：保存结果时发生异常 - {str(e)}")


if __name__ == "__main__":
    main()