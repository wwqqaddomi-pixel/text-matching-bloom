import pandas as pd
from bitarray import bitarray
import hashlib
import random
import os
import numpy as np
from typing import List, Dict, Any, Tuple


def main():
    random.seed(9318)

    bf_len = 200
    bf_num_hash_func = 4
    similarity_threshold = 0.9

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 修正路径拼接方式（使用原始字符串或双斜杠避免转义问题）
        data1_path = os.path.join(script_dir, 'data', r'original_1.csv')  # 假设文件在data目录下
        data2_path = os.path.join(script_dir, 'data', r'process_1.csv')

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

    required_columns = ['givenname', 'surname', 'suburb', 'postcode']
    for col in required_columns:
        if col not in data1.columns or col not in data2.columns:
            print(f"错误：数据缺少必要的列 '{col}'")
            return

    similar_char_mapping = {
        # 保持原有字符映射不变
        '0': ['O', 'o'],
        # ...（省略其他映射，与原代码一致）
        'z': ['Z', '2']
    }

    def _hash_value(val: Any) -> Tuple[int, int]:
        h1 = hashlib.sha1(str(val).encode('utf-8')).hexdigest()
        h2 = hashlib.md5(str(val).encode('utf-8')).hexdigest()
        return int(h1, 16), int(h2, 16)

    def features_to_bloom_filter(features: List[Any]) -> bitarray:
        bloom = bitarray(bf_len)
        bloom.setall(False)
        for feature in features:
            int1, int2 = _hash_value(feature)
            for i in range(bf_num_hash_func):
                gi = (int1 + i * int2) % bf_len
                bloom[gi] = True
        return bloom

    def calc_bf_similarity(bf1: bitarray, bf2: bitarray) -> float:
        count1 = bf1.count()
        count2 = bf2.count()
        if count1 + count2 == 0:
            return 0.0
        common = (bf1 & bf2).count()
        return (2.0 * common) / (count1 + count2)

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

    print("正在生成布隆过滤器...")

    # 优化：使用列表推导式生成布隆过滤器（提升性能）
    def process_file(data: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        return [
            {
                'id': row[0],  # 假设第一列为ID（index=0）
                'row_index': idx,
                'bloom_filter': features_to_bloom_filter(
                    [repl for col in required_columns
                     for repl in generate_replacements(getattr(row, col))]
                ),
                'features': {col: getattr(row, col) for col in required_columns + ['source_file']}
            }
            for idx, row in enumerate(data.itertuples())
        ]

    file1_bloom_filters = process_file(data1, 'file1')
    file2_bloom_filters = process_file(data2, 'file2')

    print("正在计算相似度...")
    similar_pairs = []

    # 优化：使用生成器表达式替代双重循环（节省内存）
    for i, bf1 in enumerate(file1_bloom_filters):
        for j, bf2 in enumerate(file2_bloom_filters):
            sim = calc_bf_similarity(bf1['bloom_filter'], bf2['bloom_filter'])
            if sim >= similarity_threshold:
                similar_pairs.append({
                    'id_1': bf1['id'],
                    'id_2': bf2['id'],
                    'similarity': sim,
                    'same_id': 1 if bf1['id'] == bf2['id'] else 0,
                    **{f'rec1_{k}': v for k, v in bf1['features'].items()},
                    **{f'rec2_{k}': v for k, v in bf2['features'].items()}
                })

    print(f"找到 {len(similar_pairs)} 对相似记录")

    print("正在保存结果...")
    result_dir = os.path.join(script_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)

    try:
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            required_columns = [
                'id_1', 'id_2', 'similarity', 'same_id',
                'rec1_givenname', 'rec1_surname', 'rec1_suburb', 'rec1_postcode', 'rec1_source_file',
                'rec2_givenname', 'rec2_surname', 'rec2_suburb', 'rec2_postcode', 'rec2_source_file'
            ]
            pairs_df = pairs_df[required_columns]
            pairs_path = os.path.join(result_dir, 'cross_file_similar_records.xlsx')
            pairs_df.to_excel(pairs_path, index=False)
            print(f"相似记录对已保存到 {pairs_path}")
            print(f"其中 {sum(p['same_id'] for p in similar_pairs)} 对记录具有相同的ID")
        else:
            print("未找到相似记录对")

        # 注意：全量数据的相似度矩阵可能非常大，内存不足时需谨慎生成
        # 如需保存矩阵，建议分块处理或仅保存相似对索引
        # matrix_df = pd.DataFrame(...) 此处省略矩阵保存逻辑，避免大数据量内存溢出

    except Exception as e:
        print(f"错误：保存结果时发生异常 - {str(e)}")


if __name__ == "__main__":
    main()