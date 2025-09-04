import pandas as pd
from bitarray import bitarray
import hashlib
import random


def main():
    # 设置随机数种子，确保结果可复现
    random.seed(9318)

    # 布隆过滤器参数设置
    bf_len = 50
    bf_num_hash_func = 2
    similarity_threshold = 0.8

    # 读取数据文件 - 需要替换为实际文件路径
    try:
        # 假设文件格式为 CSV，第一列是 ID，后续列是特征
        data1 = pd.read_csv('D:\实习\COMP1\COMP3850_PPRL\\ncvr_numrec_100_modrec_2_ocp_20_myp_0_nump_5.csv')
        data2 = pd.read_csv('D:\实习\COMP1\COMP3850_PPRL\\ncvr_numrec_100_modrec_2_ocp_20_myp_1_nump_5.csv')
        data = pd.concat([data1, data2])
    except FileNotFoundError:
        print("错误：找不到输入文件，请检查文件路径。")
        return

    # 确保数据至少有两列（ID列和特征列）
    if data.shape[1] < 2:
        print("错误：数据格式不正确，至少需要ID列和一个特征列。")
        return

    # 定义形状相似字符的替换映射（简化版）
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
    def _hash_value(val):
        """计算输入值的两个哈希值"""
        h1 = hashlib.sha1
        h2 = hashlib.md5
        hex_str1 = h1(str(val).encode('utf-8')).hexdigest()
        int1 = int(hex_str1, 16)
        hex_str2 = h2(str(val).encode('utf-8')).hexdigest()
        int2 = int(hex_str2, 16)
        return int1, int2

    # 将特征转换为布隆过滤器
    def features_to_bloom_filter(features):
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
    def calc_bf_similarity(bf1, bf2):
        """计算两个布隆过滤器的Dice系数相似度"""
        count1 = bf1.count()
        count2 = bf2.count()
        common = (bf1 & bf2).count()

        if count1 + count2 == 0:
            return 0.0

        return (2.0 * common) / (count1 + count2)

    # 生成可能的字符替换（简化版）
    def generate_replacements(value):
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

    # 为每个ID计算布隆过滤器
    id_bloom_filters = {}

    for _, row in data.iterrows():
        id_value = row.iloc[0]
        features = row.iloc[1:].tolist()

        # 生成特征及其替换的布隆过滤器
        all_features = []
        for feature in features:
            all_features.extend(generate_replacements(feature))

        bloom_filter = features_to_bloom_filter(all_features)

        if id_value not in id_bloom_filters:
            id_bloom_filters[id_value] = []

        id_bloom_filters[id_value].append({
            'row_index': _,
            'bloom_filter': bloom_filter,
            'original_features': features
        })

    # 计算ID相同的记录之间的相似度
    similarity_results = []

    for id_value, records in id_bloom_filters.items():
        if len(records) < 2:
            continue

        # 比较每对记录
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                sim = calc_bf_similarity(
                    records[i]['bloom_filter'],
                    records[j]['bloom_filter']
                )

                is_similar = 1 if sim >= similarity_threshold else 0

                similarity_results.append({
                    'id': id_value,
                    'row_index_1': records[i]['row_index'],
                    'row_index_2': records[j]['row_index'],
                   'similarity': sim,
                    'is_similar': is_similar
                })

    # 保存结果到Excel文件
    if similarity_results:
        result_df = pd.DataFrame(similarity_results)
        result_df.to_excel('similarity_results.xlsx', index=False)
        print(f"处理完成，结果已保存到similarity_results.xlsx文件")
        print(f"共找到{len(similarity_results)}对需要比较的记录")
        print(f"其中{result_df['is_similar'].sum()}对被判定为相似")
    else:
        print("没有找到需要比较的ID重复记录")


if __name__ == "__main__":
    main()