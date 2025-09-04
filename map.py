import pandas as pd


# 读取 CSV 文件
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


# 定义映射关系
def define_mapping():
    #针对emnist-digits-mapping
    mapping = {
        0: 48,
        1: 49,
        2: 50,
        3: 51,
        4: 52,
        5: 53,
        6: 54,
        7: 55,
        8: 56,
        9: 57,
    }
    return mapping


# 应用映射
def apply_mapping(df, mapping):
    df = df.replace(mapping)
    return df


# 保存修改后的 CSV 文件
def save_csv(df, output_path):
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # 输入文件路径
    input_file_path = 'emnist-digits-test.csv'
    # 输出文件路径
    output_file_path = 'map-emnist-digits-test.csv'
    # 读取 CSV 文件
    df = read_csv(input_file_path)
    # 定义映射关系
    mapping = define_mapping()
    # 应用映射
    df = apply_mapping(df, mapping)
    # 保存修改后的 CSV 文件
    save_csv(df, output_file_path)