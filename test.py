import numpy as np
import cv2
import csv


def csv_to_gray_image(csv_file, output_image):
    # 从 CSV 文件读取数据
    data = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_row = [float(item) for item in row]
            data.append(data_row)
    # 将数据转换为 numpy 数组
    data_array = np.array(data)
    # 归一化数据到 0-255 的范围
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    normalized_data = ((data_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    # 将数据转换为灰度图像
    gray_image = cv2.cvtColor(normalized_data, cv2.COLOR_GRAY2BGR)
    # 保存图像
    cv2.imwrite(output_image, gray_image)


# 调用函数
csv_to_gray_image('map-emnist-digits-test.csv', 'map-emnist-digits-test.jpg')