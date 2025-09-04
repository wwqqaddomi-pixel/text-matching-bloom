import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_emnist_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    return labels, pixels


def display_image(pixels, index):
    image = pixels[index].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {labels[index]}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    file_path = 'emnist-digits-test.csv'
    labels, pixels = load_emnist_csv(file_path)
    # 显示第 0 个样本的图像
    display_image(pixels, 1)