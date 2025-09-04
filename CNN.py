import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.keras.datasets import emnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout #二维卷积层，二维最大池化层，Flatten将多维的输入数据压平为一维数组，全连接层，在训练过程中随机将部分神经元的输出置为 0，防止过拟合
from tensorflow.keras.utils import to_categorical #分类函数


# 加载 EMNIST 数据集
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # 第一列是标签，其余列是图像像素
    labels = df.iloc[:, 0].values
    pixels = df.iloc[:, 1:].values
    # 将像素数据重塑为 (样本数, 28, 28, 1) 并归一化到 [0, 1] 范围
    x = pixels.reshape(pixels.shape[0], 28, 28, 1) / 255.0
    # 将标签转换为 one-hot 编码
    y = to_categorical(labels)
    return x, y


# 构建 CNN 模型
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    # 确保输出层的 units 与标签的类别数一致
    model.add(Dense(47, activation='softmax'))  # 这里的 47 是根据 num_classes 来的，根据实际情况修改
    return model



# 训练模型
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))


# 评估模型
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}")


if __name__ == "__main__":
    
    #csv_file_path = 'emnist-balanced-test.csv'
    x_test, y_test = load_data('emnist-balanced-test.csv')

    #csv_file_path = 'emnist-balanced-train.csv'
    x_train, y_train = load_data('emnist-balanced-train.csv')

    # 构建模型
    model = build_model()
    # 训练模型
    train_model(model, x_train, y_train, x_test, y_test)
    # 评估模型
    evaluate_model(model, x_test, y_test)