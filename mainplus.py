import torch #PyTorch 深度学习框架的核心库  张量操作
import torch.nn as nn #定义神经网络模块
import torch.optim as optim # 优化器
import torchvision.transforms as transforms #图像数据的预处理
from torch.utils.data import Dataset, DataLoader #分别用于自定义数据集和批量加载数据
import numpy as np #数值计算
import pandas as pd #读取和处理 CSV 文件
import matplotlib.pyplot as plt #数据可视化
import seaborn as sns #基于 matplotlib 的统计数据可视化库，用于绘制混淆矩阵
from sklearn.metrics import confusion_matrix #计算混淆矩阵
import csv
import datetime
import os



# Check Apple Metal MPS backend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# EMNIST character labels full label:(62 classes)
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
] #定义 EMNIST 数据集的类别标签

# Define dataset class for PyTorch
class EMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):#初始化数据集，接收图像数据、标签数据和可选的预处理转换。
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):#返回数据集的长度
        return len(self.labels)

    def __getitem__(self, idx):#根据索引返回对应的图像和标签，
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:#如果有预处理转换则对图像进行转换。
            image = self.transform(image)

        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(), #将图像数据转换为 PIL 图像格式
    transforms.Grayscale(num_output_channels=1), #将图像转换为单通道灰度图像
    transforms.ToTensor(), #将 PIL 图像转换为 PyTorch 张量
    transforms.Normalize((0.5,), (0.5,)) #对图像张量进行归一化处理，将像素值缩放到 [-1, 1] 范围
])

# Load datasets
batch_size = 64 #批量大小
emnist_data_path = "D:\\实习\\EMNIST"   # file location
train_df = pd.read_csv(emnist_data_path + '\emnist-byclass-train.csv', header=None) #读取
test_df = pd.read_csv(emnist_data_path + '\emnist-byclass-test.csv', header=None)
#指定了 header=None，意味着文件中没有表头行，pandas 会将第一行数据当作普通的数据行进行处理。

# train data
train_labels = train_df.iloc[:, 0].values #提取标签
train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
#提取图像数据并将图像数据转换为合适的形状（28x28）和数据类型（np.uint8）
train_dataset = EMNISTDataset(train_images, train_labels, transform=transform)
#transform 是一个 torchvision.transforms 定义的预处理转换序列，例如可以进行图像的缩放、归一化等操作
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#DataLoader 是 PyTorch 中用于批量加载数据的工具类，它可以将数据集对象封装成一个可迭代的数据加载器，方便在训练模型时按批次获取数据。
#batch_size 是一个整数，表示每个批次加载的数据样本数量。在训练过程中，模型会按批次处理数据，这样可以提高训练效率。
#shuffle=True 表示在每个训练周期开始时，会对数据进行随机打乱，这样可以增加数据的随机性，有助于模型更好地学习数据的特征。

# test data
test_labels = test_df.iloc[:, 0].values
test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
test_dataset = EMNISTDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()#调用父类 nn.Module 的构造函数，确保子类能够正确初始化
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #输入通道数为 1，意味着输入的图像是单通道的。输出通道数为 32，即经过该卷积层后会生成 32 个特征图。
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128) #28/2/2=7 ； 7-1）/2+1=3
        self.fc2 = nn.Linear(128, 62)  # 62 output classes (A-Z, a-z, 0-9)  # last output layer, number of clusters
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))#x 依次进行卷积、ReLU 激活函数和最大池化操作
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)  # Flatten 是将卷积和池化后的多维特征图展平为一维向量，-1 表示自动计算该维度的大小，以保证总元素数量不变
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model
model = CNN().to(device) #创建 CNN 模型实例，并将其移动到指定的设备（CPU 或 MPS）上

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss() #使用交叉熵损失函数作为模型的损失函数。
optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam 优化器来更新模型的参数，学习率设置为 0.001

# Training loop
num_epochs = 10 #定义训练的总轮数
train_losses = [] #存储每一轮训练的平均损失
test_accuracies = [] #存储每一轮测试的准确率

for epoch in range(num_epochs):
    model.train() #将模型设置为训练模式，这会启用一些在训练时需要的特殊层，如 Dropout 层
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) #将图像和标签数据移动到指定的设备
        optimizer.zero_grad() #清空优化器中的梯度信息，避免梯度累积
        outputs = model(images) #前向传播，将图像输入模型得到预测输出
        loss = criterion(outputs, labels) #计算预测输出和真实标签之间的损失
        loss.backward() #计算损失相对于模型参数的梯度
        optimizer.step() #根据计算得到的梯度更新模型的参数

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader) #计算当前轮次的平均训练损失，并将其添加到 train_losses 列表中
    train_losses.append(avg_loss)

    # Evaluate model
    model.eval() #将模型设置为评估模式，这会禁用一些在训练时需要的特殊层，如 Dropout 层
    correct = 0 #记录模型预测正确的样本数量
    total = 0 #用于记录总共预测的样本数量
    all_preds = [] #用于存储模型对所有样本的预测结果
    all_true = [] #用于存储所有样本的真实标签
    error_indices = []  # 存储错误样本的索引
    error_true_labels = []  # 存储错误样本的真实标签
    error_pred_labels = []  # 存储错误样本的预测标签

    with torch.no_grad():
#这是一个上下文管理器，用于关闭 PyTorch 的自动求导机制。在评估阶段，我们只关心模型的预测结果，不需要计算梯度，关闭自动求导可以减少内存消耗，提高计算速度。在这个上下文管理器内部的所有操作都不会进行梯度计算。
        for i, (images, labels) in enumerate(test_loader): #test_loader：是一个数据加载器，它会按照指定的批次大小将测试数据集分成多个批次进行加载。
            images, labels = images.to(device), labels.to(device) #将图像数据和对应的标签数据移动到指定的设备（如 GPU 或 CPU）上，确保数据和模型在同一设备上进行计算。
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) #对 outputs 张量在维度 1（即类别维度）上求最大值。该函数返回两个张量，第一个张量是最大值本身，第二个张量是最大值所在的索引
            total += labels.size(0) #将当前批次的样本数量累加到 total 中
            correct += (predicted == labels).sum().item() # (predicted == labels).sum()统计预测正确的样本数量，返回一个标量张量
            #(predicted == labels).sum().item()：将标量张量转换为 Python 标量，并累加到 correct 中

            all_preds.extend(predicted.cpu().numpy())#predicted.cpu().numpy()：将预测标签从GPU(如果使用了GPU)移动到CPU上，并转换为NumPy数组。
            all_true.extend(labels.cpu().numpy())
            #将当前批次的预测标签和真实标签添加到 all_preds 和 all_true 列表中。

            #找到错误样本
            batch_errors = (predicted != labels).cpu().numpy()
            batch_indices = np.arange(i * batch_size, i * batch_size + labels.size(0))[batch_errors] #np.arange 是 numpy 中的一个函数，用于生成一个连续的整数序列。
            #labels.size(0) 表示当前批次的样本数量（通常等于 batch_size，除非最后一个批次样本数量不足）。
            #[batch_errors]：这是 numpy 的布尔索引操作，会根据 batch_errors 中的布尔值筛选出 np.arange(i * batch_size, i * batch_size + labels.size(0)) 中对应位置为 True 的元素。也就是说，只会保留预测错误样本的索引。
            error_indices.extend(batch_indices) #将当前批次中预测错误样本的索引添加到 error_indices 列表中
            error_true_labels.extend(labels[batch_errors].cpu().numpy()) #将当前批次中预测错误样本的真实标签添加到 error_true_labels 列表中
            error_pred_labels.extend(predicted[batch_errors].cpu().numpy()) #将当前批次中预测错误样本的预测标签添加到 error_true_labels 列表中

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy) #将当前轮次的测试准确率添加到 test_accuracies 列表中，以便后续分析和可视化。

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")#.4 表示保留小数点后 4 位，f 表示将值格式化为浮点数
    #在每个训练轮次（epoch）结束后，输出当前轮次的编号、该轮次的平均训练损失以及模型在测试集上的准确率

# Plot training loss and test accuracy
plt.figure(figsize=(10, 4)) #plt.figure() 用于创建一个新的图形窗口。
plt.subplot(1, 2, 1) #表示将图形窗口划分为 1 行 2 列的网格，当前要绘制的是第 1 个子图（从左到右，从上到下编号）。
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Train Loss")
#marker='o' 为每个数据点添加圆形标记。  label="Test Accuracy" 设置该折线图的标签。
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Test Accuracy")

plt.show() #用于显示创建好的图形窗口，将绘制的子图展示出来

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, cmap='Blues', xticklabels=emnist_classes, yticklabels=emnist_classes) #sns.heatmap()用于绘制热力图。
#annot=False 表示不在热力图的每个单元格中显示具体的数值 cmap='Blues' 表示使用蓝色色调的颜色映射来绘制热力图。不同的颜色深浅表示不同的数值大小。
# xticklabels=emnist_classes 和 yticklabels=emnist_classes 分别设置 x 轴和 y 轴的刻度标签为 emnist_classes 列表中的元素。
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of EMNIST Classification")
plt.show()

# Display sample test images with predictions
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.ravel() #将二维的 axes 数组展平为一维数组，方便后续循环访问每个子图
for i in range(15):
    idx = np.random.randint(0, len(test_images))
    image = test_images[idx] #从测试集中随机选取一张图像
    true_label = test_labels[idx]

    # Convert image to PyTorch tensor and normalize
    image_tensor = transform(image).unsqueeze(0).to(device)
    #transform(image)：使用定义好的 transform 函数对图像进行预处理，例如将图像转换为张量、归一化等。
    #在张量的第 0 维添加一个维度，以符合模型输入的批量维度要求。
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        predicted_label = torch.argmax(pred, dim=1).cpu().item()
        #torch.argmax(pred, dim=1)：在预测输出的第 1 维（类别维度）上找到最大值的索引，即预测的类别标签。

    axes[i].imshow(image, cmap="gray") #在第 i 个子图中显示选取的图像，使用灰度颜色映射。
    axes[i].set_title(f"True: {true_label}, Pred: {predicted_label}")
    #在子图上方设置标题，显示图像的真实标签和预测标签。
    axes[i].axis("off") #关闭子图的坐标轴显示

plt.tight_layout() #自动调整子图的布局，使它们之间的间距合适，避免标签重叠
plt.show() #显示包含所有子图的图形。

# 统计错误样本
error_count = len(error_indices)
print(f"Total number of errors: {error_count}")

# Save model and error.csv:
today = datetime.datetime.now().strftime('%Y-%m-%d')
#获取当前的日期,并将其格式化为YYYY-MM-DD的字符串形式，存储在变量today中
# new folder
results_dir = f"results_{today}" #创建一个以 results_ 加上当前日期命名的文件夹名称
os.makedirs(results_dir, exist_ok=True) #使用 os.makedirs 函数创建指定名称的文件夹。exist_ok=True 表示如果该文件夹已经存在，不会抛出错误。

# Save the model
datasetname = "emnist-byclass"  # 实际的数据集
file_name_model = f"emnist_CNN_emnist_{datasetname}_{num_epochs}epochs_{today}.pth"
#生成一个包含数据集名称、训练轮数和日期的模型文件名。
file_path_model = os.path.join(results_dir, file_name_model)
#使用 os.path.join 函数将结果文件夹路径和模型文件名拼接成完整的文件路径。
torch.save(model.state_dict(), file_path_model)
#使用 torch.save 函数将模型的状态字典（即模型的参数）保存到指定的文件路径。
print("Model saved as emnist_cnn_model.pth")

# 将错误样本信息保存到 CSV 文件
file_name = f"error_samples_CNN_emnist_{datasetname}_{today}.csv"
#生成一个包含数据集名称和日期的错误样本 CSV 文件名。
file_path = os.path.join(results_dir, file_name)
#将结果文件夹路径和 CSV 文件名拼接成完整的文件路径。
with open(file_path, mode='w', newline='') as file: #with 语句是一个上下文管理器，它会自动处理文件的打开和关闭操作，确保在代码块执行完毕后，文件会被正确关闭，避免资源泄漏。
#mode='w' 表示以写入模式打开文件，如果文件不存在则创建它，如果文件已存在则会清空文件内容。
#newline='' 参数是为了确保在写入 CSV 文件时，换行符的处理符合标准。在不同操作系统中，换行符的表示可能不同（如 Windows 是 \r\n，Unix/Linux 是 \n），
# 使用 newline='' 可以避免 Python 在写入文件时对换行符进行额外处理。
    writer = csv.writer(file)
#csv.writer() 是 Python 内置的 csv 模块中的一个函数，用于创建一个 CSV 写入器对象。这个对象可以方便地将数据写入 CSV 文件。
#file 是前面 open() 函数返回的文件对象，将其传递给 csv.writer() 以指定要写入的文件。
    writer.writerow(['Index', 'True Label', 'Predicted Label']) #writerow() 方法用于向 CSV 文件中写入一行数据。这里传入的是一个包含三个字符串元素的列表
    for idx, true_label, pred_label in zip(error_indices, error_true_labels, error_pred_labels):
        writer.writerow([idx, emnist_classes[true_label], emnist_classes[pred_label]])

print("Error samples saved to {file_name}")