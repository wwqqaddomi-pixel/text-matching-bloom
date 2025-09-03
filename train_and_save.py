import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
import pandas as pd


# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义类别标签
Chars74K_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

# 自定义数据集类
class Chars74kDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 划分训练集和测试集
def split_dataset(root_dir):
    train_data = []
    test_data = []
    for i in range(1, 63):
        folder_name = f"Sample{str(i).zfill(3)}"
        folder_path = os.path.join(root_dir, folder_name)
        img_files = os.listdir(folder_path)
        np.random.shuffle(img_files)
        split_index = len(img_files) // 2
        for img_name in img_files[:split_index]:
            img_path = os.path.join(folder_path, img_name)
            train_data.append((img_path, i - 1))
        for img_name in img_files[split_index:]:
            img_path = os.path.join(folder_path, img_name)
            test_data.append((img_path, i - 1))
    return train_data, test_data

# 加载数据集
root_dir = 'D:\实习\Chars74K\English\Fnt'  # 替换为你的 Chars74k 数据集根目录
train_data, test_data = split_dataset(root_dir)

train_dataset = Chars74kDataset(train_data, transform=transform)
test_dataset = Chars74kDataset(test_data, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # 128x128 经过三次池化后变为 16x16
        self.fc2 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 1
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_true = []
    error_indices = []
    error_true_labels = []
    error_pred_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

            batch_errors = (predicted != labels).cpu().numpy()
            batch_indices = np.arange(i * batch_size, i * batch_size + labels.size(0))[batch_errors]
            error_indices.extend(batch_indices)
            error_true_labels.extend(labels[batch_errors].cpu().numpy())
            error_pred_labels.extend(predicted[batch_errors].cpu().numpy())

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 绘制训练损失和测试准确率
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label="Train Loss")
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

plt.show()

# 生成并绘制混淆矩阵
conf_matrix = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, cmap='Blues', xticklabels=Chars74K_classes, yticklabels=Chars74K_classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of Chars74k Classification")
plt.show()

# 显示样本测试图像及其预测结果
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.ravel()
for i in range(15):
    idx = np.random.randint(0, len(test_dataset))
    image, true_label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(image)
        predicted_label = torch.argmax(pred, dim=1).cpu().item()

    image = image.cpu().squeeze().numpy()
    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(f"True: {Chars74K_classes[true_label]}, Pred: {Chars74K_classes[predicted_label]}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# 统计错误样本
error_count = len(error_indices)
print(f"Total number of errors: {error_count}")

# 保存模型
today = str(pd.Timestamp.now().date())
results_dir = f"results_{today}"
os.makedirs(results_dir, exist_ok=True)

datasetname = "Chars74k"
file_name_model = f"Chars74k_CNN_{datasetname}_{num_epochs}epochs_{today}.pth"
file_path_model = os.path.join(results_dir, file_name_model)
torch.save(model.state_dict(), file_path_model)
print("Model saved as Chars74k_cnn_model.pth")