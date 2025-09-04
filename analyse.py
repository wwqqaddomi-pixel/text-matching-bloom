import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import datetime
import os
import csv

# Check Apple Metal MPS backend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# EMNIST character labels full label:(62 classes)
emnist_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n',
    'o', 'p', 'q', 'r','s', 't', 'u', 'v', 'w', 'x',
    'y', 'z'
]

# Define dataset class for PyTorch
class EMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load test dataset
batch_size = 64
emnist_data_path = "D:\\实习\\EMNIST"
test_df = pd.read_csv(emnist_data_path + '\emnist-byclass-test.csv', header=None)
test_labels = test_df.iloc[:, 0].values
test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
test_dataset = EMNISTDataset(test_images, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load the saved model
today = datetime.datetime.now().strftime('%Y-%m-%d')
results_dir = f"results_{today}"
num_epochs = 10
datasetname = "emnist-byclass"
file_name_model = f"emnist_CNN_emnist_{datasetname}_{num_epochs}epochs_{today}.pth"
file_path_model = os.path.join(results_dir, file_name_model)

model = CNN().to(device)
model.load_state_dict(torch.load(file_path_model, weights_only=True))
model.eval()

# Evaluate model and count errors
correct = 0
total = 0
error_indices = []
error_true_labels = []
error_pred_labels = []
correct_indices = []
correct_true_labels = []
correct_pred_labels = []

# 用于存储每种原始 - 替换对的错误数量和总数量
error_type_count_dict = {}
total_type_count_dict = {}

# 用于存储每个字符的总识别次数
total_char_count_dict = {char: 0 for char in emnist_classes}

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        correct_mask = (predicted == labels).cpu().numpy()
        correct_batch_indices = np.arange(i * batch_size, i * batch_size + labels.size(0))[correct_mask]
        correct_indices.extend(correct_batch_indices)
        correct_true_labels.extend(labels[correct_mask].cpu().numpy())
        correct_pred_labels.extend(predicted[correct_mask].cpu().numpy())

        error_mask = (predicted != labels).cpu().numpy()
        error_batch_indices = np.arange(i * batch_size, i * batch_size + labels.size(0))[error_mask]
        error_indices.extend(error_batch_indices)
        error_true_labels.extend(labels[error_mask].cpu().numpy())
        error_pred_labels.extend(predicted[error_mask].cpu().numpy())

        for true_lbl, pred_lbl in zip(labels[error_mask].cpu().numpy(), predicted[error_mask].cpu().numpy()):
            true_char = emnist_classes[true_lbl]
            pred_char = emnist_classes[pred_lbl]
            pair_key = f"{true_char}-{pred_char}"
            total_type_count_dict[pair_key] = total_type_count_dict.get(pair_key, 0) + 1
            error_type_count_dict[pair_key] = error_type_count_dict.get(pair_key, 0) + 1

        for lbl in labels.cpu().numpy():
            char = emnist_classes[lbl]
            total_char_count_dict[char] += 1

error_count = len(error_indices)
correct_count = len(correct_indices)
print(f"Total number of errors: {error_count}")
print(f"Total number of correct predictions: {correct_count}")

# 确保保存目录存在
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save samples to CSV
file_name = f"classification_results_CNN_emnist_{datasetname}_{today}.csv"
file_path = os.path.join(results_dir, file_name)
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'True Label', 'Predicted Label', 'Is Correct'])
    for idx, true_label, pred_label in zip(correct_indices, correct_true_labels, correct_pred_labels):
        writer.writerow([idx, emnist_classes[true_label], emnist_classes[pred_label], True])
    for idx, true_label, pred_label in zip(error_indices, error_true_labels, error_pred_labels):
        writer.writerow([idx, emnist_classes[true_label], emnist_classes[pred_label], False])

print(f"Classification results saved to {file_name}")

# 读取保存的分类结果 CSV 文件
df = pd.read_csv(file_path)

# 筛选出错误分类的样本
error_df = df[df['Is Correct'] == False].copy()

# 定义错误种类的分析函数
def error_type(true_label, pred_label):
    # 大小写错误
    if true_label.isalpha() and pred_label.isalpha() and true_label.upper() == pred_label.upper():
        return '大小写错误'
    # 字符形状相似错误（这里简单定义一些相似的字符对，可根据实际情况扩展）
    similar_pairs = [('b', 'h'), ('G', '6'), ('g', '9'), ('9', 'q'), ('O', '0'), ('o', '0'), ('L', '1'), ('i', 'l'), ('i', '1'), ('l', '1'), ('l', 'I'), ('I', '1'), ('Z', '2'), ('2', 'z'), ('U', 'V'), ('5', 'S'), ('5','s'), ('8', 'B'), ('6', 'b'), ('M', 'N'), ('Q', 'O'), ('Q', 'o'), ('Q', '0'), ('W', 'V'), ('W', 'v')]
    if (true_label, pred_label) in similar_pairs or (pred_label, true_label) in similar_pairs:
        return '形状相似错误'
    # 其他不相似的纯错误
    return '纯错误'

# 应用错误类型分析函数到错误数据框
error_df['Error Type'] = error_df.apply(lambda row: error_type(row['True Label'], row['Predicted Label']), axis=1)

# 计算每种错误类型的数量
error_type_counts = error_df['Error Type'].value_counts().reset_index(name='错误数量')

# 计算每种错误类型的占比
total_errors = len(error_df)
error_type_counts['错误占比'] = (error_type_counts['错误数量'] / total_errors).apply(lambda x: '{:.2%}'.format(x))

# 输出结果
print(error_type_counts)

# 筛选出错误类型为纯错误的数据
pure_error_df = error_df[error_df['Error Type'] == '纯错误']

# 保存纯错误数据到新的 CSV 文件
pure_error_file_name = f"pure_error_results_CNN_emnist_{datasetname}_{today}.csv"
pure_error_file_path = os.path.join(results_dir, pure_error_file_name)
pure_error_df.to_csv(pure_error_file_path, index=False)

print(f"Pure error classification results saved to {pure_error_file_name}")

# 统计每一个类的三种错误的个数和错误率
class_error_stats = []
for cls in emnist_classes:
    cls_error_df = error_df[error_df['True Label'] == cls]
    cls_total = len(df[df['True Label'] == cls])
    case_error_count = len(cls_error_df[cls_error_df['Error Type'] == '大小写错误'])
    shape_error_count = len(cls_error_df[cls_error_df['Error Type'] == '形状相似错误'])
    pure_error_count = len(cls_error_df[cls_error_df['Error Type'] == '纯错误'])
    total_cls_error = case_error_count + shape_error_count + pure_error_count
    error_rate = total_cls_error / cls_total if cls_total > 0 else 0
    class_error_stats.append({
        'Class': cls,
        '大小写错误数量': case_error_count,
        '形状相似错误数量': shape_error_count,
        '纯错误数量': pure_error_count,
        '总错误数量': total_cls_error,
        '错误率': '{:.2%}'.format(error_rate)
    })

# 将统计结果保存到 CSV 文件
class_error_stats_df = pd.DataFrame(class_error_stats)
class_error_stats_file_name = f"class_error_stats_CNN_emnist_{datasetname}_{today}.csv"
class_error_stats_file_path = os.path.join(results_dir, class_error_stats_file_name)
class_error_stats_df.to_csv(class_error_stats_file_path, index=False)

print(f"Class error statistics saved to {class_error_stats_file_name}")

# 计算每种原始 - 替换对的错误率并保存为 xls 文件
error_rate_data = []
for pair_key, error_count in error_type_count_dict.items():
    total_count = total_type_count_dict[pair_key]
    original, replacement = pair_key.split('-')
    original_total_count = total_char_count_dict[original]
    error_rate = error_count / original_total_count if original_total_count > 0 else 0
    error_rate_data.append({
        'original': original,
       'replacement': replacement,
        'error_rate': error_rate
    })
error_rate_df = pd.DataFrame(error_rate_data)
error_rate_file_name = f"error_rate_results_CNN_emnist_{datasetname}_{today}.csv"
error_rate_file_path = os.path.join(results_dir, error_rate_file_name)
error_rate_df.to_csv(error_rate_file_path, index=False)

print(f"Error rate results saved to {error_rate_file_name}")
