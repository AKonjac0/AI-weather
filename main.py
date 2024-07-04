import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取 CSV 文件数据
data_path = 'data.csv'  # 请确保 CSV 文件放在与此脚本相同的文件夹中
data = pd.read_csv(data_path)

# 转换所需列为浮点型，并处理 NaN 值
columns_to_convert = ['Ta_Avg', 'Ta_Max', 'Ta_Min', 'RH_Avg', 'RH_Max', 'RH_Min', 'rain_Tot']
for column in columns_to_convert:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# 填充 NaN 值（可以根据需要选择其他策略，如删除含 NaN 的行）
data = data.dropna()

# 选择所需的列
selected_columns = ['Ta_Avg', 'Ta_Max', 'Ta_Min', 'RH_Avg', 'RH_Max', 'RH_Min']
X = data[selected_columns].values

# 目标变量
Y = (data['rain_Tot'] > 0).astype(np.float32)  # 假设下雨量大于0即为下雨

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集与训练集
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量并添加批次维度
train_x = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
train_y = torch.tensor(train_y.tolist(), dtype=torch.float32)
test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)
test_y = torch.tensor(test_y.tolist(), dtype=torch.float32)

# 创建数据加载器
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_ds = TensorDataset(test_x, test_y)
valid_dl = DataLoader(valid_ds, batch_size=128)

# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 2, 1)  # 根据调整的输出尺寸
    
    def forward(self, x):
        print("Input shape:", x.shape)  # 添加调试信息
        x = F.relu(self.conv1(x))
        print("After conv1:", x.shape)  # 添加调试信息
        x = self.pool(x)
        print("After pool:", x.shape)  # 添加调试信息
        x = x.view(x.size(0), -1)
        print("After view:", x.shape)  # 添加调试信息
        x = torch.sigmoid(self.fc1(x))
        print("After fc1:", x.shape)  # 添加调试信息
        return x

# 创建模型
model = CNNModel()

# 损失函数和优化器
loss_fn = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
def train_model(model, train_dl, valid_dl, loss_fn, opt, epochs=20):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred.squeeze(), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_fn(model(xb).squeeze(), yb) for xb, yb in valid_dl) / len(valid_dl)
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {valid_loss.item()}")

# 训练模型
train_model(model, train_dl, valid_dl, loss_fn, opt)

# 保存模型参数
torch.save(model.state_dict(), "cnn_model.pth")

# 评估模型
model.eval()
with torch.no_grad():
    test_pred = model(test_x).squeeze()
    test_pred = (test_pred > 0.5).float()  # 阈值为0.5
    accuracy = accuracy_score(test_y, test_pred)
    precision = precision_score(test_y, test_pred)
    recall = recall_score(test_y, test_pred)
    f1 = f1_score(test_y, test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 对新的数据进行预测
def predict(new_data):
    new_data = scaler.transform(new_data)  # 标准化数据
    new_data = torch.tensor(new_data, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred = model(new_data).squeeze()
        pred = (pred > 0.5).float()  # 阈值为0.5
    return pred

# 示例新数据

new_data = np.array([[15.2, 20.5, 10.1, 90.0, 90.0, 90.0]])  # 示例数据
prediction = predict(new_data)
print(prediction.item())
print(f"Prediction: {'Rain' if prediction.item() == 1.0 else 'No Rain'}")
