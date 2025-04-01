import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# 随机种子设置
torch.manual_seed(42)
np.random.seed(42)

# 一、数据生成模块
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=10000, num_bases=4, z_base=5.0, area_size=100.0):
        """
        参数：
        num_samples: 总样本数
        num_bases: 基站数量(4-6)
        z_base: 基站固定高度(米)
        area_size: 目标点生成区域范围(米)
        """
        self.num_samples = num_samples
        self.num_bases = num_bases
        self.z_base = z_base
        self.area_size = area_size
        
        # 生成固定基站位置 (均匀分布在区域内)
        self.base_positions = torch.stack([
            torch.FloatTensor([area_size * (i%2), area_size * (i//2)]) 
            for i in range(num_bases)
        ], dim=0)
        
        # 计算基站坐标中心
        self.base_center = self.base_positions.mean(dim=0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成目标点坐标
        target = torch.FloatTensor(2).uniform_(0, self.area_size)
        z_base_tensor = torch.tensor(self.z_base, dtype=torch.float32)
        
        features = []
        rssi_values = []
        for base_pos in self.base_positions:
            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            distance = torch.sqrt(dx**2 + dy**2)
            
            # 方位角计算
            azimuth = torch.atan2(dy, dx)
            sin_theta = torch.sin(azimuth)
            cos_theta = torch.cos(azimuth)
            
            # 仰角计算修正
            elev_angle = torch.atan2(z_base_tensor, distance)
            d_horizontal = z_base_tensor * torch.tan(elev_angle)
            
            # RSSI计算修正（关键修改点）
            rssi = -50 - 30 * torch.log10(distance + 1e-6)
            rssi += torch.randn_like(rssi) * 2  # 保持形状一致
            
            # 特征构建
            delta_x = base_pos[0] - self.base_center[0]
            delta_y = base_pos[1] - self.base_center[1]
            features.append(torch.tensor([
                delta_x.item(), delta_y.item(),
                sin_theta.item(),
                cos_theta.item(),
                d_horizontal.item(),
                (rssi.item() + 100)/50  # 转换为标量
            ]))
            rssi_values.append(rssi.item())  # 转换为Python标量
        
        return (
            torch.stack(features),  # shape: (num_bases, 6)
            torch.tensor(rssi_values, dtype=torch.float32),  # 显式指定类型
            target
        )

# 二、模型定义
class LightweightLocator(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        
        # 共享基站编码器
        self.base_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2)
        )
        
        # RSSI动态权重生成器
        self.rssi_weight = nn.Sequential(
            nn.Linear(hidden_dim//2 + 1, 1),  # 输入特征+RSSI
            nn.Sigmoid()
        )
        
        # 坐标回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim//2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, rssi):
        """
        输入：
        x: 基站特征 (batch_size, num_bases, 6)
        rssi: RSSI值 (batch_size, num_bases)
        """
        batch_size, num_bases, _ = x.shape
        
        # 编码所有基站特征
        encoded = self.base_encoder(x.view(-1, 6))  # (batch*num_bases, hidden//2)
        encoded = encoded.view(batch_size, num_bases, -1)
        
        # 生成注意力权重
        weight_input = torch.cat([
            encoded,
            rssi.unsqueeze(-1)  # 添加原始RSSI值
        ], dim=-1)
        weights = self.rssi_weight(weight_input)  # (batch, num_bases, 1)
        
        # 加权聚合特征
        weighted_feats = (encoded * weights).sum(dim=1)  # (batch, hidden//2)
        
        return self.regressor(weighted_feats)

# 三、训练辅助模块
class CurriculumTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 混合损失函数
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
        
    def train_epoch(self, epoch, noise_level=0.0):
        self.model.train()
        total_loss = 0.0
        
        for features, rssi, targets in self.train_loader:
            # 课程学习：逐步增加噪声
            if noise_level > 0:
                # 对角度添加噪声
                features[:, :, 2:4] += noise_level * torch.randn_like(features[:, :, 2:4]) * 0.1  # sin/cos
                features[:, :, 4] += noise_level * torch.randn_like(features[:, :, 4]) * 2.0  # d_horizontal
                rssi += noise_level * torch.randn_like(rssi) * 3.0  # RSSI
            
            self.optimizer.zero_grad()
            
            # 前向计算
            preds = self.model(features, rssi)
            
            # 混合损失
            loss = 0.8*self.mse_loss(preds, targets) + 0.2*self.huber_loss(preds, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_rmse = 0.0
        
        with torch.no_grad():
            for features, rssi, targets in self.val_loader:
                preds = self.model(features, rssi)
                rmse = torch.sqrt(self.mse_loss(preds, targets))
                total_rmse += rmse.item()
        
        return total_rmse / len(self.val_loader)

# 四、训练流程示例
if __name__ == "__main__":
    # 参数设置
    NUM_BASES = 4
    BATCH_SIZE = 32
    EPOCHS = 100
    
    # 数据准备
    train_set = SyntheticDataset(num_samples=10000, num_bases=NUM_BASES)
    val_set = SyntheticDataset(num_samples=2000, num_bases=NUM_BASES)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    # 模型初始化
    model = LightweightLocator()
    trainer = CurriculumTrainer(model, train_loader, val_loader)
    
    # 课程学习噪声计划
    noise_schedule = np.linspace(0, 1, EPOCHS)
    
    # 训练循环
    best_rmse = float('inf')
    for epoch in range(EPOCHS):
        train_loss = trainer.train_epoch(epoch, noise_level=noise_schedule[epoch])
        val_rmse = trainer.validate()
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.2f}m")
        
        # 保存最佳模型
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), "best_model.pth")
    
    # 可视化测试
    test_sample = val_set[0]
    with torch.no_grad():
        pred = model(test_sample[0].unsqueeze(0), test_sample[1].unsqueeze(0))
    
    plt.figure(figsize=(8,6))
    # 绘制基站位置
    for base in train_set.base_positions:
        plt.scatter(*base, c='red', marker='^', s=100, label="Base Station")
    # 绘制真实目标
    plt.scatter(*test_sample[2], c='green', s=200, label="True Target")
    # 绘制预测目标
    plt.scatter(*pred.squeeze(), c='blue', s=200, marker='x', label="Predicted")
    plt.legend()
    plt.title(f"Localization Result (Error: {torch.norm(pred - test_sample[2]).item():.2f}m)")
    plt.show()