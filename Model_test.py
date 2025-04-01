import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from DL_test import LightweightLocator

class ModelTester:
    def __init__(self, model_path, base_positions, z_base=5.0):
        """
        初始化测试器
        :param model_path: 训练好的模型路径
        :param base_positions: 基站坐标列表 [(x1,y1), (x2,y2),...]
        :param z_base: 基站固定高度
        """
        # 加载模型
        self.model = LightweightLocator()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 设置基站参数
        self.base_positions = torch.tensor(base_positions)
        self.base_center = self.base_positions.float().mean(dim=0)
        self.z_base = z_base
        
        # 可视化设置
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
    def generate_measurements(self, target):
        """
        根据目标坐标生成测量数据
        :param target: 目标坐标 (x, y)
        :return: (features, rssi_values)
        """
        target_tensor = torch.tensor(target)
        features = []
        rssi_values = []
        
        for base in self.base_positions:
            dx = target_tensor[0] - base[0]
            dy = target_tensor[1] - base[1]
            distance = torch.sqrt(dx**2 + dy**2)
            
            # 计算方位角
            azimuth = torch.atan2(dy, dx)
            sin_theta = torch.sin(azimuth)
            cos_theta = torch.cos(azimuth)
            
            # 计算水平投影距离
            elev_angle = torch.atan2(torch.tensor(self.z_base), distance)
            d_horizontal = self.z_base * torch.tan(elev_angle)
            
            # 生成RSSI（含噪声）
            rssi = -50 - 30 * torch.log10(distance + 1e-6)
            rssi += torch.randn(1).item() * 2
            
            # 构建特征
            delta_x = base[0] - self.base_center[0]
            delta_y = base[1] - self.base_center[1]
            features.append([
                delta_x.item(), delta_y.item(),
                sin_theta.item(), cos_theta.item(),
                d_horizontal.item(),
                (rssi + 100) / 50  # 归一化
            ])
            rssi_values.append(rssi)
            
        return (
            torch.tensor([features]),  # 添加batch维度
            torch.tensor([rssi_values])
        )
    
    def test_point(self, target):
        """
        测试单个目标点
        :param target: 目标坐标 (x, y)
        """
        # 生成数据
        features, rssi = self.generate_measurements(target)
        
        # 推理
        with torch.no_grad():
            pred = self.model(features, rssi)[0].numpy()
        
        # 计算误差
        error = np.linalg.norm(pred - target)
        
        # 可视化
        self.plot_result(target, pred, error)
        
    def plot_result(self, true_pos, pred_pos, error):
        """
        可视化定位结果
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制基站
        for i, (x, y) in enumerate(self.base_positions.numpy()):
            plt.scatter(x, y, c=self.colors[i], marker='^', s=200, 
                       label=f'Base Station {i+1}')
            
        # 绘制真实位置
        plt.scatter(*true_pos, c='green', s=300, label='True Position')
        
        # 绘制预测位置
        plt.scatter(*pred_pos, c='blue', s=300, marker='x', linewidth=3, 
                   label=f'Predicted (Error: {error:.2f}m)')
        
        # 绘制误差圆
        error_circle = Circle(true_pos, error, fill=False, 
                             linestyle='--', edgecolor='red')
        plt.gca().add_patch(error_circle)
        
        # 设置图形属性
        plt.title("Localization Result Visualization")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.show()

# 使用示例 ##################################################
if __name__ == "__main__":
    # 基站配置（需与训练时一致）
    BASE_POSITIONS = [
        [0, 0], 
        [100, 0], 
        [0, 100], 
        [100, 100]
    ]
    
    # 初始化测试器
    tester = ModelTester(
        model_path="best_model.pth",
        base_positions=BASE_POSITIONS,
        z_base=5.0
    )
    
    # 交互式测试
    while True:
        try:
            # 输入目标坐标
            x = float(input("请输入目标点X坐标（0-100）："))
            y = float(input("请输入目标点Y坐标（0-100）："))
            
            # 执行测试
            tester.test_point((x, y))
            
        except ValueError:
            print("输入错误，请确保输入的是数字！")
        except KeyboardInterrupt:
            print("\n测试结束")
            break