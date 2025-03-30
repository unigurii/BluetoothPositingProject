import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import threading
import queue
import time
from queue import LifoQueue


port = 'COM18'  # 串口号，根据实际情况修改
baud_rate = 115200  # 波特率，根据实际情况修改
timeout = 1  # 超时时间
tag_num = 2 #tag数量
tag_id1 = '5464DEA3721B'
tag_id2 = '5464DE2C20E4'

# 打开串口
ser = serial.Serial(port, baud_rate, timeout=timeout)
if not ser.is_open:
    print("串口打开失败！")
    exit()

# 创建文件用于存储数据
file_path = "data.txt"
file = open(file_path, "a")

# 初始化三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
ax.set_zlim([0, 100])

# 绘制坐标轴
# X轴
ax.quiver(0, 0, 0, 100, 0, 0, color='r', arrow_length_ratio=0.1)  # 正X轴，带箭头
ax.quiver(0, 0, 0, -100, 0, 0, color='r', arrow_length_ratio=0)   # 负X轴，无箭头
ax.text(110, 0, 0, 'X', color='r')  # 标注X轴

# Y轴
ax.quiver(0, 0, 0, 0, 100, 0, color='g', arrow_length_ratio=0.1)  # 正Y轴，带箭头
ax.quiver(0, 0, 0, 0, -100, 0, color='g', arrow_length_ratio=0)   # 负Y轴，无箭头
ax.text(0, 110, 0, 'Y', color='g')  # 标注Y轴

# Z轴
ax.quiver(0, 0, 0, 0, 0, 100, color='b', arrow_length_ratio=0.1)  # 正Z轴，带箭头
ax.quiver(0, 0, 0, 0, 0, -100, color='b', arrow_length_ratio=0)   # 负Z轴，无箭头
ax.text(0, 0, 110, 'Z', color='b')  # 标注Z轴

# 初始化数据点
x, y, z = 0, 0, 0
scatter = ax.scatter(x, y, z, color='r')
line, = ax.plot([0, x], [0, y], [0, z], color='k')  # 连接原点和目标点的线

x1, y1, z1 = 0, 0, 0
scatter1 = ax.scatter(x1, y1, z1, color='b')
line1, = ax.plot([0, x1], [0, y1], [0, z1], color='k')  # 连接原点和目标点的线

# 初始化文本显示
text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
text1 = ax.text2D(0.05, 0.50, "", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# 数据队列
data_queue = queue.LifoQueue()

# 数据处理函数
def process_data(data):
    try:
        # 解析数据
        parts = data.split(',')
        #print(len(parts))
        if len(parts) < 9:
            return None
        tag_id = parts[0].split(':')[1]  # 提取Tag ID
        rssi = float(parts[1])  # RSSI
        azimuth = float(parts[2])  # 方位角
        elevation = float(parts[3])  # 仰角
        timestamp = parts[-1].strip('"')  # 时间戳

        # 计算三维坐标
        distance = 100 * (1 - rssi / -90)  # 将RSSI映射到距离
        distance = 100
        #print('the distance is :',distance)

        z = abs(distance * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)))
        y = -distance * abs(np.cos(np.radians(elevation))) * np.sin(np.radians(azimuth))
        x = -distance * np.sin(np.radians(elevation))

        return x, y, z, rssi, azimuth, elevation, tag_id, timestamp
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


# 串口读取线程
def read_serial():
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            print("接收到的原始数据：", data)
            #file.write("原始数据：" + data + "\n")
            #file.flush()

            result = process_data(data)
            if result:
                data_queue.put(result)

# 更新图像的函数
def update(frame):
    global x, y, z
    if not data_queue.empty():
        x, y, z, rssi, azimuth, elevation, tag_id, timestamp = data_queue.get()
        if tag_id == tag_id1:
            # 更新散点图的坐标
            scatter._offsets3d = ([x], [y], [z])
            # 更新连接线
            line.set_data([0, x], [0, y])
            line.set_3d_properties([0, z])
            # 更新文本显示
            text.set_text(f"RSSI: {rssi}\nAzimuth: {azimuth}°\nElevation: {elevation}°\nTag ID: {tag_id}\nTimestamp: {timestamp}")

        if tag_id == tag_id2:
            # 更新散点图的坐标
            scatter1._offsets3d = ([x], [y], [z])
            # 更新连接线
            line1.set_data([0, x], [0, y])
            line1.set_3d_properties([0, z])
            # 更新文本显示
            text1.set_text(f"RSSI: {rssi}\nAzimuth: {azimuth}°\nElevation: {elevation}°\nTag ID: {tag_id}\nTimestamp: {timestamp}")
        #file.write("处理结果：RSSI: {}, Azimuth: {}°, Elevation: {}°, Tag ID: {}, Timestamp: {}\n".format(rssi, azimuth, elevation, tag_id, timestamp))
        #file.flush()
    return [scatter,scatter1], [line,line1], [text,text1]

# 启动串口读取线程
threading.Thread(target=read_serial, daemon=True).start()

# 创建动画
ani = animation.FuncAnimation(fig, update, interval=0, blit=False)

# 显示图像
plt.show()

# 关闭串口和文件
ser.close()
file.close()
print("串口已关闭，数据已保存到文件")


