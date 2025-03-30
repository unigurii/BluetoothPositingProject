import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 创建画布和坐标轴
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 定义点集，每个点包含坐标、颜色和大小
points = [
    (1, 1, 'red', 50),
    (2, 3, 'blue', 100),
    (4, 5, 'green', 150),
    (6, 7, 'purple', 200),
    (8, 9, 'orange', 250)
]

# 创建一个点，初始位置为点集的第一个点
dot, = ax.plot(points[0][0], points[0][1], 'ro', markersize=points[0][3])

# 定义更新函数
def update(frame):
    # 更新点的位置、颜色和大小
    dot.set_data(frame[0], frame[1])
    dot.set_color(frame[2])
    dot.set_markersize(frame[3])
    return dot,

# 创建动画
ani = FuncAnimation(fig, update, frames=points, interval=500)

# 显示动画
plt.show()
