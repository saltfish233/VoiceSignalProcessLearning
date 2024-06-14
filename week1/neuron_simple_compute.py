import numpy as np

# 输入向量
x = np.array([0.5, -0.2, 0.1])

# 权重向量
w = np.array([0.3,0.4,0.3])

# 偏置
b = 0.1

# 加权和
z = np.dot(x,w) + b

# 激活函数
def ReLU(z):
    return max(0,z)

# 神经元输出
output = ReLU(z)

print(output)