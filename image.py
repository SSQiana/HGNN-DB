import matplotlib.pyplot as plt
import numpy as np


# 定义函数计算系数
def calculate_coefficient(beta, k):
    coefficients = []
    denominator = sum([np.log(beta + i) for i in range(1, k + 1)])

    for i in range(1, k + 1):
        coefficient = np.log(beta + i) / denominator
        coefficients.append(coefficient)

    return coefficients


# 设置参数
beta_value = 0
k_value = 5

# 计算系数
coefficients = calculate_coefficient(beta_value, k_value)

# 绘制折线图
plt.plot(range(1, k_value + 1), coefficients, marker='o')
plt.xlabel('i')
plt.ylabel('Coefficient')
plt.title(f'Coefficients for Z^i with beta={beta_value}')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 定义对数函数
# def log_function(x, base):
#     return np.log(x) / np.log(base)
# # 生成x值,修改范围为0-5
# x_values = np.arange(2, 7, 1)
#
# base_values = 5
# y = []
#
# for i in range(5):
#     y_values = log_function(x_values[i], base_values)
#     y.append(y_values)
# print(y)
#
# y_1 = np.max(y) - np.min(y)
# print(y_1)
# for j in range(len(y)):
#     y[j] = y[j] / y_1
# print(y)
#
# y_sum = np.sum(y)
# new_y = []
# for j in range(len(y)):
#     new_y.append(y[j] / y_sum)
# plt.plot(x_values, new_y)
# print(new_y)
#
# # 设置图形属性
# plt.title('Logarithmic Functions with Varying Bases')
# plt.xlabel('x')
# plt.ylabel('log(x)')
# plt.legend()
# plt.grid(True)
# plt.ylim(0, 0.5)  # 修改纵坐标范围为0-1
#
# plt.show()
