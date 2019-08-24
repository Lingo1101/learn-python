import numpy as np      #导入numpy库-数值计算扩展库（可用来表示矩阵）
from matplotlib import pyplot as plt    #这两句输出‘矩阵图’
#从matplotlib-2D绘图库中导入pylot!!
#导库：cmd→输入pip3 install ***

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata

x = np.arange(-200, -100, 1)        #相当于熟悉的.range()作用；是序列，可当向量使用；可迭代
y = np.arange(-5, 5, 0.1)       #起点对应楼下绘图起始坐标；可步长为小数
Z = np.zeros((len(x), len(y)))  #zeros(形状, dtype=float, order='C')
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# ydata = b + w * xdata
b = -120    #initial b
w = -4      #initial w
lr = 1   #Learning rate
iteration = 100000      #train set

#Store initial values for plotting
b_history = [b]
w_history = [w]

lr_b=0
lr_w=0

#Iterations       迭代！！！
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    lr_b=lr_b+b_grad**2
    lr_w=lr_w+w_grad**2

    #Update parameters
    b = b - lr/np.sqrt(lr_b) * b_grad
    w = w - lr/np.sqrt(lr_w) * w_grad

    #store parameter for platting
    b_history.append(b)
    w_history.append(w)

#plot the figure
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()







