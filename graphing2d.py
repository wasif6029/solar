import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# data = [[561, 338], [568, 333], [510, 329], [582, 324], [590, 318], [596, 311], [
#     602, 305], [617, 291], [662, 225], [669, 214], [674, 203], [680, 191], [692, 169]]


# x = [301, 308, 330, 338, 345, 352, 366, 374, 382, 408, 416, 424, 431, 439, 446, 453, 460, 467, 410, 482,
#      489, 503, 511, 537, 545, 552, 559, 567, 574, 581, 588, 595, 602, 608, 667, 672, 678, 684, 691, 691]

# y = [313, 302, 265, 255, 245, 234, 216, 206, 197, 167, 161, 156, 151, 146, 142, 138, 136, 132, 130, 128,
#      127, 127, 127, 132, 135, 138, 141, 146, 150, 155, 160, 167, 174, 180, 263, 210, 287, 297, 310, 310]

# y_new = []

# for i in y:
#     y_new.append(480 - i)

# y = y_new

x = y = []

file1 = open('centers.txt', 'r')
count = 0
while True:
    count += 1

    # Get next line from file
    line = file1.readline()
    # print(line)
    if not line:
        break
    cord = re.findall(r"[-+]?\d*\.\d+|\d+", line)

    x.append(float(cord[0]))
    y.append(float(cord[1]))

#
print()
print()
print()
print()
print()
print()
print()
# print(y)

# z = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10, 10,
#      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

x_data = x
y_data = y

# for i in data:
#     x_data.append(i[0])
#     y_data.append(i[1])

x_data = np.array(x_data)
y_data = np.array(y_data)

# print(x_data)
# print(y_data)

plt.scatter(x_data, y_data)
plt.show()


def model_f(x, a, b, c):
    return a * x**2 + b * x + c


popt, pcov = curve_fit(model_f, x_data, y_data, p0=[3, 2, -16])

a_opt, b_opt, c_opt = popt
x_model = np.linspace(min(x_data), max(x_data), 100)
y_model = model_f(x_model, a_opt, b_opt, c_opt)

plt.scatter(x_data, y_data)
plt.plot(x_model, y_model, color='g')
plt.show()

a_opt, b_opt, c_opt = popt
x_model = np.linspace(min(x_data), max(x_data) + 50, 100)
y_model = model_f(x_model, a_opt, b_opt, c_opt)

plt.scatter(x_data, y_data)
plt.plot(x_model, y_model, color='r')
plt.show()
