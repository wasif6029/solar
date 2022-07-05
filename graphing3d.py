# import stuff2
import sys
# import stuff1
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
import re


print("In graphing 3d")

x = []
y = []
z = []

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
    z.append(float(cord[2]))

numbers = len(x)
np.random.seed(123)
n = numbers
t = np.random.choice(np.linspace(-3000, 2000, 1000) / 500, n)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, label="raw data")


def func(t, x2, x1, x0, y2, y1, y0, z2, z1, z0):
    Px = Polynomial([x2, x1, x0])
    Py = Polynomial([y2, y1, y0])
    Pz = Polynomial([z2, z1, z0])
#     print(x2, x1, x0)
    return np.concatenate([Px(t), Py(t), Pz(t)])


start_vals = [x[0], x[1], x[2],
              y[0], y[1], y[2],
              z[0], z[1], z[2]]

xyz = np.concatenate([x, y, z])
popt, _ = curve_fit(func, t, xyz, p0=start_vals)

t_fit = np.linspace(min(t), max(t), 100)
xyz_fit = func(t_fit, *popt).reshape(3, -1)
ax.plot(xyz_fit[0, :], xyz_fit[1, :], xyz_fit[2, :], color="green", label="fitted data")

ax.legend()
plt.show()


# if __name__ == '__main__':
#     sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
