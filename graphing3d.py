# import stuff2
import sys
# import stuff1
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
import re

from matplotlib.animation import PillowWriter


def make3d(filePath):

    x = []
    y = []
    z = []

    file1 = open(filePath, 'r')
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
    # t = np.random.choice(np.linspace(-1000000, 10000000, 10000002), n)

    t = []
    for i in range(numbers):
        t.append(i)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter(x, y, z, label="raw data")
    # for i in range(1):
    #     ax.scatter(x[0], y[0], z[0])
    # ax.scatter(x[n-1], y[n-1], z[n-1])

    def func(t, x2, x1, x0, y2, y1, y0, z2, z1, z0):
        Px = Polynomial([x2, x1, x0])
        Py = Polynomial([y2, y1, y0])
        Pz = Polynomial([z2, z1, z0])

        return np.concatenate([Px(t), Py(t), Pz(t)])

    start_vals = [x[0], x[1], x[2],
                  y[0], y[1], y[2],
                  z[0], z[1], z[2]]

    xyz = np.concatenate([x, y, z])
    popt, _ = curve_fit(func, t, xyz, p0=start_vals)

    t_fit = np.linspace(min(t), max(t) + (abs(max(t) - min(t)) // 10))
    xyz_fit = func(t_fit, *popt).reshape(3, -1)
    ax.plot(xyz_fit[0, :], xyz_fit[1, :], xyz_fit[2, :], color="green", label="fitted data")

    metadata = dict(title="Movie")
    writer = PillowWriter(fps=15, metadata=metadata)

    ax.legend()
    # ax.view_init(270, 270)
    plt.show()


make3d("E:/Python/yolov5/yolov5/runs/detect/exp302/right_centers.txt")
