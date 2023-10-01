import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad


def f(x_):
    return 4 * (math.pi ** 2) * np.sin(2 * math.pi * x_)


def v(k, x_, x):
    if x[k - 1] <= x_ <= x[k]:
        res = (x_ - x[k - 1]) / (x[k] - x[k - 1])
    elif x[k] < x_ <= x[k + 1]:
        res = (x[k + 1] - x_) / (x[k + 1] - x[k])
    else:
        res = 0
    return res


def der_v(k, x_, x):
    if x[k - 1] <= x_ <= x[k]:
        res = 1.0 / (x[k] - x[k - 1])
    elif x[k] < x_ <= x[k + 1]:
        res = -1.0 / (x[k + 1] - x[k])
    else:
        res = 0
    return res


def phi(k, j, x):
    # print(f"{k=}  {j=}")
    def vk_vj(x_):
        # print(f"{k=} {j=} {x_=} {der_v(k, x_) * der_v(j, x_)=}")
        return der_v(k, x_, x) * der_v(j, x_, x)

    # print(f"{k=} {j=} {quad(vk_vj, 0, 1)[0]=}")
    return quad(vk_vj, 0, 1, points=np.linspace(x[j - 1], x[j + 1], 10).tolist())[0]


def solver(n, x):
    A = np.zeros((n, n))
    b = np.zeros(n)
    for j in range(1, n - 1):  # j = [1, n-1]
        for k in range(1, n - 1):  # k = [1, n-1]
            A[j][k] = phi(k, j, x)
        b[j] = quad(lambda _x: f(_x) * v(j, _x, x), 0, 1, points=np.linspace(x[j - 1], x[j + 1], 10).tolist())[0]
    A[0][0] = 1
    A[-1][-1] = 1
    b[0] = 0
    b[-1] = 0
    # print(A)
    # print(b)
    # plt.imshow(A, cmap='viridis')
    # plt.colorbar()
    # plt.show()

    res = np.linalg.solve(A, b)
    return res


def solve(h):
    n = int(1 / h) + 1
    x = np.linspace(0, 1, n)
    p = solver(n, x)
    target = [np.sin(2 * math.pi * iks) for iks in x]
    return p, target, x

def count_c_norm(res, target):
    tmp = np.abs(np.array(res[1:-1]) - np.array(target[1:-1])) / np.array(target[1:-1])
    # print(len(np.array(target)))
    # print(len(np.array(target[1:-1])))
    # print(tmp)
    return max(tmp)

c_norm = []
h = [0.1, 0.01]#, 0.001]
plt.figure(figsize=(20, 20))
for i, h_ in enumerate(h):

    p, target, x_ax = solve(h_)
    plt.subplot(2, 2, 2*i + 1)
    plt.title(f'Solution $h={h_}$')
    plt.plot(x_ax, p, label='my')
    plt.plot(x_ax, target, label='target')
    plt.legend()
    plt.grid()
    plt.subplot(2, 2, 2*i + 2)
    plt.plot(x_ax, np.abs(p - target))
    plt.title("Error")
    c_norm.append(count_c_norm(p, target))
    # plt.legend()
# plt.plot(h, c_norm)
# plt.title("C norm")
plt.show()
