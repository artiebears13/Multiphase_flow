import math
import numpy as np
import matplotlib.pyplot as plt

rho = 1
mu = 1
k = 1
Delta_z = 1
r_w = 0.001
h = 0.125

inj_well_coord = (2, 2)
prod_well_coord = (6, 6)

p_bh_inj = 1
p_bh_prod = -1

r_e = np.exp(-math.pi / 2) * 0.125

FRAC = 2 * math.pi * rho * k * Delta_z / (mu * np.log(r_w / r_e))

N = 9


def delta_kr(i, j):
    if i == j:
        return 1
    else:
        return 0


def fill_upper_bound(i, j, A):
    if j == i:
        A[i][j] += 1
    if j == i + N:
        A[i][j] += -1


def fill_lower_bound(i, j, A):
    if j == i:
        A[i][j] += 1
    if j == i - N:
        A[i][j] += -1


def fill_left_bound(i, j, A):
    if j == i:
        A[i][j] += 1
    if j == i + 1:
        A[i][j] += -1


def fill_right_bound(i, j, A):
    if j == i:
        A[i][j] += 1
    if j == i - 1:
        A[i][j] += -1


def fill_boundary(i, j, A):
    boundary = False
    if 0 <= i <= (N - 1):
        boundary = True
        fill_upper_bound(i, j, A)
    if ((N - 1) * N) <= i <= (N ** 2 - 1):
        boundary = True
        fill_lower_bound(i, j, A)
    if i % N == 0:
        boundary = True
        fill_left_bound(i, j, A)
    if i % N == (N - 1):
        boundary = True
        fill_right_bound(i, j, A)
    return boundary


def fill_neighbours(i, j, A):
    if j == i - 1 or j == i + 1 or j == i - 9 or j == i + N:
        A[i][j] = -1 / h ** 2


def fill_wells(i, j, A):
    if i == j:
        A[i, j] = 4 / h ** 2 - FRAC
    else:
        fill_neighbours(i, j, A)


def fill_non_wells(i, j, A):
    if i == j:
        A[i, j] = 4 / h ** 2
    else:
        fill_neighbours(i, j, A)


def fill_matrix():
    A = np.zeros((N ** 2, N ** 2))
    b = np.zeros(N ** 2)
    for i in range(N ** 2):
        for j in range(N ** 2):
            boundary = fill_boundary(i, j, A)
            if not boundary:
                if ((i == j == N * inj_well_coord[1] + inj_well_coord[0])
                        or (i == j == N * prod_well_coord[1] + prod_well_coord[0])):
                    fill_wells(i, j, A)
                    if i == N * inj_well_coord[1] + inj_well_coord[0]:
                        b[i] = -p_bh_inj * FRAC
                    else:
                        b[i] = -p_bh_prod * FRAC
                else:
                    fill_non_wells(i, j, A)
    plt.imshow(A, cmap='viridis')
    plt.colorbar()
    plt.title('Matrix')
    plt.show()
    res = np.linalg.solve(A, b)
    res = np.array(res).reshape((N, N))
    return res


res = fill_matrix()
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

plt.imshow(res, cmap='viridis', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
plt.colorbar(label='Colorbar Label')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('$P(x,y)$')
# plt.grid()

plt.show()
print()
