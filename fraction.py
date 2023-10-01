import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

phi_m = 0.2
phi_f = 0.01

mu = 1
b_w = 1
p = 500
# c_t = 10 ** -6

L = 10 ** 4
A_ = 200000
c_t = 10**-6 * A_
k_m = 0.01 * A_
k_f = 50 * A_
lambda_ = 100
pm_t0 = 1000
pf_t0 = 1000
pf_x0 = 500
pm_x0 = 500

x_step = 500
t_step = 0.01
# t_end = 10
t_end = 0.03
N = int(L / x_step)




def create_matrix(p_f_prev, p_m_prev):
    A = np.zeros((2 * N, 2 * N))
    b = np.zeros(2 * N)

    # fill for m
    for i in range(1, N -1):
        for j in range(0, N):
            if i == j:
                A[i][j] = phi_m * c_t / t_step + k_m / mu * 2 / (x_step ** 2) - lambda_
                A[i][j + N] = lambda_
            if j == i - 1 or j == i + 1:
                A[i][j] = - k_m / mu * 1 / (x_step ** 2)
        b[i] = phi_m * c_t / t_step * p_m_prev[i]
    # boundaries
    A[0][0] = 1
    b[0] = pm_x0
    A[N - 1][N - 1] = 1
    A[N - 1][N - 2] = -1
    b[N - 1] = 0
    # fill for f
    for i in range(N+1, 2 * N-1):
        for j in range(0, N):
            num = i - N
            if num == j:
                A[i][j + N] = phi_f * c_t / t_step + k_f / mu * 2 / (x_step ** 2) - lambda_
                A[i][j] = lambda_
            if j == num - 1 or j == num + 1:
                A[i][j + N] = - k_f / mu * 1 / (x_step ** 2)
        b[i] = phi_f * c_t / t_step * p_f_prev[i-N]
    A[N][N] = 1
    b[N] = pf_x0
    A[2 * N - 1][2 * N - 1] = 1
    A[2 * N - 1][2 * N - 2] = -1
    b[2*N - 1] = 0

    plt.imshow(A, cmap='viridis')
    plt.colorbar()
    plt.show()
    # plt.plot(np.linspace(0, 200, 200), b )
    # plt.show()

    # print(A, b)

    # print(A[100])
    return A, b

def solver():
    t = t_step
    p_f_prev = np.ones(N) * pf_t0

    p_m_prev = np.ones(N) * pm_t0
    print(p_m_prev)
    x = np.linspace(0, L, N)
    while t < t_end:
        A, b = create_matrix(p_f_prev, p_m_prev)

        res = solve(A,b)
        # print(f"{res=}")
        p_m_prev = list(res)[:N]
        p_f_prev = list(res)[N:2*N]

        # print(f"{np.array(p_m_prev)-np.array(p_f_prev)=}")
        # print(p_m_prev[10])
        # print(p_f_prev[10])
        plt.plot(x, p_m_prev, label='p_m')
        plt.plot(x, p_f_prev, label='p_f')
        plt.ylim(0,1000)
        plt.legend()
        plt.show()
        t+=t_step





if __name__ == '__main__':
    solver()
