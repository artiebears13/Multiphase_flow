import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# parameters init
L = 100  # ft
phi = 0.2
krw_st = 0.6
kro_st = 1.0
mu_o = 50  # cP
n_w = 2
n_o = 2
S_ali = 1.0  # 453.6 * 28.32 # lb/ft^3
A = 1  # ft
S_wr = 0.2
S_or = 0.15
T = 70  # C
q = 1
t_end = 10
t_step = 0.1
tau = t_step
x_step = 1  # ft
P = 187 * 14.6  # psi
T_f = T * 1.8 + 32.0
F_sv = 1 - 1.87 * (10 ** (-3)) * (S_ali ** (1 / 2)) + 2.18 * (10 ** (-4)) * (S_ali ** (5 / 2)) + \
       (T_f ** (1 / 2) - 0.0135 * T_f) * (2.76 * (10 ** (-3)) * S_ali - 3.44 * (10 ** (-4)) * (S_ali ** (3 / 2)))
T_k = (T + 273.15)
hx = 1
F_pv = 1 + 3.5 * 10 ** -12 * P * (T_f - 40)
mu_w = 0.02414 * np.power(10, 247.8 / (T_k - 140)) * F_sv * F_pv


def k_ro(sw_n):
    # if np.abs(sw_n - S_wr) < 0.0001:
    #     return kro_st
    return kro_st * ((1 - sw_n) ** n_o)


def k_rw(sw_n):
    # if np.abs(sw_n - (1 - S_or)) < 0.0001:
    #     return krw_st
    return krw_st * (sw_n ** n_w)


def lambda_w(sw_n):
    return k_rw(sw_n) / mu_w


def lambda_o(sw_n):
    return k_ro(sw_n) / mu_o


def get_swn(sw):
    return (sw - S_wr) / (1 - S_or - S_wr)


def fw(sw):
    swn = get_swn(sw)
    if sw <= S_wr:
        return 0.0
    if sw >= 1 - S_or:
        return 1.0
    return lambda_w(swn) / (lambda_w(swn) + lambda_o(swn))


def solver():
    sw_cur = np.zeros(100)
    sw_prev = np.ones(100) * S_wr
    sw_prev[0] = 1 - S_or
    number = 0

    def func(sw):
        res = np.zeros(len(sw))
        for i in range(1, len(sw) - 1):
            res[i] = -tau / phi * q / A * ((fw(sw[i]) - fw(sw[i - 1]))  / hx) + sw_prev[i] - sw[i]

        res[0] = (1 - S_or) - sw[0]
        res[-1] = S_wr - sw[-1]
        return res

    # print(res)
    t = 0.0 + tau
    t_end = 1

    while t < t_end:
        root = fsolve(func, sw_prev)
        sw_prev = root
        sw_prev[0] = 1 - S_or
        sw_prev[-1] = S_wr
        x = np.linspace(0, 100, 100)
        plt.plot(x, sw_prev)
        plt.title(f"t={round(t, 2)}")
        plt.savefig(f'graphs/png_{number}.png')
        plt.show()
        # print(sw_prev)

        t += t_step
        number += 1
    return x, sw_prev

sw_step = 0.01
def find_s_star():
    # sw_step = 0.01
    # sw = [i for i in np.arange(S_wr, 1 - S_or, sw_step)]

    def equation(sw_st):
        return (fw(sw_st + sw_step) - fw(sw_st)) / sw_step - fw(sw_st) / (sw_st - S_wr)

    root = fsolve(equation, 0.3)
    print(root)
    return root
    # print(root)


def analytic_sol(sw, swf, step):
    t = 1.0
    if sw > swf:
        res = t * q / A / phi * (fw(sw+step) - fw(sw)) / step
    else:
        res = t * q / A / phi * (fw(swf+step) - fw(swf)) / step
    if res < 0:
        print(f"res < 0 if {sw=} {sw + step=} {fw(sw+step)=} {fw(sw)=} {res =}")
    return res
    #     return S_wr
    # return dx_dt

def solve_analytical():

    sw = np.linspace(S_wr, 1-S_or, 101)
    print(f"{sw=}")
    plt.plot(sw, [fw(sw_) for sw_ in sw])
    plt.show()
    plt.plot(sw, [k_rw(get_swn(sw_)) for sw_ in sw])
    plt.plot(sw, [k_ro(get_swn(sw_)) for sw_ in sw])
    plt.show()
    # print(f"{sw=}")

    swf = find_s_star()[0]
    print(f"{swf=}")
    step = 0.01
    res = [analytic_sol(sw_, swf, step) for sw_ in sw]
    print(res)
    # plt.plot(res, sw)
    # plt.show()
    return res, sw


if __name__ == '__main__':
    # print(mu_w)
    # sw = np.linspace(0, 1, 100)
    # k_rw_vec = [k_rw(sw_i) for sw_i in sw]
    # k_ro_vec = [k_ro(sw_i) for sw_i in sw]
    # plt.plot(sw, k_rw_vec)
    # plt.plot(sw, k_ro_vec)
    # plt.show()
    # fw_vec = [fw(sw_i) for sw_i in sw]
    # plt.plot(sw, fw_vec)
    # plt.grid()
    # plt.show()

    # x1, sw1 = solve_analytical()
    x, sw = solver()


    # plt.plot(x1, sw1)
    # plt.plot(x, sw)
    # plt.show()
