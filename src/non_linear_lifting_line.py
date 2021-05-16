import numpy as np

N = 20
Vinf = 1
vinf = Vinf / np.linalg.norm(Vinf)
gamma = ''
va = ''
w = ''
v = ''
un= ''
vn = ''
ua = ''
Cell = ''
dCl = ''


def f(G):
    for i in range(0, N):
        F = np.zeros((1, N))
        S = 0
        for j in range(N):
            S += v[j][i] * G[j]
        F[i] = 2 * np.linalg.norm(np.cross(vinf + S, gamma[i])) * G[i] - Cell[i]
    return F


def df(G):
    for i in range(0, N):
        df = np.zeros((N, N))
    for j in range(0, N):
        if i != j:
            df[i, j] = 2 * np.dot(w[i], np.cross(v[j, i], gamma[i])) / np.norm(w[i]) * G[i] - dCl[i] * (
                        va[i] * np.dot(v[j, i], un[i]) - vn[i] * np.dot(v[j, i], ua[i])) / (va[i] ** 2 + vn[i] ** 2)
        else:
            df[i, j] = 2 * np.norm(w[i]) + 2 * np.dot(w[i], np.cross(v[j, i], gamma[i])) * G[i] / np.norm(w[i]) - dCl[
                i] * (va[i] * np.dot(v[j, i], un[i]) - vn[i] * np.dot(v[j, i], ua[i])) / (va[i] ** 2 + vn[i] ** 2)
