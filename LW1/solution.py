import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def rot2d(_x, _y, angle):
    return (
        _x * np.cos(angle) - _y * np.sin(angle),
        _x * np.sin(angle) + _y * np.cos(angle)
    )


def mdl(data, i):
    return round((data[i] ** 2 + data[i] ** 2) ** 0.5, 5)


t = sp.Symbol('t')

x = (1 + sp.sin(5 * t)) * sp.cos(t)
y = (1 + sp.sin(5 * t)) * sp.sin(t)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
V = (Vx ** 2 + Vy ** 2) ** 0.5

Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
W = (Wx ** 2 + Wy ** 2) ** 0.5

Wt = sp.diff(V, t)
Wn = (W ** 2 - Wt ** 2) ** 0.5

Wtx = Vx / V * Wt
Wty = Vy / V * Wt

Wnx = Wx - Wtx
Wny = Wy - Wty

Nx = Wnx / Wn
Ny = Wny / Wn

curvature_module = ((V * V) / Wn)
curvature_x = curvature_module * Nx
curvature_y = curvature_module * Ny

T = np.linspace(0, 6.28, 1000)

Fx = sp.lambdify(t, x, 'numpy')
Fy = sp.lambdify(t, y, 'numpy')

FVx = sp.lambdify(t, Vx, 'numpy')
FVy = sp.lambdify(t, Vy, 'numpy')

FWx = sp.lambdify(t, Wx, 'numpy')
FWy = sp.lambdify(t, Wy, 'numpy')

F_curvature_x = sp.lambdify(t, curvature_x, 'numpy')
F_curvature_y = sp.lambdify(t, curvature_y, 'numpy')

X = Fx(T)
Y = Fy(T)

VX = FVx(T) * 0.2
VY = FVy(T) * 0.2

WX = FWx(T) * 0.05
WY = FWy(T) * 0.05

CX = F_curvature_x(T)
CY = F_curvature_y(T)

figure = plt.figure()
figure.canvas.manager.set_window_title('Вариант 2')

axis = figure.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(ylim=[-2.5, 3])

axis.plot(X, Y, 'black')
point = axis.plot(X[0], Y[0], marker='o', markerfacecolor='grey', markeredgecolor='red', markersize=9)[0]

v_line = axis.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r', label='Скорость (V)')[0]

x_v_arrow = np.array([-0.05, 0, -0.05])
y_v_arrow = np.array([0.05, 0, -0.05])

r_x_v_arrow, r_y_v_arrow = rot2d(x_v_arrow, y_v_arrow, math.atan2(VY[0], VX[0]))
v_arrow = axis.plot(r_x_v_arrow + X[0] + VX[0], r_y_v_arrow + Y[0] + VY[0], 'r')[0]

w_line = axis.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'g', label='Ускорение (W)')[0]

x_w_arrow = np.array([-0.05, 0, -0.05])
y_w_arrow = np.array([0.05, 0, -0.05])

r_x_w_arrow, r_y_w_arrow = rot2d(x_w_arrow, y_w_arrow, math.atan2(WY[0], WX[0]))
w_arrow = axis.plot(r_x_w_arrow + X[0] + WX[0], r_y_w_arrow + Y[0] + WY[0], 'g')[0]

r_line = axis.plot([0, X[0]], [0, Y[0]], 'b', label='Радиус-вектор (R)')[0]

x_r_arrow = np.array([-0.05, 0, -0.05])
y_r_arrow = np.array([0.05, 0, -0.05])

r_x_r_arrow, r_y_r_arrow = rot2d(x_r_arrow, y_r_arrow, math.atan2(Y[0], X[0]))
r_arrow = axis.plot(r_x_r_arrow + X[0], r_y_w_arrow + Y[0], 'b')[0]

curvature_radius = axis.plot([X[0], X[0] + CX[0]], [Y[0], Y[0] + CY[0]], 'black', label='Радиус кривизны (CR)',
                             linestyle='--')[0]

raw_text = 'R   = {r}\nV   = {v}\nW  = {w}\nCR = {cr}'

text = axis.text(0.03, 0.03, raw_text.format(r=mdl(X, 0), v=mdl(VX, 0), w=mdl(WX, 0), cr=mdl(CX, 0)),
                 transform=axis.transAxes, fontsize=8)


def animate(i):
    global r_y_v_arrow, r_x_v_arrow, r_x_w_arrow, r_y_w_arrow, r_x_r_arrow, r_y_r_arrow

    point.set_data(X[i], Y[i])

    v_line.set_data([X[i], X[i] + VX[i]], [Y[i], Y[i] + VY[i]])
    r_x_v_arrow, r_y_v_arrow = rot2d(x_v_arrow, y_v_arrow, math.atan2(VY[i], VX[i]))
    v_arrow.set_data(r_x_v_arrow + X[i] + VX[i], r_y_v_arrow + Y[i] + VY[i])

    w_line.set_data([X[i], X[i] + WX[i]], [Y[i], Y[i] + WY[i]])
    r_x_w_arrow, r_y_w_arrow = rot2d(x_w_arrow, y_w_arrow, math.atan2(WY[i], WX[i]))
    w_arrow.set_data(r_x_w_arrow + X[i] + WX[i], r_y_w_arrow + Y[i] + WY[i])

    r_line.set_data([0, X[i]], [0, Y[i]])
    r_x_r_arrow, r_y_r_arrow = rot2d(x_r_arrow, y_r_arrow, math.atan2(Y[i], X[i]))
    r_arrow.set_data(r_x_r_arrow + X[i], r_y_r_arrow + Y[i])

    curvature_radius.set_data([X[i], X[i] + CX[i]], [Y[i], Y[i] + CY[i]])

    text.set_text(raw_text.format(r=mdl(X, i), v=mdl(VX, i), w=mdl(WX, i), cr=mdl(CX, i)))

    return point, v_line, v_arrow, curvature_radius


animation = FuncAnimation(figure, animate, frames=1000, interval=1)

axis.legend(fontsize=7)

plt.show()
