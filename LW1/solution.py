import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math


def rot2d(_x, _y, angle):
    """Преобразование данных координат
    на соответствующий угол поворота"""
    return (
        _x * np.cos(angle) - _y * np.sin(angle),
        _x * np.sin(angle) + _y * np.cos(angle)
    )


def mdl(data_x, data_y, i):
    """Получение модуля вектора на i-ом шаге"""
    return round((data_x[i] ** 2 + data_y[i] ** 2) ** 0.5, 5)


# Вводим символьную переменную
t = sp.Symbol('t')

# Получаем уравнения изменения x и y по t
x = (1 + sp.sin(5 * t)) * sp.cos(t)
y = (1 + sp.sin(5 * t)) * sp.sin(t)

# Получаем скорость по x и y, а также модуль
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
V = (Vx ** 2 + Vy ** 2) ** 0.5

# То же для полного ускорения
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
W = (Wx ** 2 + Wy ** 2) ** 0.5

# То же для тангенциального ускорения
Wt = sp.diff(V, t)

# То же для нормального ускорения
Wn = (W ** 2 - Wt ** 2) ** 0.5

# Проекции тангенциального ускорения
Wtx = Vx / V * Wt
Wty = Vy / V * Wt

# Проекции нормального ускорения
Wnx = Wx - Wtx
Wny = Wy - Wty

# Единичные направления нормального ускорения
Nx = Wnx / Wn
Ny = Wny / Wn

# Радиус кривизны
curvature_module = ((V * V) / Wn)

# Его проекции
curvature_x = curvature_module * Nx
curvature_y = curvature_module * Ny

# Массив временных точек
T = np.linspace(0, 6.28, 1000)

# Получение функций изменения величин
Fx = sp.lambdify(t, x, 'numpy')
Fy = sp.lambdify(t, y, 'numpy')

FVx = sp.lambdify(t, Vx, 'numpy')
FVy = sp.lambdify(t, Vy, 'numpy')

FWx = sp.lambdify(t, Wx, 'numpy')
FWy = sp.lambdify(t, Wy, 'numpy')

F_curvature_x = sp.lambdify(t, curvature_x, 'numpy')
F_curvature_y = sp.lambdify(t, curvature_y, 'numpy')

# Массивы координат
X = Fx(T)
Y = Fy(T)

# Отмасштабированные массивы скорости и ускорения
VX = FVx(T) * 0.2
VY = FVy(T) * 0.2

WX = FWx(T) * 0.05
WY = FWy(T) * 0.05

# Массивы радиуса кривизны
CX = F_curvature_x(T)
CY = F_curvature_y(T)

# Инициализация pyplot
figure = plt.figure()
figure.canvas.manager.set_window_title('Вариант 2')

# Настройка окна отображения графика
axis = figure.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(ylim=[-2.5, 3])

# Отрисовка материальной точки
axis.plot(X, Y, 'black')
point = axis.plot(X[0], Y[0], marker='o', markerfacecolor='grey', markeredgecolor='red', markersize=9)[0]

# Отрисовка вектора скорости
v_line = axis.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]], 'r', label='Скорость (V)')[0]

# Массивы координат стрелки вектора скорости
x_v_arrow = np.array([-0.05, 0, -0.05])
y_v_arrow = np.array([0.05, 0, -0.05])

# Стартовый поворот стрелки и её отрисовка
r_x_v_arrow, r_y_v_arrow = rot2d(x_v_arrow, y_v_arrow, math.atan2(VY[0], VX[0]))
v_arrow = axis.plot(r_x_v_arrow + X[0] + VX[0], r_y_v_arrow + Y[0] + VY[0], 'r')[0]

# Отрисовка вектора ускорения
w_line = axis.plot([X[0], X[0] + WX[0]], [Y[0], Y[0] + WY[0]], 'g', label='Ускорение (W)')[0]

# Массивы координат стрелки вектора ускорения
x_w_arrow = np.array([-0.05, 0, -0.05])
y_w_arrow = np.array([0.05, 0, -0.05])

# Стартовый поворот стрелки и её отрисовка
r_x_w_arrow, r_y_w_arrow = rot2d(x_w_arrow, y_w_arrow, math.atan2(WY[0], WX[0]))
w_arrow = axis.plot(r_x_w_arrow + X[0] + WX[0], r_y_w_arrow + Y[0] + WY[0], 'g')[0]

# Отрисовка радиус-вектора
r_line = axis.plot([0, X[0]], [0, Y[0]], 'b', label='Радиус-вектор (R)')[0]

# Массивы координат стрелки радиус-вектора
x_r_arrow = np.array([-0.05, 0, -0.05])
y_r_arrow = np.array([0.05, 0, -0.05])

# Стартовый поворот стрелки и её отрисовка
r_x_r_arrow, r_y_r_arrow = rot2d(x_r_arrow, y_r_arrow, math.atan2(Y[0], X[0]))
r_arrow = axis.plot(r_x_r_arrow + X[0], r_y_w_arrow + Y[0], 'b')[0]

# Отрисовка радиуса кривизны
curvature_radius = axis.plot([X[0], X[0] + CX[0]], [Y[0], Y[0] + CY[0]], 'black', label='Радиус кривизны (CR)',
                             linestyle='--')[0]

# Шаблон сообщения со значениями величин
raw_text = 'R   = {r}\nV   = {v}\nW  = {w}\nCR = {cr}'

# Установка сообщения в начальное состояние
text = axis.text(0.03, 0.03, raw_text.format(r=mdl(X, Y, 0), v=mdl(VX, VY, 0), w=mdl(WX, WY, 0),
                                             cr=mdl(CX, CY, 0)), transform=axis.transAxes, fontsize=8)


def animate(i):
    """Изменение величин для нового шага анимации"""
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

    text.set_text(raw_text.format(r=mdl(X, Y, i), v=mdl(VX, VY, i), w=mdl(WX, WY, i), cr=mdl(CX, CY, i)))

    return point, v_line, v_arrow, curvature_radius


# Запуск анимации
animation = FuncAnimation(figure, animate, frames=1000, interval=1)

# Вывод легенды
axis.legend(fontsize=7)

# Запуск pyplot
plt.show()
