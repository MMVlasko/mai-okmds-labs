import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Количество шагов анимации
STEPS = 1001

# Продолжительность анимации
FINAL_TIME = 20

# Длина тележки
BOX_X = 6

# Высота тележки
BOX_Y = 2

# Длина пружины в недеформированном состоянии
X0 = 4

# Радиус колеса
WHEEL_R = 0.5

# Длина стержня
LENGTH = 3
BLACK = 'black'
RED = 'red'

# Массив временных точек
t = np.linspace(0, FINAL_TIME, STEPS)

# Выражения для изменения угла и координаты
x = np.cos(0.2 * t) + 3 * np.sin(1.8 * t)
phi = 2 * np.sin(1.7 * t) + 5 * np.cos(1.2 * t)

# Выражения координат центра тележки
x_a = X0 + x + BOX_X / 2
y_a = 2 * WHEEL_R + BOX_Y / 2 + 0.2

# Выражения координат конца стержня
x_b = x_a + LENGTH * np.sin(phi)
y_b = y_a + LENGTH * np.cos(phi)

# Массивы относительных координат углов тележки для отрисовки
x_box = np.array([-BOX_X / 2, BOX_X / 2, BOX_X / 2, -BOX_X / 2, -BOX_X / 2])
y_box = np.array([BOX_Y / 2, BOX_Y / 2, -BOX_Y / 2, -BOX_Y / 2, BOX_Y / 2])

# Массив углов и выражения для отрисовки окружностей колёс
psi = np.linspace(0, 2 * math.pi, 30)
x_wheel = WHEEL_R * np.cos(psi)
y_wheel = WHEEL_R * np.sin(psi)

# Выражения для координат центров колёс
x_c1 = X0 + x + BOX_X / 5
y_c1 = WHEEL_R
x_c2 = X0 + x + 4 * BOX_X / 5
y_c2 = WHEEL_R

# Координаты отрисовки координатных осей
x_ground = np.array([0, 0, 15])
y_ground = np.array([6, 0, 0])

# Число точек перегиба пружины
K = 19

# Относительная высота точки перегиба пружины
SH = 0.4

# Расстояние между двумя точками перегиба (кроме крайних)
B = 1 / (K - 2)

# Инициализация массивов координат точек перегиба
x_spr = np.zeros(K)
y_spr = np.zeros(K)

# Начальная координата крайней правой точки
x_spr[K - 1] = 1

# Заполнение массивов координатами
for j in range(K - 2):
    x_spr[j + 1] = B * ((j + 1) - 1 / 2)
    y_spr[j + 1] = SH * (-1) ** j

# Массив длин пружины в разные моменты
spright_length = X0 + x

# Количество полных витков
NV = 3

# Минимальный радиус пружины
R1 = 0.2

# Максимальный радиус пружины
R2 = 0.9

# Получения массива углов и координат точек спиральной пружины
theta = np.linspace(0, NV * 2 * math.pi - phi[0], 100)
x_spiral_spr = -(R1 + theta * (R2 - R1) / theta[-1]) * np.sin(theta)
y_spiral_spr = (R1 + theta * (R2 - R1) / theta[-1]) * np.cos(theta)

# Массив углов вращения колёс
alpha = x / WHEEL_R

# Инициализация окна с графиком
figure = plt.figure(figsize=[10, 5])
figure.canvas.manager.set_window_title('Вариант 2')

axis = figure.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim=[0, 15], ylim=[-1.5, 6.5])

# Отрисовка координатных осей
axis.plot(x_ground, y_ground, BLACK)

# Отрисовка тележки и колёс
drawn_box = axis.plot(x_box + x_a[0], y_box + y_a, BLACK)[0]
drawn_wheel1 = axis.plot(x_wheel + x_c1[0], y_wheel + y_c1, BLACK)[0]
drawn_wheel2 = axis.plot(x_wheel + x_c2[0], y_wheel + y_c2, BLACK)[0]

# Отрисовка креплений колёс
drawn_vc1 = axis.plot([x_c1[0] - WHEEL_R, x_c1[0], x_c1[0] + WHEEL_R], [2 * y_c1 + 0.2, y_c1, 2 * y_c1 + 0.2], BLACK)[0]
drawn_vc2 = axis.plot([x_c2[0] - WHEEL_R, x_c2[0], x_c2[0] + WHEEL_R], [2 * y_c2 + 0.2, y_c2, 2 * y_c2 + 0.2], BLACK)[0]

# Отрисовка обычной пружины
drawn_spring = axis.plot(x_spr * spright_length[0], y_spr + y_a, RED)[0]

# Отрисовка спиральной пружины
drawn_spiral_spring = axis.plot(x_spiral_spr + x_a[0], y_spiral_spr + y_a, RED)[0]

# Отрисовка диаметров колёс
drawn_wheel1_d = axis.plot([x_c1[0] + WHEEL_R * np.sin(alpha[0]), x_c1[0] - WHEEL_R * np.sin(alpha[0])],
                           [y_c1 + WHEEL_R * np.cos(alpha[0]), y_c1 - WHEEL_R * np.cos(alpha[0])], BLACK)[0]
drawn_wheel2_d = axis.plot([x_c2[0] + WHEEL_R * np.sin(alpha[0]), x_c2[0] - WHEEL_R * np.sin(alpha[0])],
                           [y_c2 + WHEEL_R * np.cos(alpha[0]), y_c2 - WHEEL_R * np.cos(alpha[0])], BLACK)[0]

# Отрисовка стержня
drawn_ab = axis.plot([x_a[0], x_b[0]], [y_a, y_b[0]], BLACK)[0]

# Отрисовка шарнира и груза на конце стержня
point_a = axis.plot(x_a[0], y_a, marker='o')[0]
point_b = axis.plot(x_b[0], y_b[0], marker='o', markersize=20)[0]


def animate(i):
    """Изменение величин для нового шага анимации"""
    drawn_box.set_data(x_box + x_a[i], y_box + y_a)

    drawn_wheel1.set_data(x_wheel + x_c1[i], y_wheel + y_c1)
    drawn_wheel2.set_data(x_wheel + x_c2[i], y_wheel + y_c2)

    drawn_vc1.set_data([x_c1[i] - WHEEL_R, x_c1[i], x_c1[i] + WHEEL_R], [2 * y_c1 + 0.2, y_c1, 2 * y_c1 + 0.2])
    drawn_vc2.set_data([x_c2[i] - WHEEL_R, x_c2[i], x_c2[i] + WHEEL_R], [2 * y_c2 + 0.2, y_c2, 2 * y_c2 + 0.2])

    drawn_ab.set_data([x_a[i], x_b[i]], [y_a, y_b[i]])
    point_a.set_data(x_a[i], y_a)
    point_b.set_data(x_b[i], y_b[i])

    drawn_spring.set_data(x_spr * spright_length[i], y_spr + y_a)
    _theta = np.linspace(0, NV * 2 * math.pi - phi[i], 100)
    _x_spiral_spr = -(R1 + _theta * (R2 - R1) / _theta[-1]) * np.sin(_theta)
    _y_spiral_spr = (R1 + _theta * (R2 - R1) / _theta[-1]) * np.cos(_theta)
    drawn_spiral_spring.set_data(_x_spiral_spr + x_a[i], _y_spiral_spr + y_a)

    drawn_wheel1_d.set_data([x_c1[i] + WHEEL_R * np.sin(alpha[i]), x_c1[i] - WHEEL_R * np.sin(alpha[i])],
                            [y_c1 + WHEEL_R * np.cos(alpha[i]), y_c1 - WHEEL_R * np.cos(alpha[i])])
    drawn_wheel2_d.set_data([x_c2[i] + WHEEL_R * np.sin(alpha[i]), x_c2[i] - WHEEL_R * np.sin(alpha[i])],
                            [y_c2 + WHEEL_R * np.cos(alpha[i]), y_c2 - WHEEL_R * np.cos(alpha[i])])

    return (
        drawn_box, drawn_wheel1, drawn_wheel2, drawn_vc1, drawn_vc2, drawn_ab, point_a,
        point_b, drawn_spring, drawn_spiral_spring, drawn_wheel1_d, drawn_wheel2_d
    )


# Запуск анимации
anim = FuncAnimation(figure, animate, frames=len(t), interval=40, repeat=False)

# Запуск pyplot
plt.show()
