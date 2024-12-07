import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Ellipse
import sympy as sp


def spring_coords(x1, x2, y, num_segments=15):
    """Создаёт координаты для ломаной линии пружины"""
    x_coords = np.linspace(x1, x2, num_segments + 1)
    y_coords = np.zeros_like(x_coords) + y
    for i in range(1, len(x_coords) - 1, 2):
        y_coords[i] += 0.008
        y_coords[i + 1] -= 0.008
    return x_coords, y_coords


# Исходные параметры
R = 0.1  # Радиус колеса
OA = 0.06  # Длина маятника (большая полуось эллипса)
l0 = 0.11  # Длина недеформированной пружины
bar_length = 0.11  # Длина стержня
bar_thickness = 0.01  # Толщина стержня

# Параметры движения
a_s = 0.16  # Амплитуда колебания пружины
omega_s = 2 * np.pi / 4  # Угловая частота пружины

g = 9.81  # Ускорение свободного падения
L = OA + 0.01  # Длина математического маятника
omega_phi = 2 * np.pi  # Частота 1 полный цикл в секунду
A_phi = np.pi / 6  # Амплитуда угла (30 градусов)

# Временные параметры
t_max = 4  # Длительность анимации
fps = 30  # Частота кадров
t = np.linspace(0, t_max, t_max * fps)

t_sym = sp.symbols('t')

# Функции для колебаний с использованием символьной переменной и lambdify
s_expr = l0 + a_s * (1.2 + sp.sin(omega_s * t_sym))
phi_expr = A_phi * sp.cos(0.75 * omega_phi * t_sym)

s_func = sp.lambdify(t_sym, s_expr, 'numpy')
phi_func = sp.lambdify(t_sym, phi_expr, 'numpy')

s_values = s_func(t)  # Массив значений для колебания пружины
phi_values = phi_func(t)  # Массив значений для угла маятника

# Инициализация значений для t = 0
x_cylinder_0 = s_values[0]
phi_angle_0 = phi_values[0]

# Координаты цилиндра
theta = np.linspace(0, 2 * np.pi, 100)
x_circle_0 = x_cylinder_0 + R * np.cos(theta)
y_circle_0 = R + R * np.sin(theta)  # Центр поднят на R

# Координаты пружины
spring_x_0, spring_y_0 = spring_coords(0, x_cylinder_0 - bar_length, R, num_segments=15)

# Координаты стержня
bar_x_0 = [
    x_cylinder_0 - bar_length,
    x_cylinder_0 - bar_length,
    x_cylinder_0,
    x_cylinder_0,
]
bar_y_0 = [
    R - bar_thickness / 2,
    R + bar_thickness / 2,
    R + bar_thickness / 2,
    R - bar_thickness / 2,
]

# Эллипс маятника
x_pendulum_0 = x_cylinder_0 + (OA * 2 / 3) * np.sin(phi_angle_0)
y_pendulum_0 = R - (OA * 2 / 3) * np.cos(phi_angle_0)

# Точка O
pivot_0 = (x_cylinder_0, R)

# Линия OA
x_OA_0 = [x_cylinder_0, x_cylinder_0 + OA * np.sin(phi_angle_0)]
y_OA_0 = [R, R - OA * np.cos(phi_angle_0)]

# Точка А
end_point_0 = (x_OA_0[1], y_OA_0[1])

fig, ax = plt.subplots()
ax.set_xlim(-0.1, 0.6)
ax.set_ylim(-0.05, 0.25)
ax.set_aspect('equal')
ax.grid()

angle_diameter = -(s_values[0] - l0) * (2 * np.pi) / a_s
x_diameter = [s_values[0] - R * np.cos(angle_diameter), s_values[0] + R * np.cos(angle_diameter)]
y_diameter = [R - R * np.sin(angle_diameter), R + R * np.sin(angle_diameter)]

diameter_line, = ax.plot(x_diameter, y_diameter, 'b-', lw=2, zorder=1)

cylinder, = ax.plot(x_circle_0, y_circle_0, 'b', lw=2)  # Цилиндр
spring, = ax.plot(spring_x_0, spring_y_0, 'k', lw=2)  # Пружина
bar = Polygon([[bar_x_0[0], bar_y_0[0]], [bar_x_0[1], bar_y_0[1]], [bar_x_0[2], bar_y_0[2]],
               [bar_x_0[3], bar_y_0[3]]], closed=True, color='m')  # Прямоугольный стержень
ax.add_patch(bar)

# Эллипс маятника
pendulum_ellipse = Ellipse((x_pendulum_0, y_pendulum_0), width=0.04, height=(2 * OA) / 1.5, color='g')
ax.add_patch(pendulum_ellipse)

point_o, = ax.plot(pivot_0[0], pivot_0[1], 'ro')  # Точка O
line_oa, = ax.plot(x_OA_0, y_OA_0, 'r-', lw=2)  # Линия OA
point_a, = ax.plot(end_point_0[0], end_point_0[1], 'ro')  # Точка на конце линии OA

ax.plot([-0.1, 0.6], [0, 0], 'black')  # Ось OX
ax.plot([0, 0], [-0.15, 0.25], 'black')  # Ось OY


def animate(i):
    # Обновление диаметра колеса
    angle_diameter = -(s_values[i] - l0) * (2 * np.pi) / a_s
    x_diameter = [s_values[i] - R * np.cos(angle_diameter), s_values[i] + R * np.cos(angle_diameter)]
    y_diameter = [R - R * np.sin(angle_diameter), R + R * np.sin(angle_diameter)]
    diameter_line.set_data(x_diameter, y_diameter)

    # Координаты колеса
    x_circle = s_values[i] + R * np.cos(theta)
    y_circle = R + R * np.sin(theta)  # Центр поднят на R
    cylinder.set_data(x_circle, y_circle)

    # Координаты пружины
    spring_x, spring_y = spring_coords(0, s_values[i] - bar_length, R)
    spring.set_data(spring_x, spring_y)

    # Координаты стержня
    bar_x = [
        s_values[i] - bar_length,
        s_values[i] - bar_length,
        s_values[i],
        s_values[i],
    ]
    bar_y = [
        R - bar_thickness / 2,
        R + bar_thickness / 2,
        R + bar_thickness / 2,
        R - bar_thickness / 2,
    ]
    bar.set_xy(np.column_stack([bar_x, bar_y]))

    # Эллипс маятника
    x_pendulum = s_values[i] + (OA * 2 / 3) * np.sin(phi_values[i])
    y_pendulum = R - (OA * 2 / 3) * np.cos(phi_values[i])
    pendulum_ellipse.set_center((x_pendulum, y_pendulum))
    pendulum_ellipse.angle = np.degrees(phi_values[i])

    # Точка O
    point_o.set_data(s_values[i], R)

    # Линия OA
    x_oa = [s_values[i], s_values[i] + OA * np.sin(phi_values[i])]
    y_oa = [R, R - OA * np.cos(phi_values[i])]
    line_oa.set_data(x_oa, y_oa)

    # Точка на конце линии OA
    point_a.set_data(x_oa[1], y_oa[1])

    return diameter_line, pendulum_ellipse, cylinder, spring, bar, point_o, line_oa, point_a


# Создание анимации
animation = FuncAnimation(fig, animate, frames=len(t), interval=1000 / fps, blit=True)

plt.show()
