import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Ellipse
from scipy.integrate import odeint


def ode_system(y, _t, h, _m1, _m2, _g, j0, m0, r, _w, _c):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = 1.5 * _m1 + _m2
    a12 = _m2 * h * np.cos(y[1])
    b1 = _m2 * h * y[3] ** 2 * np.sin(y[1]) + (m0 / r) * np.cos(_w * _t) - _c * y[0]

    a21 = _m2 * h * np.cos(y[1])
    a22 = j0
    b2 = -_m2 * _g * h * np.sin(y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy


def spring_coords(x1, x2, y, num_segments=15):
    """Создаёт координаты для ломаной линии пружины"""
    x_coords = np.linspace(x1, x2, num_segments + 1)
    y_coords = np.zeros_like(x_coords) + y
    for i in range(1, len(x_coords) - 1, 2):
        y_coords[i] += 0.008
        y_coords[i + 1] -= 0.008
    return x_coords, y_coords


m1 = 40
m2 = 10
R = 0.1  # Радиус колеса
OA = 0.06  # Длина маятника (большая полуось эллипса)
J0 = 0.04
M0 = 39.2
c = 0.2
w = np.pi / c
y0 = [0, 0.5, 0, 0]

l0 = 0.11  # Длина недеформированной пружины
bar_length = 0.11  # Длина стержня
bar_thickness = 0.01  # Толщина стержня

x_cil_center = l0 + bar_length  # Координата x центра цилиндра

# Параметры движения
a_s = 0.16  # Амплитуда колебания пружины

g = 9.81  # Ускорение свободного падения
L = OA + 0.01  # Длина математического маятника

# Временные параметры
t_max = 4  # Длительность анимации
fps = 60  # Частота кадров
t = np.linspace(0, t_max, t_max * fps)

Y = odeint(ode_system, y0, t, (OA, m1, m2, g, J0, M0, R, w, c))

s_values = Y[:, 0]
phi_values = Y[:, 1]

d_s_values = Y[:, 2]
d_phi_values = Y[:, 3]

dd_s_values = [ode_system(y, t, OA, m1, m2, g, J0, M0, R, w, c)[2] for y, t in zip(Y, t)]
dd_phi_values = [ode_system(y, t, OA, m1, m2, g, J0, M0, R, w, c)[3] for y, t in zip(Y, t)]

rox_values = m2 * (dd_s_values + OA * (dd_phi_values * np.cos(phi_values) - d_phi_values * np.sin(phi_values)))
roy_values = m2 * (g + OA * (dd_phi_values * np.sin(phi_values) + d_phi_values * np.cos(phi_values)))

# Создаем окно с 4 графиками в формате 2x2
gr_fig, gr_axs = plt.subplots(2, 2, figsize=(15, 7))
gr_fig.canvas.manager.set_window_title('Вариант 4')

x_ticks = [i * 0.5 for i in range(9)]

# График 1: s_values от времени
gr_axs[0, 0].plot(t, s_values, color='blue')
gr_axs[0, 0].set_title('s(t)')
gr_axs[0, 0].set_ylabel('s')
gr_axs[0, 0].set_xticks(x_ticks)
gr_axs[0, 0].grid(True)

# График 2: phi_values от времени
gr_axs[0, 1].plot(t, phi_values, color='green')
gr_axs[0, 1].set_title('phi(t)')
gr_axs[0, 1].set_ylabel('phi')
gr_axs[0, 1].set_xticks(x_ticks)
gr_axs[0, 1].grid(True)

# График 3: rox_values от времени
gr_axs[1, 0].plot(t, rox_values, color='red')
gr_axs[1, 0].set_title('Rox(t)')
gr_axs[1, 0].set_ylabel('rox')
gr_axs[1, 0].set_xlabel('Time')
gr_axs[1, 0].set_xticks(x_ticks)
gr_axs[1, 0].grid(True)

# График 4: roy_values от времени
gr_axs[1, 1].plot(t, roy_values, color='purple')
gr_axs[1, 1].set_title('Roy(t)')
gr_axs[1, 1].set_ylabel('roy')
gr_axs[1, 1].set_xlabel('Time')
gr_axs[1, 1].set_xticks(x_ticks)
gr_axs[1, 1].grid(True)

# Оставляем место между графиками
plt.tight_layout()
plt.get_current_fig_manager().window.wm_geometry('+5+5')

# Инициализация значений для t = 0
x_spring_0 = l0 + s_values[0]
phi_angle_0 = phi_values[0]

# Координаты цилиндра
theta = np.linspace(0, 2 * np.pi, 100)
x_circle_0 = s_values[0] + x_cil_center + R * np.cos(theta)
y_circle_0 = R + R * np.sin(theta)

# Координаты пружины
spring_x_0, spring_y_0 = spring_coords(0, x_spring_0, R, num_segments=15)

# Координаты стержня
bar_x_0 = [
    x_spring_0,
    x_spring_0,
    x_spring_0 + bar_length,
    x_spring_0 + bar_length,
]
bar_y = [
    R - bar_thickness / 2,
    R + bar_thickness / 2,
    R + bar_thickness / 2,
    R - bar_thickness / 2,
]

# Эллипс маятника
x_pendulum_0 = x_cil_center + (OA * 2 / 3) * np.sin(phi_angle_0)
y_pendulum_0 = R - (OA * 2 / 3) * np.cos(phi_angle_0)

# Точка O
pivot_0 = (x_cil_center, R)

# Линия OA
x_OA_0 = [x_cil_center, x_cil_center + OA * np.sin(phi_angle_0)]
y_OA_0 = [R, R - OA * np.cos(phi_angle_0)]

# Точка А
end_point_0 = (x_OA_0[1], y_OA_0[1])

figure, axis = plt.subplots()
figure.canvas.manager.set_window_title('Вариант 4')
axis.set_xlim(-0.1, 0.6)
axis.set_ylim(-0.05, 0.25)
axis.set_aspect('equal')
axis.grid()

angle_diameter = -(s_values[0] - l0) * (2 * np.pi) / a_s
x_diameter = [
    s_values[0] + x_cil_center - R * np.cos(angle_diameter),
    s_values[0] + x_cil_center + R * np.cos(angle_diameter)
]
y_diameter = [R - R * np.sin(angle_diameter), R + R * np.sin(angle_diameter)]

diameter_line, = axis.plot(x_diameter, y_diameter, 'b-', lw=2, zorder=1)
cylinder, = axis.plot(x_circle_0, y_circle_0, 'b', lw=2, zorder=1)  # Цилиндр
spring, = axis.plot(spring_x_0, spring_y_0, 'k', lw=2)  # Пружина
bar = Polygon([[bar_x_0[0], bar_y[0]], [bar_x_0[1], bar_y[1]], [bar_x_0[2], bar_y[2]],
               [bar_x_0[3], bar_y[3]]], closed=True, color='m')  # Прямоугольный стержень
axis.add_patch(bar)

# Эллипс маятника
pendulum_ellipse = Ellipse((x_pendulum_0, y_pendulum_0), width=0.04, height=(2 * OA) / 1.5, color='g')
axis.add_patch(pendulum_ellipse)

point_o, = axis.plot(pivot_0[0], pivot_0[1], 'ro')  # Точка O
line_oa, = axis.plot(x_OA_0, y_OA_0, 'r-', lw=2)  # Линия OA
point_a, = axis.plot(end_point_0[0], end_point_0[1], 'ro')  # Точка на конце линии OA

axis.plot([-0.1, 0.6], [0, 0], 'black')  # Ось OX
axis.plot([0, 0], [-0.15, 0.25], 'black')  # Ось OY


def animate(i):
    global angle_diameter, x_diameter, y_diameter
    angle_diameter = -(s_values[i] + x_cil_center) * (2 * np.pi) / a_s
    x_diameter = [
        s_values[i] + x_cil_center - R * np.cos(angle_diameter),
        s_values[i] + x_cil_center + R * np.cos(angle_diameter)
    ]
    y_diameter = [R - R * np.sin(angle_diameter), R + R * np.sin(angle_diameter)]
    diameter_line.set_data(x_diameter, y_diameter)

    # Координаты колеса
    x_circle = s_values[i] + x_cil_center + R * np.cos(theta)
    y_circle = R + R * np.sin(theta)
    cylinder.set_data(x_circle, y_circle)

    # Координаты пружины
    spring_x, spring_y = spring_coords(0, s_values[i] + l0, R)
    spring.set_data(spring_x, spring_y)

    # Координаты стержня
    bar_x = [
        s_values[i] + l0,
        s_values[i] + l0,
        s_values[i] + x_cil_center,
        s_values[i] + x_cil_center,
    ]
    bar.set_xy(np.column_stack([bar_x, bar_y]))

    # Эллипс маятника
    x_pendulum = s_values[i] + x_cil_center + (OA * 2 / 3) * np.sin(phi_values[i])
    y_pendulum = R - (OA * 2 / 3) * np.cos(phi_values[i])
    pendulum_ellipse.set_center((x_pendulum, y_pendulum))
    pendulum_ellipse.angle = np.degrees(phi_values[i])

    # Точка O
    point_o.set_data(s_values[i] + x_cil_center, R)

    # Линия OA
    x_oa = [s_values[i] + x_cil_center, s_values[i] + x_cil_center + OA * np.sin(phi_values[i])]
    y_oa = [R, R - OA * np.cos(phi_values[i])]
    line_oa.set_data(x_oa, y_oa)

    # Точка на конце линии OA
    point_a.set_data(x_oa[1], y_oa[1])

    return diameter_line, pendulum_ellipse, cylinder, spring, bar, point_o, line_oa, point_a


# Создание анимации
animation = FuncAnimation(figure, animate, frames=len(t), interval=1000 / fps, blit=True)

# Показываем график
plt.show()
