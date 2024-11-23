import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

STEPS = 1001
FINAL_TIME = 20
BOX_X = 6
BOX_Y = 2
SPR_X_0 = 4
WHEEL_R = 0.5
LENGTH = 1.85
BLACK = 'black'
RED = 'red'

t = np.linspace(0, FINAL_TIME, STEPS)

x = np.cos(0.2 * t) + 3 * np.sin(1.8 * t)
phi = 2 * np.sin(1.7 * t) + 5 * np.cos(1.2 * t)

x_a = SPR_X_0 + x + BOX_X / 2
y_a = 2 * WHEEL_R + BOX_Y / 2 + 0.2

x_b = x_a + LENGTH * np.sin(phi)
y_b = y_a + LENGTH * np.cos(phi)

x_box = np.array([-BOX_X / 2, BOX_X / 2, BOX_X / 2, -BOX_X / 2, -BOX_X / 2])
y_box = np.array([BOX_Y / 2, BOX_Y / 2, -BOX_Y / 2, -BOX_Y / 2, BOX_Y / 2])

psi = np.linspace(0, 2 * math.pi, 30)
x_wheel = WHEEL_R * np.cos(psi)
y_wheel = WHEEL_R * np.sin(psi)

x_c1 = SPR_X_0 + x + BOX_X / 5
y_c1 = WHEEL_R
x_c2 = SPR_X_0 + x + 4 * BOX_X / 5
y_c2 = WHEEL_R

x_ground = np.array([0, 0, 15])
y_ground = np.array([6, 0, 0])

K = 19
SH = 0.4
B = 1 / (K - 2)
x_spr = np.zeros(K)
y_spr = np.zeros(K)
x_spr[0] = 0
y_spr[0] = 0
x_spr[K - 1] = 1
y_spr[K - 1] = 0

for j in range(K - 2):
    x_spr[j + 1] = B * ((j + 1) - 1 / 2)
    y_spr[j + 1] = SH * (-1) ** j

spright_length = SPR_X_0 + x

NV = 3
R1 = 0.05
R2 = 0.8

theta = np.linspace(0, NV * 2 * math.pi - phi[0], 100)
x_spiral_spr = -(R1 + theta * (R2 - R1) / theta[-1]) * np.sin(theta)
y_spiral_spr = (R1 + theta * (R2 - R1) / theta[-1]) * np.cos(theta)

alpha = x / WHEEL_R

figure = plt.figure(figsize=[10, 5])
figure.canvas.manager.set_window_title('Вариант 2')

axis = figure.add_subplot(1, 1, 1)
axis.axis('equal')
axis.set(xlim=[0, 15], ylim=[-1.5, 6.5])

axis.plot(x_ground, y_ground, BLACK)

drew_box = axis.plot(x_box + x_a[0], y_box + y_a, BLACK)[0]
drew_wheel1 = axis.plot(x_wheel + x_c1[0], y_wheel + y_c1, BLACK)[0]
drew_wheel2 = axis.plot(x_wheel + x_c2[0], y_wheel + y_c2, BLACK)[0]

drew_vc1 = axis.plot([x_c1[0] - WHEEL_R, x_c1[0], x_c1[0] + WHEEL_R], [2 * y_c1 + 0.2, y_c1, 2 * y_c1 + 0.2], BLACK)[0]
drew_vc2 = axis.plot([x_c2[0] - WHEEL_R, x_c2[0], x_c2[0] + WHEEL_R], [2 * y_c2 + 0.2, y_c2, 2 * y_c2 + 0.2], BLACK)[0]

drew_spring = axis.plot(x_spr * spright_length[0], y_spr + y_a, RED)[0]
drew_spiral_spring = axis.plot(x_spiral_spr + x_a[0], y_spiral_spr + y_a, RED)[0]

drew_wheel1_d1 = axis.plot([x_c1[0] + WHEEL_R * np.sin(alpha[0]), x_c1[0] - WHEEL_R * np.sin(alpha[0])],
                           [y_c1 + WHEEL_R * np.cos(alpha[0]), y_c1 - WHEEL_R * np.cos(alpha[0])], BLACK)[0]
drew_wheel2_d2 = axis.plot([x_c2[0] + WHEEL_R * np.sin(alpha[0]), x_c2[0] - WHEEL_R * np.sin(alpha[0])],
                           [y_c2 + WHEEL_R * np.cos(alpha[0]), y_c2 - WHEEL_R * np.cos(alpha[0])], BLACK)[0]

drew_ab = axis.plot([x_a[0], x_b[0]], [y_a, y_b[0]], BLACK)[0]

point_a = axis.plot(x_a[0], y_a, marker='o')[0]
point_b = axis.plot(x_b[0], y_b[0], marker='o', markersize=20)[0]


def animate(i):
    drew_box.set_data(x_box + x_a[i], y_box + y_a)

    drew_wheel1.set_data(x_wheel + x_c1[i], y_wheel + y_c1)
    drew_wheel2.set_data(x_wheel + x_c2[i], y_wheel + y_c2)

    drew_vc1.set_data([x_c1[i] - WHEEL_R, x_c1[i], x_c1[i] + WHEEL_R], [2 * y_c1 + 0.2, y_c1, 2 * y_c1 + 0.2])
    drew_vc2.set_data([x_c2[i] - WHEEL_R, x_c2[i], x_c2[i] + WHEEL_R], [2 * y_c2 + 0.2, y_c2, 2 * y_c2 + 0.2])

    drew_ab.set_data([x_a[i], x_b[i]], [y_a, y_b[i]])
    point_a.set_data(x_a[i], y_a)
    point_b.set_data(x_b[i], y_b[i])

    drew_spring.set_data(x_spr * spright_length[i], y_spr + y_a)
    _theta = np.linspace(0, NV * 2 * math.pi - phi[i], 100)
    _x_spiral_spr = -(R1 + _theta * (R2 - R1) / _theta[-1]) * np.sin(_theta)
    _y_spiral_spr = (R1 + _theta * (R2 - R1) / _theta[-1]) * np.cos(_theta)
    drew_spiral_spring.set_data(_x_spiral_spr + x_a[i], _y_spiral_spr + y_a)

    drew_wheel1_d1.set_data([x_c1[i] + WHEEL_R * np.sin(alpha[i]), x_c1[i] - WHEEL_R * np.sin(alpha[i])],
                            [y_c1 + WHEEL_R * np.cos(alpha[i]), y_c1 - WHEEL_R * np.cos(alpha[i])])
    drew_wheel2_d2.set_data([x_c2[i] + WHEEL_R * np.sin(alpha[i]), x_c2[i] - WHEEL_R * np.sin(alpha[i])],
                            [y_c2 + WHEEL_R * np.cos(alpha[i]), y_c2 - WHEEL_R * np.cos(alpha[i])])

    return (
        drew_box, drew_wheel1, drew_wheel2, drew_vc1, drew_vc2, drew_ab, point_a,
        point_b, drew_spring, drew_spiral_spring, drew_wheel1_d1, drew_wheel2_d2
    )


anim = FuncAnimation(figure, animate, frames=len(t), interval=40, repeat=False)

plt.show()
