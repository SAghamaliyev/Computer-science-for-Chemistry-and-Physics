import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
t = np.linspace(0, 20, 1000)
dt = t[1] - t[0]

x_t = 50 * np.sin(0.1 * np.pi * t)
y_t = 50 * np.sin(0.2 * np.pi * t)
ax_t = -0.5 * np.pi**2 * np.sin(0.1 * np.pi * t)
ay_t = -2 * np.pi**2 * np.sin(0.2 * np.pi * t)

x_t_noisy = x_t + np.random.normal(5, 0.001, len(t))
y_t_noisy = y_t + np.random.normal(5, 0.001, len(t))
ax_t_noisy = ax_t + np.random.normal(0.1, 1, len(t))
ay_t_noisy = ay_t + np.random.normal(0.1, 1, len(t))

vx_noisy = np.gradient(x_t_noisy, dt)
vy_noisy = np.gradient(y_t_noisy, dt)
ax_from_noisy_pos = np.gradient(vx_noisy, dt)
ay_from_noisy_pos = np.gradient(vy_noisy, dt)

vx_from_noisy_acc = cumulative_trapezoid(ax_t_noisy, t, initial=0)
vy_from_noisy_acc = cumulative_trapezoid(ay_t_noisy, t, initial=0)
x_from_noisy_acc = cumulative_trapezoid(vx_from_noisy_acc, t, initial=0)
y_from_noisy_acc = cumulative_trapezoid(vy_from_noisy_acc, t, initial=0)

#1a:Trajectory
plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.plot(x_t, y_t, 'b-', linewidth=2)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('1a:Trajectory')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

#1b:Acceleration
plt.plot(t, ax_t, 'g-', label='ax(t)', linewidth=2)
plt.plot(t, ay_t, 'm-', label='ay(t)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('1b:Acceleration/Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#2a:Noisy Trajectory
plt.plot(x_t, y_t, 'b--', label='Original', linewidth=2, alpha=0.5)
plt.plot(x_t_noisy, y_t_noisy, 'r-', label='Noisy', linewidth=1)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('2a:Noisy Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#2b:Noisy Acceleration
plt.plot(t, ax_t, 'g--', label='Ideal ax(t)', linewidth=2, alpha=0.5)
plt.plot(t, ax_from_noisy_pos, 'darkgreen', label='Calculated ax(t)', linewidth=1, alpha=0.7)
plt.plot(t, ay_t, 'm--', label='Ideal ay(t)', linewidth=2, alpha=0.5)
plt.plot(t, ay_from_noisy_pos, 'purple', label='Calculated ay(t)', linewidth=1, alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('2b. Acceleration from Noisy Position')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#2c:
error_ax = ax_from_noisy_pos - ax_t
error_ay = ay_from_noisy_pos - ay_t
plt.plot(t, error_ax, 'g-', label='Error ax', linewidth=1)
plt.plot(t, error_ay, 'm-', label='Error ay', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Acceleration Error')
plt.title('2c. Impact on derivative process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#3a:
plt.plot(t, ax_t, 'g--', label='Ideal ax(t)', linewidth=2, alpha=0.5)
plt.plot(t, ax_t_noisy, 'darkgreen', label='Noisy ax(t)', linewidth=1)
plt.plot(t, ay_t, 'm--', label='Ideal ay(t)', linewidth=2, alpha=0.5)
plt.plot(t, ay_t_noisy, 'purple', label='Noisy ay(t)', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('3a:Noisy Acceleration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#3b:
plt.plot(t, x_from_noisy_acc, 'b-', label='Calculated x(t)', linewidth=1)
plt.plot(t, y_from_noisy_acc, 'r-', label='Calculated y(t)', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('3b. Position from Noisy Acceleration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#3c:
error_x = x_from_noisy_acc - x_t
error_y = y_from_noisy_acc - y_t
plt.plot(t, error_x, 'b-', label='Error x', linewidth=1)
plt.plot(t, error_y, 'r-', label='Error y', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Position Error')
plt.title('3c. Impact on derivative process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
