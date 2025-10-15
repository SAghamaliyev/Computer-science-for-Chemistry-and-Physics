import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =============================================================================
# 1. RC-BRIDGE CIRCUIT
# =============================================================================

# Circuit parameters
R = 1e3  # 1 kΩ
C = 1e-9  # 1 nF
V_in = 5  # 5 V
tau = R * C  # Time constant

# Time span for simulation
t_final = 10e-6  # 10 µs
t_eval = np.linspace(0, t_final, 1000)

# Define the differential equation: dV/dt = (V_in - V) / (R*C)
def rc_circuit(t, V):
    return (V_in - V) / tau

# Initial condition: V(0) = 0
V0 = [0]

# Solve using scipy.integrate
solution_rc = solve_ivp(rc_circuit, [0, t_final], V0, t_eval=t_eval, method='RK45')

# Analytical solution: V(t) = V_in * (1 - exp(-t/tau))
V_analytical = V_in * (1 - np.exp(-solution_rc.t / tau))

# Calculate relative difference
relative_diff = (solution_rc.y[0] - V_analytical) / V_analytical * 100

# Create figure for RC circuit
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Voltage vs Time
ax1.plot(solution_rc.t * 1e6, solution_rc.y[0], 'b-', label='Numerical (scipy)', linewidth=2)
ax1.plot(solution_rc.t * 1e6, V_analytical, 'r--', label='Analytical', linewidth=2)
ax1.set_xlabel('Time (µs)', fontsize=12)
ax1.set_ylabel('Voltage (V)', fontsize=12)
ax1.set_title(f'RC Circuit: Capacitor Voltage Evolution\nR = {R/1e3} kΩ, C = {C*1e9} nF, τ = {tau*1e6:.2f} µs', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Relative Difference
ax2.plot(solution_rc.t * 1e6, relative_diff, 'g-', linewidth=2)
ax2.set_xlabel('Time (µs)', fontsize=12)
ax2.set_ylabel('Relative Difference (%)', fontsize=12)
ax2.set_title('Relative Difference: (Numerical - Analytical) / Analytical × 100%', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()



# =============================================================================
# 2. SIMPLE PENDULUM
# =============================================================================


# Parameters
L = 0.20  # 20 cm = 0.20 m
g = 9.81  # gravity

# Convert 2nd order to 1st order system
# d²θ/dt² = -(g/L)sin(θ)
# Let: θ = y[0], ω = dθ/dt = y[1]
# Then: dθ/dt = ω, dω/dt = -(g/L)sin(θ)

def simple_pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Analytical solution for small angles: θ(t) = θ0*cos(ωt) where ω = sqrt(g/L)
omega_0 = np.sqrt(g/L)

# Simulation time
t_span = [0, 10]
t_eval_pend = np.linspace(0, 10, 1000)

# Test small angle (5 degrees)
theta_small = np.radians(5)
y0_small = [theta_small, 0]
sol_small = solve_ivp(simple_pendulum, t_span, y0_small, t_eval=t_eval_pend, method='RK45')
theta_analytical_small = theta_small * np.cos(omega_0 * sol_small.t)

# Test large angle (60 degrees)
theta_large = np.radians(60)
y0_large = [theta_large, 0]
sol_large = solve_ivp(simple_pendulum, t_span, y0_large, t_eval=t_eval_pend, method='RK45')
theta_analytical_large = theta_large * np.cos(omega_0 * sol_large.t)

# Create figure for simple pendulum
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Small angle - trajectory
axes[0, 0].plot(sol_small.t, np.degrees(sol_small.y[0]), 'b-', label='Numerical', linewidth=2)
axes[0, 0].plot(sol_small.t, np.degrees(theta_analytical_small), 'r--', label='Analytical (small angle)', linewidth=2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (degrees)')
axes[0, 0].set_title('Small Angle (5°): Numerical vs Analytical')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Small angle - relative difference
rel_diff_small = (sol_small.y[0] - theta_analytical_small) / theta_analytical_small * 100
axes[0, 1].plot(sol_small.t, rel_diff_small, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Relative Difference (%)')
axes[0, 1].set_title('Small Angle: Relative Error')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

# Large angle - trajectory
axes[1, 0].plot(sol_large.t, np.degrees(sol_large.y[0]), 'b-', label='Numerical', linewidth=2)
axes[1, 0].plot(sol_large.t, np.degrees(theta_analytical_large), 'r--', label='Analytical (small angle)', linewidth=2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].set_title('Large Angle (60°): Numerical vs Analytical')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Large angle - relative difference
rel_diff_large = (sol_large.y[0] - theta_analytical_large) / theta_analytical_large * 100
axes[1, 1].plot(sol_large.t, rel_diff_large, 'g-', linewidth=2)
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Relative Difference (%)')
axes[1, 1].set_title('Large Angle: Relative Error')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

plt.tight_layout()


# =============================================================================
# 3. DOUBLE PENDULUM
# =============================================================================

# Parameters
L1 = 1.0  # Length of first pendulum (m)
L2 = 1.0  # Length of second pendulum (m)
m1 = 1.0  # Mass of first bob (kg)
m2 = 1.0  # Mass of second bob (kg)

# Convert to 4 first-order ODEs
# State vector: [θ1, ω1, θ2, ω2]
# where ω1 = dθ1/dt and ω2 = dθ2/dt

def double_pendulum(t, y):
    theta1, omega1, theta2, omega2 = y
    
    delta = theta2 - theta1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1
    
    dtheta1_dt = omega1
    
    domega1_dt = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * L2 * omega2**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1
    
    dtheta2_dt = omega2
    
    domega2_dt = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                  (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

# Simulation time
t_span_dp = [0, 20]
t_eval_dp = np.linspace(0, 20, 5000)

# Case 1: θ1 = 45°, θ2 = -45°
y0_case1 = [np.radians(45), 0, np.radians(-45), 0]
sol_case1 = solve_ivp(double_pendulum, t_span_dp, y0_case1, t_eval=t_eval_dp, method='RK45', rtol=1e-9)

# Case 2: θ1 = 30°, θ2 = 0°
y0_case2 = [np.radians(30), 0, np.radians(0), 0]
sol_case2 = solve_ivp(double_pendulum, t_span_dp, y0_case2, t_eval=t_eval_dp, method='RK45', rtol=1e-9)

# Calculate positions
def get_positions(sol):
    x1 = L1 * np.sin(sol.y[0])
    y1 = -L1 * np.cos(sol.y[0])
    x2 = x1 + L2 * np.sin(sol.y[2])
    y2 = y1 - L2 * np.cos(sol.y[2])
    return x1, y1, x2, y2

x1_c1, y1_c1, x2_c1, y2_c1 = get_positions(sol_case1)
x1_c2, y1_c2, x2_c2, y2_c2 = get_positions(sol_case2)

# Create figure for double pendulum cases 1 and 2
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# Case 1 - Angles
axes[0, 0].plot(sol_case1.t, np.degrees(sol_case1.y[0]), 'b-', label='θ₁', linewidth=1.5)
axes[0, 0].plot(sol_case1.t, np.degrees(sol_case1.y[2]), 'r-', label='θ₂', linewidth=1.5)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Angle (degrees)')
axes[0, 0].set_title('Case 1: θ₁(0)=45°, θ₂(0)=-45° - Angles vs Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Case 1 - Trajectory
axes[0, 1].plot(x2_c1, y2_c1, 'b-', linewidth=0.5, alpha=0.7)
axes[0, 1].plot(x2_c1[0], y2_c1[0], 'go', markersize=10, label='Start')
axes[0, 1].plot(x2_c1[-1], y2_c1[-1], 'ro', markersize=10, label='End')
axes[0, 1].set_xlabel('x₂ (m)')
axes[0, 1].set_ylabel('y₂ (m)')
axes[0, 1].set_title('Case 1: Trajectory of Second Mass')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axis('equal')
axes[0, 1].legend()

# Case 2 - Angles
axes[1, 0].plot(sol_case2.t, np.degrees(sol_case2.y[0]), 'b-', label='θ₁', linewidth=1.5)
axes[1, 0].plot(sol_case2.t, np.degrees(sol_case2.y[2]), 'r-', label='θ₂', linewidth=1.5)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Angle (degrees)')
axes[1, 0].set_title('Case 2: θ₁(0)=30°, θ₂(0)=0° - Angles vs Time')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Case 2 - Trajectory
axes[1, 1].plot(x2_c2, y2_c2, 'b-', linewidth=0.5, alpha=0.7)
axes[1, 1].plot(x2_c2[0], y2_c2[0], 'go', markersize=10, label='Start')
axes[1, 1].plot(x2_c2[-1], y2_c2[-1], 'ro', markersize=10, label='End')
axes[1, 1].set_xlabel('x₂ (m)')
axes[1, 1].set_ylabel('y₂ (m)')
axes[1, 1].set_title('Case 2: Trajectory of Second Mass')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axis('equal')
axes[1, 1].legend()

plt.tight_layout()

angles_test = [89, 90, 91]
theta2_test = 15
t_span_chaos = [0, 30]
t_eval_chaos = np.linspace(0, 30, 10000)

solutions_chaos = []
for angle in angles_test:
    y0 = [np.radians(angle), 0, np.radians(theta2_test), 0]
    sol = solve_ivp(double_pendulum, t_span_chaos, y0, t_eval=t_eval_chaos, method='RK45', rtol=1e-10)
    solutions_chaos.append(sol)

# Create figure for chaos analysis
fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['blue', 'red', 'green']
for i, (angle, sol) in enumerate(zip(angles_test, solutions_chaos)):
    x1, y1, x2, y2 = get_positions(sol)
    ax1.plot(x2, y2, color=colors[i], linewidth=0.5, alpha=0.6, label=f'θ₁(0) = {angle}°')

ax1.set_xlabel('x₂ (m)', fontsize=12)
ax1.set_ylabel('y₂ (m)', fontsize=12)
ax1.set_title('Trajectory Comparison: θ₁(0) = 89°, 90°, 91° (θ₂(0) = 15°)', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')

# Plot trajectory divergence over time
for i, (angle, sol) in enumerate(zip(angles_test, solutions_chaos)):
    x1, y1, x2, y2 = get_positions(sol)
    distance_from_origin = np.sqrt(x2**2 + y2**2)
    ax2.plot(sol.t, np.degrees(sol.y[0]), color=colors[i], linewidth=1.5, alpha=0.7, label=f'θ₁(0) = {angle}°')

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('θ₁ (degrees)', fontsize=12)
ax2.set_title('Angular Evolution: Sensitivity to Initial Conditions', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

plt.show()
