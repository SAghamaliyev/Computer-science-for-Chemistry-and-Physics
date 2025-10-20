import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
k1 = 1      # Forward rate E + S -> ES
k_minus1 = 0.01  # Reverse rate ES -> E + S
k2 = 5     # Forward rate ES -> E + P

# Initial conditions
E0 = 10       # Initial enzyme
S0 = 200      # Initial substrate
P0 = 0        # Initial product
ES0 = 0       # Initial enzyme-substrate complex
max_time = 5  # seconds


def gillespie_enzymatic(E0, S0, P0, ES0, k1, k_minus1, k2, max_time):
    
    E = E0
    S = S0
    P = P0
    ES = ES0
    t = 0.0
    
    time_points = [t]
    E_vals = [E]
    S_vals = [S]
    P_vals = [P]
    ES_vals = [ES]
    
    while t < max_time:
        # Calculate reaction rates
        r1 = k1 * E * S      # E + S -> ES
        r2 = k_minus1 * ES   # ES -> E + S
        r3 = k2 * ES         # ES -> E + P
        
        r_total = r1 + r2 + r3
        
        if r_total < 1e-10:
            break
        
        # Draw time from exponential distribution
        tau = np.random.exponential(1.0 / r_total)
        t += tau
        
        # Choose which reaction occurs
        rand = np.random.uniform(0, 1)
        
        if rand < r1 / r_total:
            # Reaction 1: E + S -> ES
            E -= 1
            S -= 1
            ES += 1
        elif rand < (r1 + r2) / r_total:
            # Reaction 2: ES -> E + S
            E += 1
            S += 1
            ES -= 1
        else:
            # Reaction 3: ES -> E + P
            E += 1
            P += 1
            ES -= 1
        
        time_points.append(t)
        E_vals.append(E)
        S_vals.append(S)
        P_vals.append(P)
        ES_vals.append(ES)
    
    return np.array(time_points), np.array(E_vals), np.array(S_vals), np.array(P_vals), np.array(ES_vals)


def enzymatic_ode(y, t, k1, k_minus1, k2):
    E, S, ES, P = y
    
    dE_dt = -k1 * E * S + k_minus1 * ES + k2 * ES
    dS_dt = -k1 * E * S + k_minus1 * ES
    dES_dt = k1 * E * S - k_minus1 * ES - k2 * ES
    dP_dt = k2 * ES
    
    return [dE_dt, dS_dt, dES_dt, dP_dt]


time_g, E_g, S_g, P_g, ES_g = gillespie_enzymatic(E0, S0, P0, ES0, k1, k_minus1, k2, max_time)

y0 = [E0, S0, ES0, P0]
t_ode = np.linspace(0, max_time, 1000)
y_ode = odeint(enzymatic_ode, y0, t_ode, args=(k1, k_minus1, k2))
E_ode = y_ode[:, 0]
S_ode = y_ode[:, 1]
ES_ode = y_ode[:, 2]
P_ode = y_ode[:, 3]

# Create comparison plots
fig, axes = plt.subplots(2, 1, figsize=(11, 10))

axes[1].plot(time_g, E_g, 'o-', label='Gillespie (E)', markersize=4, alpha=0.7, color='C0')
axes[1].plot(t_ode, E_ode, '-', label='Deterministic (E)', linewidth=2, color='C0')
axes[1].plot(time_g, S_g, 's-', label='Gillespie (S)', markersize=4, alpha=0.7, color='C1')
axes[1].plot(t_ode, S_ode, '-', label='Deterministic (S)', linewidth=2, color='C1')
axes[1].plot(time_g, ES_g, '^-', label='Gillespie (ES)', markersize=4, alpha=0.7, color='C2')
axes[1].plot(t_ode, ES_ode, '-', label='Deterministic (ES)', linewidth=2, color='C2')
axes[1].plot(time_g, P_g, 'd-', label='Gillespie (P)', markersize=4, alpha=0.7, color='C3')
axes[1].plot(t_ode, P_ode, '-', label='Deterministic (P)', linewidth=2, color='C3')
axes[1].set_ylabel('Concentration', fontsize=11)
axes[1].set_title('All Species: E, S, ES & P', fontsize=12)
axes[1].legend(fontsize=8, ncol=2)
axes[1].grid(True, alpha=0.3)

# All species - Gillespie (bottom left)
axes[0].plot(time_g, E_g, 'o-', label='E', markersize=1, alpha=0.7)
axes[0].plot(time_g, S_g, 's-', label='S', markersize=1, alpha=0.7)
axes[0].plot(time_g, ES_g, '^-', label='ES', markersize=1, alpha=0.7)
axes[0].plot(time_g, P_g, 'd-', label='P', markersize=1, alpha=0.7)
axes[0].set_xlabel('Time (s)', fontsize=11)
axes[0].set_ylabel('Concentration', fontsize=11)
axes[0].set_title('Gillespie Algorithm - All Species', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

plt.show()
