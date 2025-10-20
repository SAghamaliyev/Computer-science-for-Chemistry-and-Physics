import numpy as np
import matplotlib.pyplot as plt

def gillespie_algorithm(A0, B0, AB0, kf, kr, max_time):
    A = A0
    B = B0
    AB = AB0
    t = 0.00
    
    # Storage for results
    time_points = [t]
    A_values = [A]
    B_values = [B]
    AB_values = [AB]
    

    while t < max_time:
  
        r1 = kf * A * B  # Straight
        r2 = kr * AB     # Reverse 
        r_total = r1 + r2
        
        if r_total > 0:
            tau = np.random.exponential(1.0 / r_total)
        else:
            break
        
        t += tau

        if np.random.uniform(0, 1) < r1 / r_total:
            # Straight reaction 
            A -= 1
            B -= 1
            AB += 1
        else:
            # Reverse reaction
            A += 1
            B += 1
            AB -= 1
        
        # Save
        time_points.append(t)
        A_values.append(A)
        B_values.append(B)
        AB_values.append(AB)
    
    return np.array(time_points), np.array(A_values), np.array(B_values), np.array(AB_values)

A0 = 800   
B0 = 400      
AB0 = 100     
kf = 0.05     
kr = 0.005
max_time = 1

# Gathering data
time, A, B, AB = gillespie_algorithm(A0, B0, AB0, kf, kr, max_time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, A, label='A', linewidth=2)
plt.plot(time, B, label='B', linewidth=2)
plt.plot(time, AB, label='AB', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Concentration', fontsize=12)
plt.title("Gillespie's Algorithm: Chemical Reaction Simulation", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
