import sympy
import numpy as np
from scipy.optimize import minimize

VA, VB, VC, VD, VCC, R1, R2, R3, R4 = sympy.symbols('VA VB VC VD VCC R1 R2 R3 R4')

eqs = [
    sympy.Eq(VA, (VCC/R1 + VC/R3 + VB/R3 + VD/R4)/(1/R1 + 2/R3 + 1/R4)),
    sympy.Eq(VB, (VCC/R2 + VD/R3 + VA/R3)/(1/R2 + 2/R3)),
    sympy.Eq(VC, (VA/R3 + VD/R3)/(1/R2 + 2/R3)),
    sympy.Eq(VD, (VB/R3 + VC/R3 + VA/R4)/(1/R1 + 2/R3 + 1/R4))
]

vals = {VCC: 15, R1: 1000, R2: 2000, R3: 10000, R4: 500}

print("METHOD 1:")
solutions = sympy.solve(eqs, [VA, VB, VC, VD])
numeric_m1 = {v: sol.subs(vals).evalf() for v, sol in solutions.items()}
for var in [VA, VB, VC, VD]:
    print(f"{var} = {numeric_m1[var]:.4f} V")

print("\nMETHOD 2:")
A_sym = sympy.Matrix([
    [1/R1 + 2/R3 + 1/R4, -1/R3, -1/R3, -1/R4],
    [-1/R3, 1/R2 + 2/R3, 0, -1/R3],
    [1/R3, 0, -(1/R2 + 2/R3), 1/R3],
    [-1/R4, -1/R3, 1/R3, 1/R1 + 2/R3 + 1/R4]
])
B_sym = sympy.Matrix([VCC/R1, VCC/R2, 0, 0])

A_num = np.array(A_sym.subs(vals).evalf(), dtype=float)
B_num = np.array(B_sym.subs(vals).evalf(), dtype=float).flatten()

x = np.linalg.solve(A_num, B_num)
print(f"VA = {x[0]:.4f} V")
print(f"VB = {x[1]:.4f} V")
print(f"VC = {x[2]:.4f} V")
print(f"VD = {x[3]:.4f} V")

print("\nMETHOD 3:")
def cost_function(x_vals):
    va, vb, vc, vd = x_vals
    f1 = va - (15/1000 + vc/10000 + vb/10000 + vd/500)/(1/1000 + 2/10000 + 1/500)
    f2 = vb - (15/2000 + vd/10000 + va/10000)/(1/2000 + 2/10000)
    f3 = vc - (va/10000 + vd/10000)/(1/2000 + 2/10000)
    f4 = vd - (vb/10000 + vc/10000 + va/500)/(1/1000 + 2/10000 + 1/500)
    return np.sqrt(f1**2 + f2**2 + f3**2 + f4**2)

x0 = [0, 0, 0, 0]
result = minimize(cost_function, x0)
x_opt = result.x
print(f"VA = {x_opt[0]:.4f} V")
print(f"VB = {x_opt[1]:.4f} V")
print(f"VC = {x_opt[2]:.4f} V")
print(f"VD = {x_opt[3]:.4f} V")
