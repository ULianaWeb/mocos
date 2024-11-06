import numpy as np
import matplotlib.pyplot as plt

# Визначення функції f(t)
def f(t):
    return t ** ((2 * 16 + 1) / 7)

# Діапазон значень для t
t_values = np.linspace(0, 10, 1000)
f_values = f(t_values)

# Побудова графіка
plt.figure(figsize=(8, 6))
plt.plot(t_values, f_values, label=r'$f(t) = t^{\frac{33}{7}}$', color='b')
plt.xlabel('t')
plt.ylabel('f(t)')
#plt.title('Графік функції $f(t) = |t|^{\frac{33}{7}}$')
plt.legend()
plt.grid(True)
plt.show()
