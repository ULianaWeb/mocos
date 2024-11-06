import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning
import warnings
warnings.filterwarnings("ignore", category=IntegrationWarning)


def f(t):
    return abs(t) ** ((2 * 16 + 1) / 7)


# Функція для обчислення дійсної та уявної частин інтегралу Фур’є
def compute_fourier_component(w_k, N=1600):
    Re_F, _ = quad(lambda t: f(t) * np.cos(-w_k * t), -N, N)
    Im_F, _ = quad(lambda t: f(t) * np.sin(-w_k * t), -N, N)
    print(f"F(w_{w_k:.2f}) = {Re_F:.4f} + j*{Im_F:.4f}")
    return Re_F, Im_F


# Функція для обчислення спектру амплітуд
def amplitude_spectrum(Re_F, Im_F):
    return np.sqrt(Re_F ** 2 + Im_F ** 2)


# Головна функція для обчислення та побудови графіків
def plot_fourier_transform(T_values, k_max=20, N=1600):
    for T in T_values:
        w_k_values = [(2 * np.pi * k) / T for k in range(k_max)]
        Re_F_values = []
        amplitude_values = []

        print(f"\n=== Результати для T = {T} ===")

        for w_k in w_k_values:
            Re_F, Im_F = compute_fourier_component(w_k, N)
            Re_F_values.append(Re_F)
            amplitude_values.append(amplitude_spectrum(Re_F, Im_F))

            # Виведення значень спектру амплітуд для кожного w_k
            amplitude = amplitude_spectrum(Re_F, Im_F)
            print(f"|F(w_{w_k:.2f})| = {amplitude:.4f}")

        # Побудова графіків
        plt.figure(figsize=(12, 6))

        # Графік Re F(w_k)
        plt.subplot(1, 2, 1)
        plt.plot(w_k_values, Re_F_values, label=f'T = {T}')
        plt.xlabel('w_k')
        plt.ylabel('Re F(w_k)')
        plt.title('Дійсна частина F(w_k)')
        plt.legend()

        # Графік |F(w_k)|
        plt.subplot(1, 2, 2)
        plt.plot(w_k_values, amplitude_values, label=f'T = {T}')
        plt.xlabel('w_k')
        plt.ylabel('|F(w_k)|')
        plt.title('Спектр амплітуд |F(w_k)|')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Виклик функції з різними значеннями T
T_values = [4, 8, 16, 32, 64, 128]
plot_fourier_transform(T_values)
