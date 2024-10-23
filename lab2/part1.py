import numpy as np
import matplotlib.pyplot as plt
import time


# A_k і B_k
def fourier_term(f, N, k):
    operations_count = 0  # Лічильник операцій (множення + додавання)
    A_k = (2 / N) * sum(f[n] * np.cos(2 * np.pi * k * n / N) for n in range(N))
    operations_count += 2 * N
    B_k = (2 / N) * sum(f[n] * np.sin(2 * np.pi * k * n / N) for n in range(N))
    operations_count += 2 * N
    return A_k, B_k, operations_count


# C_k = A_k + jB_k
def compute_fourier_coefficients(f, N):
    C = []
    total_operations = 0
    for k in range(N):
        A_k, B_k, operations_count = fourier_term(f, N, k)
        C.append(complex(A_k, -B_k))
        total_operations += operations_count
    return C, total_operations


def display_coefficients(C):
    print("\nКоефіцієнти C_k для ШПФ:")
    for k, c in enumerate(C):
        print(f"C[{k}] = {c:.4f}")


# спектр амплітуд |C_k| і спектр фаз argC_k
def plot_spectrums(C, N):
    amplitudes = [abs(c) for c in C]
    phases = [np.angle(c) for c in C]

    k_values = np.arange(N)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.stem(k_values, amplitudes)
    plt.title('Amplitude Spectrum')
    plt.xlabel('k')
    plt.ylabel('|C_k|')

    plt.subplot(1, 2, 2)
    plt.stem(k_values, phases)
    plt.title('Phase Spectrum')
    plt.xlabel('k')
    plt.ylabel('arg(C_k)')

    plt.tight_layout()
    plt.show()


# мейн
def part1():

    N = 26
    f = np.random.randn(N)                                      # генерація довільного сигналу

    start_time = time.time()
    C, total_operations = compute_fourier_coefficients(f, N)    # обчислення C_k

    display_coefficients(C)

    end_time = time.time()
    computation_time = end_time - start_time

    print(f"\nЧас обчислення: {computation_time:.6f} секунд")
    print(f"Загальна кількість операцій (множення + додавання): {total_operations}")

    plot_spectrums(C, N)


if __name__ == "__main__":
    part1()
