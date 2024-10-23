import numpy as np
import matplotlib.pyplot as plt
import time


def compute_fft(f):
    N = len(f)

    start_time = time.time()
    C = np.fft.fft(f)
    end_time = time.time()

    fft_time = end_time - start_time

    num_operations = N * np.log2(N) * 4

    return C, fft_time, int(num_operations)


# C_k
def display_coefficients(C):
    print("\nКоефіцієнти C_k для ШПФ:")
    for k, c in enumerate(C):
        print(f"C[{k}] = {c:.4f}")


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


def fast():
    N = 26
    f = np.random.randn(N)

    C, fft_time, total_operations = compute_fft(f)
    display_coefficients(C)

    print(f"\nЧас обчислення ШПФ: {fft_time:.6f} секунд")
    print(f"Оцінена кількість операцій (множення + додавання): {total_operations}")

    plot_spectrums(C, N)


if __name__ == "__main__":
    fast()
