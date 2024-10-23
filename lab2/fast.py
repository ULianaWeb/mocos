import numpy as np
import matplotlib.pyplot as plt
import time


# Реалізація Швидкого Перетворення Фур'є (FFT)
def compute_fft(f):
    N = len(f)

    # Використовуємо np.fft.fft для ШПФ
    start_time = time.time()
    C = np.fft.fft(f)
    end_time = time.time()

    # Час обчислення
    fft_time = end_time - start_time

    # Кількість операцій
    num_operations = N * np.log2(N) * 4

    return C, fft_time, int(num_operations)


# Побудова спектрів амплітуд та фаз
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


# Головна функція
def fast():
    N = 26
    f = np.random.randn(N)  # Генерація довільного сигналу

    # Обчислення FFT
    C, fft_time, total_operations = compute_fft(f)

    # Виведення результатів
    print(f"\nЧас обчислення ШПФ: {fft_time:.6f} секунд")
    print(f"Оцінена кількість операцій (множення + додавання): {total_operations}")

    # Побудова графіків
    plot_spectrums(C, N)


if __name__ == "__main__":
    fast()
