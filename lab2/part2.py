import numpy as np
import matplotlib.pyplot as plt


def generate_binary_signal_from_number(number):
    binary_str = format(number, '08b')                                       # 8-розрядне двійкове число
    signal = np.array([int(bit) for bit in binary_str])                      # Створюємо сигнал з 0 і 1
    print(f"\nЧисло {number} в 8-розрядній двійковій системі: {binary_str}")
    print(f"Згенерований сигнал: {signal}")
    return signal


# обчислення
def compute_complex_fourier_coefficients(signal):
    N = len(signal)
    C = np.zeros(N, dtype=complex)

    # C_k
    for k in range(N):
        for n in range(N):
            C[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        C[k] /= N

    # |C_n| і argC_n
    magnitudes = np.abs(C)
    phases = np.angle(C)

    print("\nКомплексні коефіцієнти ДПФ:")
    for n in range(N):
        print(f"C[{n}] = {C[n]:.4f}, |C[{n}]| = {magnitudes[n]:.4f}, arg(C[{n}]) = {phases[n]:.4f} рад")

    return C, magnitudes, phases


# s(t) загальне
def display_harmonic_sum(C, magnitudes, phases, Tc):
    equation = f"s(t) = {magnitudes[0]:.4f}"
    for k in range(1, len(C) // 2):
        equation += f" + 2 * {magnitudes[k]:.4f} * cos({2 * k} * pi * t / {Tc} + {phases[k]:.4f})"
    equation += f" + {magnitudes[len(C) // 2]:.4f} * cos({len(C)} * pi * t / {Tc} + {phases[len(C) // 2]:.4f})"
    print(f"\nПервинний аналоговий сигнал: {equation}")


# s(t)
def reconstruct_signal(C, N, Tc):
    t_values = np.linspace(0, Tc, N * 100, endpoint=False)
    s_t = np.zeros(len(t_values))

    for n in range(len(t_values)):
        for k in range(N // 2):
            s_t[n] += np.real(C[k] * np.exp(2j * np.pi * k * n / (N * 100)))

    print("\nТаблиця результатів розрахунку значень s(t) у 8 точках:")
    print(f"{'t/Tc':^10} {'s(t)':^10}")
    for i in range(8):
        index = i * len(t_values) // 8
        print(f"{i/8:^10.3f} {s_t[index]:^10.4f}")

    plt.plot(t_values, s_t)
    plt.title("Часова залежність відтвореного сигналу")
    plt.xlabel("Час")
    plt.ylabel("Амплітуда")
    plt.tight_layout()
    plt.show()


def part2():
    number = 112                                                            # число
    Tc = 1                                                                  # період
    signal = generate_binary_signal_from_number(number)                     # генерація сигналу
    N = len(signal)                                                         # довжина сигналу
    C, magnitudes, phases = compute_complex_fourier_coefficients(signal)    # обчислення
    display_harmonic_sum(C, magnitudes, phases, Tc)                         # s(t)
    reconstruct_signal(C, N, Tc)                                            # відновлення сигналу
    return C, N, Tc


if __name__ == "__main__":
    C, N, Tc = part2()
