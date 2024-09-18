import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# 1. Обчислення f(x)
def f(x):
    return 4 * x * np.exp(-x ** 2 / 4)


# 2. Обчислення a_k та b_k
def a_k(k):
    if k == 0:
        result, _ = quad(lambda x: f(x), -np.pi, np.pi)
        return result / (2 * np.pi)
    else:
        result, _ = quad(lambda x: f(x) * np.cos(k * x), -np.pi, np.pi)
        return result / np.pi

def b_k(k):
    result, _ = quad(lambda x: f(x) * np.sin(k * x), -np.pi, np.pi)
    return result / np.pi


# 3. Наближення рядом Фур'є
def fourier_approximation(x, N):
    sum_approx = a_k(0) / 2  # Початкове значення (a_0 / 2)
    for k in range(1, N + 1):
        sum_approx += a_k(k) * np.cos(k * x) + b_k(k) * np.sin(k * x)
    return sum_approx


# 4. Побудова графіків гармонік
def plot_harmonics(N):
    x_values = np.linspace(-np.pi, np.pi, 1000)
    plt.figure(figsize=(10, 8))

    for k in range(1, N + 1):
        harmonic = a_k(k) * np.cos(k * x_values) + b_k(k) * np.sin(k * x_values)
        plt.plot(x_values, harmonic, label=f'Гармоніка k={k}')

    plt.title('Гармоніки ряду Фур\'є')
    plt.xlabel('x')
    plt.ylabel('Амплітуда')
    plt.legend()
    plt.grid(True)
    plt.show()


# 5. Оцінка відносної похибки
def relative_error(N):
    exact_values = [f(x) for x in np.linspace(-np.pi, np.pi, 100)]
    approx_values = [fourier_approximation(x, N) for x in np.linspace(-np.pi, np.pi, 100)]
    error = np.mean(np.abs((np.array(exact_values) - np.array(approx_values)) / np.array(exact_values)))
    return error


# 6. Збереження результатів у файл
def save_results(N, filename='fourier_results.txt'):
    with open(filename, 'w') as file:
        file.write(f"Order N: {N}\n")
        file.write("Fourier coefficients:\n")
        for k in range(N + 1):
            file.write(f"a_{k} = {a_k(k)}\n")
        for k in range(1, N + 1):
            file.write(f"b_{k} = {b_k(k)}\n")
        error = relative_error(N)
        file.write(f"Approximation error: {error}\n")


# 7. Виведення ряду Фур'є
def print_fourier_series(N):
    series = f"{a_k(0) / 2}"
    for k in range(1, N + 1):
        a = a_k(k)
        b = b_k(k)
        if a != 0:
            series += f" + {a}*cos({k}*x)"
        if b != 0:
            series += f" + {b}*sin({k}*x)"
    print(f"Розкладений ряд Фур'є для порядку N={N}:\n{series}")

# 8. Графіки
def plot_approximations(N):
    # (-π, π)
    x_values_1 = np.linspace(-np.pi, np.pi, 1000)
    f_values_1 = [f(x) for x in x_values_1]
    approx_values_1 = [fourier_approximation(x, N) for x in x_values_1]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_values_1, f_values_1, label='f(x)', color='blue')
    plt.plot(x_values_1, approx_values_1, label=f'Наближення (N={N})', linestyle='--', color='red')
    plt.title('Наближення ряду Фур\'є на інтервалі (-π, π)')
    plt.legend()
    plt.grid(True)

    # (-3, -2)
    x_values_2 = np.linspace(-3, -2, 1000)
    f_values_2 = [f(x) for x in x_values_2]
    approx_values_2 = [fourier_approximation(x, N) for x in x_values_2]

    plt.subplot(1, 2, 2)
    plt.plot(x_values_2, f_values_2, label='f(x)', color='blue')
    plt.plot(x_values_2, approx_values_2, label=f'Наближення (N={N})', linestyle='--', color='red')
    plt.title('Наближення ряду Фур\'є на інтервалі (-3, -2)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# 9. Мейн
def main():
    N = 50  # Порядок ряду

    plot_approximations(N)
    plot_harmonics(N)
    print_fourier_series(N)
    error = relative_error(N)
    print(f"Відносна похибка: {error}")
    save_results(N)

if __name__ == "__main__":
    main()
