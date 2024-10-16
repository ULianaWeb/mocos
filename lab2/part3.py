import numpy as np


# s(nTδ)
def compute_samples_from_coefficients(C, N, Tc):
    t_values = np.linspace(0, Tc, N, endpoint=False)[:8]
    s_values = np.zeros(8)

    for n in range(8):
        s_values[n] = np.real(sum(C[k] * np.exp(2j * np.pi * k * n / N) for k in range(N // 2)))

    print("\nЗначення відліків s(nTδ) для n = 0...7:")
    for i in range(8):
        print(f"s({i}Tδ) = {s_values[i]:.4f}")

    return t_values, s_values


# аналітичні вирази s(0) і s(Tδ)
def compute_analytical_samples(C, N, Tc):

    # формули
    formula_s0 = "s(0) = Σ(C_k * exp(2j * π * k * 0 / N)) for k=0 to N-1"
    formula_s1 = "s(Tδ) = Σ(C_k * exp(2j * π * k * 1 / N)) for k=0 to N-1"

    # s(0)
    print(f"\nАналітичний вираз для s(0):")
    print(formula_s0)
    s_0_parts = [f"C[{k}] * exp(2j * π * {k} * 0 / {N})" for k in range(N)]
    s_0_expr = " + ".join(s_0_parts)
    print(f"= {s_0_expr}")
    # s_0 = np.real(sum(C[k] * np.exp(2j * np.pi * k * 0 / N) for k in range(N)))
    s_0 = np.real(sum(C[k] * np.exp(2j * np.pi * k * 0 / N) for k in range(N // 2)))
    print(f"= {s_0:.4f}")

    # s(Tδ)
    print(f"\nАналітичний вираз для s(Tδ):")
    print(formula_s1)
    s_1_parts = [f"C[{k}] * exp(2j * π * {k} * 1 / {N})" for k in range(N)]
    s_1_expr = " + ".join(s_1_parts)
    print(f"= {s_1_expr}")
    #s_1 = np.real(sum(C[k] * np.exp(2j * np.pi * k * 1 / N) for k in range(N)))
    s_1 = np.real(sum(C[k] * np.exp(2j * np.pi * k * 1 / N) for k in range(N // 2)))
    print(f"= {s_1:.4f}")


def part3(C, N, Tc):
    print(f"\n\n     3 частина")
    t_values, s_values = compute_samples_from_coefficients(C, N, Tc)
    compute_analytical_samples(C, N, Tc)


if __name__ == "__main__":
    from part2 import part2
    C, N, Tc = part2()
    part3(C, N, Tc)
