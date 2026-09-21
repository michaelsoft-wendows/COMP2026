import matplotlib.pyplot as plt
import numpy as np

N, H = 64, 8
xs2 = np.linspace(-6, 6, N).reshape(-1, 1)  # wide enough for psi -> 0 at the edges
ys2 = xs2 ** 2                              # a line cannot bend

#initialize, random weights
rng = np.random.default_rng(0)
W1 = rng.normal(size=(1, H)); b1 = np.zeros(H)     # hidden layer
W2 = rng.normal(size=(H, 1)); b2 = 0.0             # output layer
init = (W1.copy(), b1.copy(), W2.copy(), b2)       # kept for the gradient check
lr2, epochs2 = 0.2, 3000

def pot(x):
    return x**2 

def neural_net(W1, b1, W2, b2):
    h = np.tanh(xs2 @ W1 + b1)
    return h, h @ W2 + b2

def hamiltonian(W1, b1, W2, b2):
    """E[psi] = <psi|H|psi> / <psi|psi> for H = -1/2 d^2/dx^2 + 1/2 x^2."""
    x = xs2.flatten()
    dx = x[1] - x[0]

    h, psi = neural_net(W1, b1, W2, b2)
    psi = psi.flatten()
    dpsi = ((1 - h ** 2) * W1.flatten()) @ W2          # analytic d(psi)/dx
    dpsi = dpsi.flatten()

    norm = np.trapezoid(psi ** 2, dx=dx)
    kinetic = 0.5 * np.trapezoid(dpsi ** 2, dx=dx)          # 1/2 int |psi'|^2, boundary term dropped since psi -> 0
    potential = np.trapezoid(pot(x) * psi ** 2, dx=dx)

    return (kinetic + potential) / norm

def energy_grad(W1, b1, W2, b2, h=1e-6):
    """Central-difference gradient of hamiltonian() w.r.t. each parameter."""
    def E(W1, b1, W2, b2):
        return hamiltonian(W1, b1, W2, b2)

    gW1 = np.zeros_like(W1)
    for idx in np.ndindex(W1.shape):
        W1[idx] += h; Ep = E(W1, b1, W2, b2)
        W1[idx] -= 2 * h; Em = E(W1, b1, W2, b2)
        W1[idx] += h
        gW1[idx] = (Ep - Em) / (2 * h)

    gb1 = np.zeros_like(b1)
    for idx in np.ndindex(b1.shape):
        b1[idx] += h; Ep = E(W1, b1, W2, b2)
        b1[idx] -= 2 * h; Em = E(W1, b1, W2, b2)
        b1[idx] += h
        gb1[idx] = (Ep - Em) / (2 * h)

    gW2 = np.zeros_like(W2)
    for idx in np.ndindex(W2.shape):
        W2[idx] += h; Ep = E(W1, b1, W2, b2)
        W2[idx] -= 2 * h; Em = E(W1, b1, W2, b2)
        W2[idx] += h
        gW2[idx] = (Ep - Em) / (2 * h)

    gb2 = (E(W1, b1, W2, b2 + h) - E(W1, b1, W2, b2 - h)) / (2 * h)

    return gW1, gb1, gW2, gb2

# --- training loop: gradient descent on E[psi] = <psi|H|psi> / <psi|psi> ---
W1, b1, W2, b2 = init[0].copy(), init[1].copy(), init[2].copy(), init[3]
history = []
for epoch in range(epochs2):
    gW1, gb1, gW2, gb2 = energy_grad(W1, b1, W2, b2)
    W1 -= lr2 * gW1
    b1 -= lr2 * gb1
    W2 -= lr2 * gW2
    b2 -= lr2 * gb2
    history.append(hamiltonian(W1, b1, W2, b2))
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}: E = {history[-1]:.6f}")
print(f"epoch {epochs2:4d}: E = {history[-1]:.6f}  (E0 = 0.5)")

# --- plot: trained psi vs. the exact ground state, and the energy trace ---
x = xs2.flatten()
dx = x[1] - x[0]
_, psi = neural_net(W1, b1, W2, b2)
psi = psi.flatten()
psi /= np.sqrt(np.trapezoid(psi ** 2, dx=dx))          # normalize for plotting/overlap

psi_exact = np.exp(-x ** 2 / 2)
psi_exact /= np.sqrt(np.trapezoid(psi_exact ** 2, dx=dx))
if np.trapezoid(psi * psi_exact, dx=dx) < 0:           # fix overall sign ambiguity
    psi = -psi
overlap = np.trapezoid(psi * psi_exact, dx=dx)
print(f"overlap |<psi|psi_exact>| = {abs(overlap):.6f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
a1.plot(x, psi, "r--", lw=2, label="trained network")
a1.set_xlabel("x"); a1.set_ylabel("psi(x)"); a1.legend(); a1.set_title("wavefunction")

a2.plot(history)
a2.axhline(0.5, ls=":", c="k", label="E0 = 1/2")
a2.set_xlabel("epoch"); a2.set_ylabel("E"); a2.legend(); a2.set_title("training")

plt.tight_layout()
plt.show()

