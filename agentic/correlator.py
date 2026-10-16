import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# --- 1D scalar field on a periodic lattice ---
N, a = 64, 1.0                      # lattice sites, spacing
m, lam = 0.5, 0.0                   # bare mass, quartic coupling (0 = free field)

rng = np.random.default_rng(0)
phi = rng.normal(scale=0.5, size=N)  # field value at each site

def action(phi, a=a, m=m, lam=lam):
    """Euclidean action S[phi] = sum_x a * [ (phi[x+1]-phi[x])^2 / (2a^2) + m^2 phi[x]^2 / 2 + lam phi[x]^4 / 24 ]."""
    dphi = np.roll(phi, -1) - phi              # periodic forward difference
    kinetic = dphi ** 2 / (2 * a ** 2)
    mass = m ** 2 * phi ** 2 / 2
    interaction = lam * phi ** 4 / 24
    return a * np.sum(kinetic + mass + interaction)

def local_action(phi, x, val, a=a, m=m, lam=lam):
    """Contribution to S from site x alone, with phi[x] set to val (only the two links touching x matter)."""
    left, right = phi[x - 1], phi[(x + 1) % len(phi)]
    kinetic = ((val - left) ** 2 + (right - val) ** 2) / (2 * a ** 2)
    mass = m ** 2 * val ** 2 / 2
    interaction = lam * val ** 4 / 24
    return a * (kinetic + mass + interaction)

def metropolis_sweep(phi, step_size, rng):
    """One sweep of local Metropolis updates (site-by-site), done in place. Returns the acceptance rate."""
    accepted = 0
    for x in range(len(phi)):
        proposal = phi[x] + rng.uniform(-step_size, step_size)
        dS = local_action(phi, x, proposal) - local_action(phi, x, phi[x])
        if dS <= 0 or rng.uniform() < np.exp(-dS):
            phi[x] = proposal
            accepted += 1
    return accepted / len(phi)

def generate_samples(phi, n_samples, step_size=1.0, n_therm=200, n_decorrelate=10, rng=rng):
    """Run Metropolis on phi (in place): burn in for n_therm sweeps, then return an
    (n_samples, N) array of decorrelated field configurations plus the burn-in action trace."""
    burn_in_action = np.empty(n_therm)
    for i in range(n_therm):
        metropolis_sweep(phi, step_size, rng)
        burn_in_action[i] = action(phi)

    samples = np.empty((n_samples, len(phi)))
    acc_rate = 0.0
    for i in range(n_samples):
        for _ in range(n_decorrelate):
            acc_rate = metropolis_sweep(phi, step_size, rng)
        samples[i] = phi
    return samples, acc_rate, burn_in_action

def two_point_correlator(samples):
    """C(t) = <phi(x) phi(x+t)>, averaged over sites x and samples, for all lattice time separations t."""
    N = samples.shape[1]
    C = np.empty(N)
    for t in range(N):
        C[t] = np.mean(samples * np.roll(samples, -t, axis=1))
    return C

def exp_decay(t, A, m_eff):
    return A * np.exp(-m_eff * np.abs(t))

if __name__ == "__main__":
    samples, acc_rate, burn_in_action = generate_samples(phi, n_samples=2000, step_size=1.2)
    print(f"acceptance rate: {acc_rate:.2f}")
    print(f"<phi^2> = {np.mean(samples ** 2):.4f}")

    C_full = two_point_correlator(samples)          # t = 0 .. N-1
    ts = np.arange(-(N // 2), N // 2 + 1)            # symmetric -N/2 .. N/2
    C = C_full[ts % N]                               # C(-t) = C(N-t) = C(t) exactly, by translation invariance

    fit_range = np.abs(ts) <= N // 8                 # short separations only, before noise/wraparound dominate
    popt, _ = curve_fit(exp_decay, ts[fit_range], C[fit_range], p0=[C_full[0], m])
    A_fit, m_fit = popt
    print(f"fitted: A = {A_fit:.4f}, m_eff = {m_fit:.4f}  (input mass m = {m})")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4))

    ax0.plot(burn_in_action)
    ax0.set_xlabel("burn-in sweep"); ax0.set_ylabel("S[phi]"); ax0.set_title("thermalization")

    ax1.plot(ts, C, "o", label="C(t) data")
    ax1.plot(ts, exp_decay(ts, *popt), "r--", label=f"fit: A e^(-{m_fit:.3f} |t|)")
    ax1.set_xlabel("t"); ax1.set_ylabel("C(t)"); ax1.legend(); ax1.set_title("two-point correlator")

    n_show = min(len(samples), 200)                 # keep the y-axis readable
    vmax = np.abs(samples[:n_show]).max()
    im = ax2.imshow(samples[:n_show], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                     origin="lower", interpolation="nearest")
    ax2.set_xlabel("lattice site x"); ax2.set_ylabel("sample"); ax2.set_title("field configurations")
    fig.colorbar(im, ax=ax2, label="phi(x)")

    plt.tight_layout()
    plt.show()
