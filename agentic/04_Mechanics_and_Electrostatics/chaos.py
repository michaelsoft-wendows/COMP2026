import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def exponential_growth_mask(separation, delta, floor_factor=10.0, ceiling_frac=0.3):
    """Boolean mask over the longest contiguous stretch of `separation`
    that sits between the noise floor set by the initial perturbation
    `delta` and the ceiling where it saturates at the attractor's size --
    the regime where growth is exponential.
    """
    candidate = (separation > floor_factor * delta) & (separation < ceiling_frac * separation.max())

    mask = np.zeros_like(candidate)
    if not candidate.any():
        return mask

    # Find contiguous runs of `candidate` and keep only the longest one.
    padded = np.concatenate(([False], candidate, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    longest = np.argmax(ends - starts)
    mask[starts[longest]:ends[longest]] = True
    return mask


delta = 1e-5

t_span = (0, 50)
t_eval = np.linspace(*t_span, 20000)
initial_state_a = [1.0, 1.0, 1.0]
initial_state_b = [1.0 + delta, 1.0, 1.0]

sol_a = solve_ivp(lorenz, t_span, initial_state_a, t_eval=t_eval)
sol_b = solve_ivp(lorenz, t_span, initial_state_b, t_eval=t_eval)
xa, ya, za = sol_a.y
xb, yb, zb = sol_b.y

separation = np.sqrt((xb - xa) ** 2 + (yb - ya) ** 2 + (zb - za) ** 2)

fit_mask = exponential_growth_mask(separation, delta)
slope, intercept = np.polyfit(t_eval[fit_mask], np.log(separation[fit_mask]), 1)
lyapunov_exponent = slope

print(f"Estimated Lyapunov exponent (delta_x = {delta:.0e}): {lyapunov_exponent:.4f}")


def lorenz_jacobian(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    return np.array(
        [
            [-sigma, sigma, 0.0],
            [rho - z, -1.0, -x],
            [y, x, -beta],
        ]
    )


def lorenz_variational(t, Y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    state = Y[:3]
    M = Y[3:].reshape(3, 3)
    dstate = lorenz(t, state, sigma, rho, beta)
    dM = lorenz_jacobian(state, sigma, rho, beta) @ M
    return np.concatenate([dstate, dM.flatten()])


def lyapunov_spectrum(initial_state, sigma=10.0, rho=28.0, beta=8.0 / 3.0, renorm_interval=0.5, n_renorms=400):
    """Full Lyapunov spectrum via the Benettin et al. method: integrate the
    trajectory together with an orthonormal basis of tangent vectors,
    periodically re-orthonormalizing (QR) and accumulating the log growth
    of each basis vector.
    """
    state = np.asarray(initial_state, dtype=float)
    Q = np.eye(3)
    log_growth = np.zeros(3)

    t = 0.0
    for _ in range(n_renorms):
        Y0 = np.concatenate([state, Q.flatten()])
        sol = solve_ivp(
            lorenz_variational, (t, t + renorm_interval), Y0, args=(sigma, rho, beta),
            rtol=1e-9, atol=1e-9,
        )
        Yend = sol.y[:, -1]
        state = Yend[:3]
        M = Yend[3:].reshape(3, 3)

        Q, R = np.linalg.qr(M)
        log_growth += np.log(np.abs(np.diag(R)))
        t += renorm_interval

    return log_growth / (n_renorms * renorm_interval), state, Q


# Average the spectrum over ~10 initial conditions sampled from points
# along the attractor (after discarding the initial transient), so the
# estimate isn't tied to a single starting point.
n_initial_conditions = 10
transient_cutoff = len(xa) // 4
sample_indices = np.linspace(transient_cutoff, len(xa) - 1, n_initial_conditions, dtype=int)

spectra = []
for idx in sample_indices:
    ic = [xa[idx], ya[idx], za[idx]]
    spectrum_i, final_state, flow_directions = lyapunov_spectrum(ic)
    spectra.append(sorted(spectrum_i, reverse=True))

spectra = np.array(spectra)
spectrum_mean = spectra.mean(axis=0)
spectrum_std = spectra.std(axis=0)

print(f"Lyapunov spectrum averaged over {n_initial_conditions} initial conditions:")
for i, (mean, std) in enumerate(zip(spectrum_mean, spectrum_std), start=1):
    print(f"  lambda_{i} = {mean:.4f} +/- {std:.4f}")
print(
    f"  sum = {spectrum_mean.sum():.4f} "
    f"(expected -(sigma + 1 + beta) = {-(10.0 + 1.0 + 8.0 / 3.0):.4f})"
)

# Plot the attractor together with the Lyapunov directions (the columns of
# the final orthonormal frame Q) anchored at the point they were computed
# at, each one labeled by the exponent it corresponds to.
fig3d = plt.figure(figsize=(8, 7))
ax3d = fig3d.add_subplot(projection="3d")
ax3d.plot(xa, ya, za, linewidth=0.3, color="gray", alpha=0.6)
ax3d.scatter(*final_state, color="black", s=20)

arrow_length = 10.0
colors = ["tab:red", "tab:green", "tab:blue"]
for i in range(3):
    direction = flow_directions[:, i]
    ax3d.quiver(
        *final_state, *(arrow_length * direction),
        color=colors[i], linewidth=2,
        label=f"lambda = {spectrum_i[i]:.3f}",
    )

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("Lyapunov directions on the Lorenz attractor")
ax3d.legend()
fig3d.tight_layout()
plt.show()
