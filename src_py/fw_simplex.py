# %% [markdown]
# # Conditional gradient vs projected gradient — quadratic over the simplex
#
# $\min_{x\in\Delta_n}\tfrac12 x^\top Qx+c^\top x$, $\nabla f(x)=Qx+c$.
# FW-LMO over $\Delta_n$: $\min_{s\in\Delta_n}\langle\nabla f,s\rangle=e_{i^\star}$,
# $i^\star=\arg\min_i[\nabla f]_i$ (linear function on a simplex is minimised at a vertex).
# PGD uses the exact Euclidean simplex projection (Duchi et al. 2008) and step $1/L$,
# $L=\lambda_{\max}(Q)$ (Lecture 12, projected-gradient $O(1/k)$, FW $O(1/k)$).

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "../figures/"
rng = np.random.default_rng(2)

n = 20
mu, L = 1.0, 100.0                               # prescribed spectrum [mu; L]
spec = np.concatenate(([mu, L], rng.uniform(mu, L, n - 2)))
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
Q = (U * spec) @ U.T
Q = 0.5 * (Q + Q.T)

x_star = rng.dirichlet(np.ones(n))               # random point in the simplex
c = -Q @ x_star                                  # so unconstrained min = x* in Delta_n
f_star = 0.5 * x_star @ Q @ x_star + c @ x_star
grad = lambda x: Q @ x + c


def simplex_proj(v):
    """Euclidean projection onto {x>=0, sum x = 1}  (Duchi et al., 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(1, len(v) + 1)) > 0)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def frank_wolfe(x0, iters):
    x = x0.copy()
    hist = [0.5 * x @ Q @ x + c @ x]
    for _ in range(iters):
        g = grad(x)
        s = np.zeros(n)
        s[np.argmin(g)] = 1.0                     # LMO vertex
        d = s - x
        denom = d @ Q @ d
        gamma = 0.0 if denom < 1e-15 else np.clip(-(g @ d) / denom, 0.0, 1.0)
        x = x + gamma * d                         # exact line search for quadratics
        hist.append(0.5 * x @ Q @ x + c @ x)
    return np.array(hist)


def pgd(x0, iters):
    x = x0.copy()
    hist = [0.5 * x @ Q @ x + c @ x]
    for _ in range(iters):
        x = simplex_proj(x - (1.0 / L) * grad(x))
        hist.append(0.5 * x @ Q @ x + c @ x)
    return np.array(hist)


starts = {
    "vertex $e_1$": np.eye(n)[0],
    r"barycenter $\mathbf{1}/n$": np.ones(n) / n,
}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for i, (lab, x0) in enumerate(starts.items()):
    hf = frank_wolfe(x0, 400)
    hp = pgd(x0, 400)
    ax[i].semilogy(np.maximum(hf - f_star, 1e-15), label="Frank–Wolfe")
    ax[i].semilogy(np.maximum(hp - f_star, 1e-15), label="Projected GD ($1/L$)")
    ax[i].set_title(f"start: {lab}")
    ax[i].set_xlabel("iteration $k$"); ax[i].set_ylabel(r"$|f(x_k)-f(x^\star)|$")
    ax[i].grid(True, which="both", alpha=0.3); ax[i].legend()
    print(f"start {lab:24s}: FW final gap {hf[-1]-f_star:.2e}, "
          f"PGD final gap {hp[-1]-f_star:.2e}")
plt.suptitle(rf"$\kappa(Q)=L/\mu={L/mu:.0f}$:  FW $O(1/k)$ sublinear vs PGD linear")
plt.tight_layout()
plt.savefig(FIG + "fw_simplex.png", dpi=130)
plt.close()
print("saved figures/fw_simplex.png")
