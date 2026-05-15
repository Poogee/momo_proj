# %% [markdown]
# # Conditional gradient — projection onto the Birkhoff polytope
#
# $\min_{X\in B_n}\tfrac12\|X-Y\|_F^2$, $\nabla f(X)=X-Y$.
# LMO $\min_{S\in B_n}\langle\nabla f(X_k),S\rangle$ is linear over a polytope, so
# attained at a vertex; by Birkhoff–von Neumann the vertices of $B_n$ are the
# permutation matrices, hence the LMO is a linear assignment problem solved by the
# Hungarian algorithm (Lecture 12, Frank–Wolfe / LMO).

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

FIG = "../figures/"
rng = np.random.default_rng(1)

n = 5
Y = rng.standard_normal((n, n))


def lmo(G):
    """argmin_{S permutation} <G,S>  (linear assignment, Hungarian)."""
    r, c = linear_sum_assignment(G)        # minimises sum G[r,c]
    S = np.zeros_like(G)
    S[r, c] = 1.0
    return S


def frank_wolfe(Y, iters=200):
    X = np.eye(n)                          # feasible start (a permutation matrix)
    hist = [0.5 * np.linalg.norm(X - Y) ** 2]
    for _ in range(iters):
        G = X - Y                          # gradient
        S = lmo(G)
        D = X - S
        denom = np.linalg.norm(D, "fro") ** 2
        gamma = 0.0 if denom < 1e-15 else np.clip(np.sum((X - Y) * D) / denom, 0.0, 1.0)
        X = X + gamma * (S - X)
        hist.append(0.5 * np.linalg.norm(X - Y) ** 2)
    return X, np.array(hist)


X200, hist = frank_wolfe(Y, 200)

row_err = np.abs(X200.sum(axis=1) - 1).max()
col_err = np.abs(X200.sum(axis=0) - 1).max()
neg_err = -min(0.0, X200.min())
print(f"f(X_200)                       = {hist[-1]:.6e}")
print(f"max |row sum - 1|              = {row_err:.3e}")
print(f"max |col sum - 1|              = {col_err:.3e}")
print(f"max negative entry            = {neg_err:.3e}")
print("X_200 is doubly stochastic up to ~1e-2 (FW gives O(1/k) feasibility/optimality).")

# closed-form check: projection onto B_n has no simple closed form, compare to a
# fine FW run as the reference optimum
Xref, _ = frank_wolfe(Y, 5000)
f_ref = 0.5 * np.linalg.norm(Xref - Y) ** 2

plt.figure(figsize=(7, 4.2))
plt.semilogy(np.maximum(hist - f_ref, 1e-12))
plt.xlabel("iteration $k$"); plt.ylabel(r"$f(X_k)-f^\star$")
plt.title(rf"Frank–Wolfe on $B_{{{n}}}$ (LMO = Hungarian),  $O(1/k)$")
plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
plt.savefig(FIG + "fw_birkhoff.png", dpi=130)
plt.close()
print("saved figures/fw_birkhoff.png")
