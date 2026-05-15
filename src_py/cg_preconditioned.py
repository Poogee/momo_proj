# %% [markdown]
# # Conjugate gradients — Randomized Hadamard preconditioner
#
# Normal equations $A x = b$, $A=\hat A^\top\hat A$, $b=\hat A^\top\hat b$.
# Preconditioner $M^{-1}=\hat A^\top\Phi^\top\Phi\hat A=(\Phi\hat A)^\top(\Phi\hat A)$,
# $\Phi=R\,H^{\mathrm{norm}}_m S$.  $A$ is never formed explicitly:
# $Av=\hat A^\top(\hat A v)$ (Lecture 10, $O(mn)$ matvec).

# %%
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
FIG = "../figures/"

m = 2 ** 12          # 4096 observations
n = 400              # features
p = 20               # oversampling

# %% [markdown]
# ### Data with a controlled (large) condition number
# Column-scaled Gaussian design so that $\kappa(\hat A^\top\hat A)$ is large and
# the preconditioner has something to fix.

# %%
G = rng.standard_normal((m, n))
scales = np.logspace(0, 3, n)                 # spread the column norms -> ill-conditioned
A_hat = G * scales
x_true = rng.standard_normal(n)
b_hat = A_hat @ x_true + 1e-2 * rng.standard_normal(m)
b = A_hat.T @ b_hat                           # rhs of the normal equations


def matvec_A(v):
    """A v = Â^T (Â v) — O(mn), A never materialised."""
    return A_hat.T @ (A_hat @ v)


# %% [markdown]
# ### Fast Walsh–Hadamard transform (natural / Hadamard order), $O(m\log m)$

# %%
def fwht(a):
    """Unnormalised Hadamard matvec H_m a in O(m log m), m = 2^k."""
    a = a.astype(np.float64).copy()
    h = 1
    N = a.shape[0]
    while h < N:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h].copy()
        y = a[:, h:].copy()
        a[:, :h] = x + y
        a[:, h:] = x - y
        a = a.reshape(-1)
        h *= 2
    return a


# sanity check vs the recursive definition for small order
H8 = np.array([[1, 1], [1, -1]])
for _ in range(2):
    H8 = np.block([[H8, H8], [H8, -H8]])
v = rng.standard_normal(8)
assert np.allclose(fwht(v), H8 @ v), "FWHT mismatch"
print("FWHT verified against recursive Hadamard definition (order 8).")

# %% [markdown]
# ### Build the preconditioner  $\Phi=R\,H^{\mathrm{norm}}_m S$, $M^{-1}=(\Phi\hat A)^\top(\Phi\hat A)$

# %%
S = rng.choice([-1.0, 1.0], size=m)                       # diag(±1)
rows = rng.integers(0, m, size=n + p)                     # R: one uniform nonzero / row
inv_sqrt_m = 1.0 / np.sqrt(m)

W = np.empty((n + p, n))
for j in range(n):
    col = fwht(S * A_hat[:, j]) * inv_sqrt_m               # H^norm S a_j
    W[:, j] = col[rows]                                    # R (...)
Minv = W.T @ W                                             # n x n  (= M^{-1})
Minv += 1e-8 * np.trace(Minv) / n * np.eye(n)              # tiny ridge for PD
L = np.linalg.cholesky(Minv)


def apply_M(r):
    """z = M r  by solving M^{-1} z = r with the Cholesky factor (O(n^2))."""
    y = np.linalg.solve(L, r)
    return np.linalg.solve(L.T, y)


# %% [markdown]
# ### (Pre)conditioned conjugate gradients (Lecture 10 algorithm)

# %%
def cg(matvec, b, x0, iters, precond=None):
    x = x0.copy()
    r = b - matvec(x)
    z = precond(r) if precond else r.copy()
    pdir = z.copy()
    rz = r @ z
    nb = np.linalg.norm(b)
    hist = [np.linalg.norm(r) / nb]
    for _ in range(iters):
        Ap = matvec(pdir)
        alpha = rz / (pdir @ Ap)
        x += alpha * pdir
        r -= alpha * Ap
        hist.append(np.linalg.norm(r) / nb)
        z = precond(r) if precond else r
        rz_new = r @ z
        beta = rz_new / rz
        rz = rz_new
        pdir = z + beta * pdir
    return x, np.array(hist)


x0 = np.zeros(n)
ITERS = 400
_, h_plain = cg(matvec_A, b, x0, ITERS)
_, h_prec = cg(matvec_A, b, x0, ITERS, precond=apply_M)

# %% [markdown]
# ### Condition numbers $\kappa(\hat A^\top\hat A)$ and $\kappa(M^{1/2}\hat A^\top\hat A M^{1/2})$

# %%
A_full = A_hat.T @ A_hat
ev_A = np.linalg.eigvalsh(A_full)
kappa_A = ev_A[-1] / ev_A[0]

M = np.linalg.inv(Minv)
M = 0.5 * (M + M.T)
w, V = np.linalg.eigh(M)
M_half = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
P = M_half @ A_full @ M_half
ev_P = np.linalg.eigvalsh(P)
ev_P = ev_P[ev_P > 1e-12 * ev_P[-1]]
kappa_P = ev_P[-1] / ev_P[0]

print(f"kappa(A_hat^T A_hat)               = {kappa_A:.3e}")
print(f"kappa(M^1/2 A_hat^T A_hat M^1/2)   = {kappa_P:.3e}")
print(f"reduction factor                   = {kappa_A / kappa_P:.3e}")
print(f"plain  CG  rel.res after {ITERS} it = {h_plain[-1]:.3e}")
print(f"precond CG rel.res after {ITERS} it = {h_prec[-1]:.3e}")

# %% [markdown]
# ### Convergence plot

# %%
plt.figure(figsize=(7, 4.3))
plt.semilogy(h_plain, label="CG (no preconditioner)")
plt.semilogy(h_prec, label="PCG (Hadamard randomized $M$)")
plt.xlabel("iteration $k$")
plt.ylabel(r"$\|\hat A^\top \hat b-\hat A^\top\hat A x_k\|_2/\|\hat A^\top\hat b\|_2$")
plt.title(rf"$\kappa(A)={kappa_A:.1e}$,  $\kappa(M^{{1/2}}AM^{{1/2}})={kappa_P:.1e}$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG + "cg_residual.png", dpi=130)
print("saved figures/cg_residual.png")
