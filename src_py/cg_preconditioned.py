# %% [markdown]
# # Conjugate gradients — Randomized Hadamard preconditioner
#
# Normal equations $A x = b$, $A=\hat A^\top\hat A$, $b=\hat A^\top\hat b$.
# Preconditioner $M^{-1}=\hat A^\top\Phi^\top\Phi\hat A=(\Phi\hat A)^\top(\Phi\hat A)$,
# $\Phi=R\,H^{\mathrm{norm}}_m S$.  $A$ is never formed explicitly:
# $Av=\hat A^\top(\hat A v)$ (Lecture 10, $O(mn)$ matvec).
#
# **Design of $\hat A$.** We take a *rapidly decaying* spectrum
# $\sigma_i^2(\hat A)=\max(10^{-i/10},10^{-12})$ ($i=0,\dots,n-1$): a dominant
# block of numerical rank $r\approx120$ and a degenerate tail clustered at the
# floor.  Then $\kappa(\hat A^\top\hat A)=10^{12}$ — in IEEE double the CG
# finite-termination property (Lecture 10, Theorem 1) is destroyed and plain CG
# stagnates.  The SRHT sketch has $s=n+p=420\gg r$ rows, so $\Phi$ is a near
# isometry **on the dominant subspace** (subspace embedding requirement scales
# with the effective rank, not $n$); the ridge maps the degenerate tail to a
# second tight cluster.  The preconditioned operator therefore has an essentially
# two-cluster spectrum and CG converges superlinearly (Lecture 10, Theorem 2)
# in *tens* of iterations.

# %%
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
FIG = "../figures/"

m = 2 ** 12          # 4096 observations
n = 400              # features
p = 20               # oversampling (prescribed)
s = n + p            # sketch rows

# %% [markdown]
# ### Data: prescribed rapidly-decaying spectrum, $\kappa(\hat A^\top\hat A)=10^{12}$
# $\hat A=U\,\mathrm{diag}(\sigma)\,V^\top$ with orthonormal $U\in R^{m\times n}$,
# $V\in R^{n\times n}$ and $\sigma_i^2=\max(10^{-i/10},10^{-12})$.

# %%
spectrum = np.maximum(10.0 ** (-np.arange(n) / 10.0), 1e-12)   # eigenvalues of A
U_cols, _ = np.linalg.qr(rng.standard_normal((m, n)))          # m x n orthonormal
V_rot, _ = np.linalg.qr(rng.standard_normal((n, n)))           # n x n orthonormal
A_hat = (U_cols * np.sqrt(spectrum)) @ V_rot.T
r_eff = int(np.sum(spectrum > 1e-12 * 1.001))                  # numerical rank

x_true = rng.standard_normal(n)
b_hat = A_hat @ x_true + 1e-6 * rng.standard_normal(m)
b = A_hat.T @ b_hat                                            # rhs of normal eqs


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
rows = rng.integers(0, m, size=s)                         # R: one uniform nonzero / row
inv_sqrt_m = 1.0 / np.sqrt(m)

W = np.empty((s, n))
for j in range(n):
    col = fwht(S * A_hat[:, j]) * inv_sqrt_m               # H^norm S a_j
    W[:, j] = col[rows]                                    # R (...)
Minv = W.T @ W                                             # n x n  (= M^{-1})
Minv += 1e-10 * np.trace(Minv) / n * np.eye(n)             # ridge -> clusters the tail
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
        denom = pdir @ Ap
        if denom <= 0:                       # numerical breakdown -> stop
            break
        alpha = rz / denom
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
w, Vm = np.linalg.eigh(M)
M_half = (Vm * np.sqrt(np.clip(w, 0, None))) @ Vm.T
P = M_half @ A_full @ M_half
ev_P = np.linalg.eigvalsh(P)
ev_P = ev_P[ev_P > 1e-14 * ev_P[-1]]
kappa_P = ev_P[-1] / ev_P[0]


def iters_to(hist, tol):
    idx = np.where(hist <= tol)[0]
    return int(idx[0]) if len(idx) else None


tols = [1e-4, 1e-6, 1e-8, 1e-10]
it_plain = [iters_to(h_plain, t) for t in tols]
it_prec = [iters_to(h_prec, t) for t in tols]

print(f"numerical rank r_eff (sigma^2 > floor)   = {r_eff}  (sketch rows s = {s})")
print(f"kappa(A_hat^T A_hat)                     = {kappa_A:.3e}")
print(f"kappa(M^1/2 A_hat^T A_hat M^1/2)         = {kappa_P:.3e}")
print(f"reduction factor                         = {kappa_A / kappa_P:.3e}")
print(f"plain  CG  rel.res after {ITERS} it       = {h_plain[-1]:.3e}")
print(f"precond CG rel.res after {ITERS} it       = {h_prec[-1]:.3e}")
print("iterations to reach relative residual:")
for t, a, c in zip(tols, it_plain, it_prec):
    print(f"  tol={t:.0e}: plain CG = {a if a is not None else '>400 (not reached)'},"
          f"  PCG = {c if c is not None else '>400'}")

# %% [markdown]
# ### Convergence plot + iterations-to-tolerance

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].semilogy(h_plain, label="CG (no preconditioner)")
ax[0].semilogy(h_prec, label="PCG (Hadamard randomized $M$)")
ax[0].axhline(1e-8, color="grey", ls="--", lw=0.8)
ax[0].set_xlabel("iteration $k$")
ax[0].set_ylabel(r"$\|\hat A^\top \hat b-\hat A^\top\hat A x_k\|_2/\|\hat A^\top\hat b\|_2$")
ax[0].set_title(rf"$\kappa(A)={kappa_A:.0e}$, $\kappa(M^{{1/2}}AM^{{1/2}})={kappa_P:.0f}$"
                rf"  ($\approx{kappa_A/kappa_P:.0e}\times$)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

xpos = np.arange(len(tols))
cap = ITERS + 25
pv = [v if v is not None else cap for v in it_prec]
av = [v if v is not None else cap for v in it_plain]
ax[1].bar(xpos - 0.2, av, 0.4, label="CG (no precond.)", color="#1f77b4")
ax[1].bar(xpos + 0.2, pv, 0.4, label="PCG", color="#ff7f0e")
for x, v, raw in zip(xpos - 0.2, av, it_plain):
    ax[1].text(x, v + 6, ">400" if raw is None else str(raw), ha="center", fontsize=8)
for x, v, raw in zip(xpos + 0.2, pv, it_prec):
    ax[1].text(x, v + 6, ">400" if raw is None else str(raw), ha="center", fontsize=8)
ax[1].set_xticks(xpos)
ax[1].set_xticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in tols])
ax[1].set_xlabel("target relative residual")
ax[1].set_ylabel("iterations to reach target")
ax[1].set_title("Iterations to tolerance (cap = 400)")
ax[1].set_ylim(0, cap + 20)
ax[1].grid(True, axis="y", alpha=0.3)
ax[1].legend()
plt.tight_layout()
plt.savefig(FIG + "cg_residual.png", dpi=130)
print("saved figures/cg_residual.png")
