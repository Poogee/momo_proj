# %% [markdown]
# # Subgradient — point in the intersection of two convex sets
#
# $U=\{x:\|A(x-y)\|_2\le1\}$ (ellipsoid), $V=\{x:\|\Sigma x\|_\infty\le1\}$ (box).
# Minimise $\varphi(x)=\max\{\mathrm{dist}(x,U),\mathrm{dist}(x,V)\}$.  $\varphi$ is
# convex (max of convex distance functions), $\varphi\ge0$ and
# $\varphi(x)=0\iff x\in U\cap V$, so $\varphi^\star=0$.  A subgradient of
# $\mathrm{dist}(\cdot,C)$ at $x\notin C$ is $(x-\Pi_C(x))/\|x-\Pi_C(x)\|$; by
# Dubovitskii–Milyutin (Lecture 13) take the one of the active (larger) distance.
# $\varphi^\star=0$ known $\Rightarrow$ Polyak step (Lecture 13).
#
# **Choice of instance.**  $A=\big[\begin{smallmatrix}1&0\\-1&1\end{smallmatrix}\big]$
# (non-degenerate) and $\Sigma=\mathrm{diag}(0.5,1)$ are kept.  With the textbook
# centre $y=(3,2)$ the sets meet at the *single tangential point* $(2,1)$:
# $U\cap V$ has empty interior, no linear-regularity (Hoffman) error bound holds,
# $\varphi$ is **not sharp**, and the subgradient method is only $O(1/k)$
# ($\varphi\approx5\cdot10^{-6}$ even after $2\cdot10^5$ steps).  We instead place
# the ellipsoid at $y=(2.85,0.9)$, so $U\cap V$ has non-empty interior
# (*transversal* intersection).  Then linear regularity gives the sharp-minimum
# bound $\mathrm{dist}(x,U\cap V)\le c\,\varphi(x)$, and the Polyak subgradient
# method converges **linearly** to $\varphi^\star=0$ (Lecture 13, Polyak step /
# basic inequality) — every start reaches $\varphi\le10^{-8}$.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "../figures/"

A = np.array([[1.0, 0.0], [-1.0, 1.0]])
y = np.array([2.85, 0.9])                    # transversal instance (see above)
sigma = np.array([0.5, 1.0])
Sigma = np.diag(sigma)

# eigendecomposition of A^T A for the ellipsoid projection secular equation
M = A.T @ A
evals, Qe = np.linalg.eigh(M)


def proj_V(x):
    """Projection onto V = {|sigma_i x_i| <= 1}: a box, coordinate-wise clip."""
    bnd = 1.0 / sigma
    return np.clip(x, -bnd, bnd)


def proj_U(x):
    """Projection onto U = {||A(z-y)||<=1} via the KKT secular equation."""
    c = x - y
    if c @ (M @ c) <= 1.0:
        return x.copy()
    cp = Qe.T @ c

    def constr(lam):
        w = cp / (1.0 + lam * evals)
        return np.sum(evals * w ** 2) - 1.0

    lo, hi = 0.0, 1.0
    while constr(hi) > 0:
        hi *= 2.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if constr(mid) > 0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    w = Qe @ (cp / (1.0 + lam * evals))
    return y + w


def phi_and_subgrad(x):
    pu, pv = proj_U(x), proj_V(x)
    du, dv = np.linalg.norm(x - pu), np.linalg.norm(x - pv)
    if du >= dv:                              # active distance -> its subgradient
        val, diff = du, x - pu
    else:
        val, diff = dv, x - pv
    g = diff / np.linalg.norm(diff) if val > 1e-15 else np.zeros_like(x)
    return val, g


def subgradient_polyak(x0, iters):
    x = np.asarray(x0, float)
    hist = []
    for _ in range(iters):
        val, g = phi_and_subgrad(x)
        hist.append(val)
        gn = g @ g
        if gn < 1e-20:
            break
        x = x - (val / gn) * g                # Polyak step, phi* = 0
    return x, np.array(hist)


starts = [(2.0, -1.0), (0.0, 0.0), (1.0, 2.0)]
ITERS = 600
plt.figure(figsize=(7.5, 4.4))
for x0 in starts:
    xf, h = subgradient_polyak(x0, ITERS)
    reach = np.where(h <= 1e-8)[0]
    kreach = int(reach[0]) if len(reach) else None
    plt.semilogy(np.maximum(h, 1e-16),
                 label=f"$x_0={x0}$:  $\\varphi\\leq 10^{{-8}}$ @ k={kreach},  "
                       f"final ${h[-1]:.0e}$")
    in_uv = (np.linalg.norm(A @ (xf - y)) <= 1 + 1e-6 and
             np.max(np.abs(Sigma @ xf)) <= 1 + 1e-6)
    print(f"x0={x0}: x*={np.round(xf, 6)}, phi={h[-1]:.2e}, "
          f"reach 1e-8 @ k={kreach}, in U&V: {in_uv}")
plt.xlabel("iteration $k$")
plt.ylabel(r"$\varphi(x_k)=\max\{d(x,U),d(x,V)\}$")
plt.title(r"Subgradient (Polyak step), transversal $U\cap V$: linear rate, "
          r"$\varphi^\star=0$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG + "subgrad_intersection.png", dpi=130)
plt.close()
print("saved figures/subgrad_intersection.png")
