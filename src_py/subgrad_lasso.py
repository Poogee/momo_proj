# %% [markdown]
# # Subgradient methods for LASSO  (+ proximal / FISTA class reference)
#
# $f(x)=\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$.  By Moreau–Rockafellar (Lecture 13)
# $\partial f(x)=A^\top(Ax-b)+\lambda\,\partial\|x\|_1$, where
# $[\partial\|x\|_1]_i=\mathrm{sign}(x_i)$ if $x_i\ne0$ and $\in[-1,1]$ if $x_i=0$.
# Canonical subgradient $g=A^\top(Ax-b)+\lambda\,\mathrm{sign}(x)$.  We compare
# the step-size rules of Lecture 13 (+ a heavy-ball/Polyak variant) against the
# **proximal class** of Lecture 14: ISTA $O(1/k)$ and FISTA $O(1/k^2)$, to make
# the $O(1/\sqrt k)$ vs $O(1/k)$ vs $O(1/k^2)$ gap explicit.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "../figures/"
rng = np.random.default_rng(3)

n, m, lam = 1000, 200, 0.01
A = rng.standard_normal((m, n)) / np.sqrt(m)
x_true = np.zeros(n)
idx = rng.choice(n, 20, replace=False)
x_true[idx] = rng.standard_normal(20)
b = A @ x_true + 1e-2 * rng.standard_normal(m)

f = lambda x: 0.5 * np.linalg.norm(A @ x - b) ** 2 + lam * np.linalg.norm(x, 1)
subg = lambda x: A.T @ (A @ x - b) + lam * np.sign(x)
Lf = float(np.linalg.eigvalsh(A.T @ A)[-1])      # smooth-part Lipschitz constant


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


# reference optimum via FISTA (proximal accelerated, Lecture 14) for f* / Polyak
def fista(iters, track=False):
    x = np.zeros(n); z = x.copy(); t = 1.0
    hist = [f(x)]
    for _ in range(iters):
        xn = soft(z - (1 / Lf) * (A.T @ (A @ z - b)), lam / Lf)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
        if track:
            hist.append(f(x))
    return x, np.array(hist)


def ista(iters):
    x = np.zeros(n); hist = [f(x)]
    for _ in range(iters):
        x = soft(x - (1 / Lf) * (A.T @ (A @ x - b)), lam / Lf)
        hist.append(f(x))
    return np.array(hist)


x_opt, _ = fista(60000)
f_star = f(x_opt)
print(f"reference f* (FISTA, 6e4 it) = {f_star:.10f}, "
      f"nnz(x*) = {np.sum(np.abs(x_opt) > 1e-6)}")

ITERS = 20000


def run(step_rule, iters=ITERS, beta=0.0):
    x = np.zeros(n); x_prev = x.copy()
    best = f(x); hist = [best]
    for k in range(1, iters + 1):
        g = subg(x)
        gn = np.linalg.norm(g)
        if step_rule == "const":
            a = 1e-3
        elif step_rule == "const-length":
            a = 1e-2 / max(gn, 1e-12)
        elif step_rule == "1/sqrt(k)":
            a = 5e-2 / np.sqrt(k)
        elif step_rule == "1/k":
            a = 1.0 / k
        elif step_rule == "polyak":
            a = (f(x) - f_star) / max(gn ** 2, 1e-18)
        x_new = x - a * g + beta * (x - x_prev)
        x_prev = x
        x = x_new
        best = min(best, f(x))
        hist.append(best)
    return np.array(hist)


# %% [markdown]
# ## (a) Step-size rules  |  (b) heavy-ball momentum on the Polyak step

# %%
rules = ["const", "const-length", "1/sqrt(k)", "1/k", "polyak"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for r in rules:
    h = run(r)
    ax[0].loglog(np.maximum(h - f_star, 1e-12), label=r)
    print(f"{r:14s}: f_best-f* = {h[-1]-f_star:.3e}")
ax[0].set_title(r"Subgradient step-size rules ($O(1/\sqrt{k})$ class)")
ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel(r"$f_k^{best}-f^\star$")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

# best subgradient (Polyak) + heavy-ball momentum vs proximal class
h_poly = run("polyak", beta=0.0)
h_hb = run("polyak", beta=0.8)
h_ista = ista(ITERS)
_, h_fista = fista(ITERS, track=True)
print(f"Polyak               : gap = {h_poly[-1]-f_star:.3e}")
print(f"Polyak + heavy-ball  : gap = {h_hb[-1]-f_star:.3e}  (beta=0.8)")
print(f"ISTA  (prox, O(1/k)) : gap = {h_ista[-1]-f_star:.3e}")
print(f"FISTA (O(1/k^2))     : gap = {h_fista[-1]-f_star:.3e}")

kk = np.arange(1, ITERS + 1)
ax[1].loglog(np.maximum(h_poly - f_star, 1e-12), label="subgradient Polyak")
ax[1].loglog(np.maximum(h_hb - f_star, 1e-12), label=r"Polyak + heavy-ball $\beta{=}0.8$")
ax[1].loglog(np.maximum(h_ista - f_star, 1e-12), label="proximal gradient (ISTA)")
ax[1].loglog(np.maximum(h_fista - f_star, 1e-12), label="FISTA (accelerated prox)")
g0 = h_poly[1] - f_star
ax[1].loglog(kk, g0 / np.sqrt(kk), "k--", lw=0.8, alpha=0.7, label=r"$\propto 1/\sqrt{k}$")
ax[1].loglog(kk, g0 / kk, "k:", lw=0.8, alpha=0.7, label=r"$\propto 1/k$")
ax[1].loglog(kk, g0 / kk ** 2, "k-.", lw=0.8, alpha=0.7, label=r"$\propto 1/k^2$")
ax[1].set_ylim(1e-12, None)
ax[1].set_title(r"Rate classes (Lecture 14): $1/\sqrt{k}$ vs $1/k$ vs $1/k^2$")
ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel(r"$f_k-f^\star$")
ax[1].grid(True, which="both", alpha=0.3); ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG + "subgrad_lasso.png", dpi=130)
plt.close()
print("saved figures/subgrad_lasso.png")
