# %% [markdown]
# # Subgradient methods for LASSO
#
# $f(x)=\tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$.  By Moreau–Rockafellar (Lecture 13)
# $\partial f(x)=A^\top(Ax-b)+\lambda\,\partial\|x\|_1$, where
# $[\partial\|x\|_1]_i=\mathrm{sign}(x_i)$ if $x_i\ne0$ and $\in[-1,1]$ if $x_i=0$.
# We use the canonical subgradient $g=A^\top(Ax-b)+\lambda\,\mathrm{sign}(x)$
# and compare step-size rules (Lecture 13) plus a heavy-ball variant.

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


# reference optimum via FISTA (proximal accelerated, Lecture 14) for f* / Polyak
def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def fista(iters=20000):
    x = np.zeros(n); z = x.copy(); t = 1.0
    for _ in range(iters):
        xn = soft(z - (1 / Lf) * (A.T @ (A @ z - b)), lam / Lf)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = xn + ((t - 1) / tn) * (xn - x)
        x, t = xn, tn
    return x


x_opt = fista()
f_star = f(x_opt)
print(f"reference f* (FISTA, 2e4 it) = {f_star:.8f}, "
      f"nnz(x*) = {np.sum(np.abs(x_opt) > 1e-6)}")


def run(step_rule, iters=2000, beta=0.0):
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
            a = (f(x) - f_star) / max(gn ** 2, 1e-12)
        x_new = x - a * g + beta * (x - x_prev)
        x_prev = x
        x = x_new
        best = min(best, f(x))
        hist.append(best)
    return np.array(hist)


rules = ["const", "const-length", "1/sqrt(k)", "1/k", "polyak"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for r in rules:
    ax[0].semilogy(np.maximum(run(r) - f_star, 1e-10), label=r)
ax[0].set_title("Subgradient method: step-size rules")
ax[0].set_xlabel("iteration $k$"); ax[0].set_ylabel(r"$f_k^{best}-f^\star$")
ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

for beta in [0.0, 0.5, 0.8, 0.95]:
    h = run("1/sqrt(k)", beta=beta)
    ax[1].semilogy(np.maximum(h - f_star, 1e-10), label=f"$\\beta={beta}$")
    print(f"heavy-ball beta={beta}: f_best-f* = {h[-1]-f_star:.3e}")
ax[1].set_title(r"Heavy-ball $\beta_k(x^k-x^{k-1})$ on $1/\sqrt{k}$ step")
ax[1].set_xlabel("iteration $k$"); ax[1].set_ylabel(r"$f_k^{best}-f^\star$")
ax[1].grid(True, which="both", alpha=0.3); ax[1].legend()
plt.tight_layout()
plt.savefig(FIG + "subgrad_lasso.png", dpi=130)
plt.close()
print("saved figures/subgrad_lasso.png")
