# %% [markdown]
# # Newton & quasi-Newton — Hessian-Free Newton for logistic regression
#
# $f(w)=\sum_i\big[\log(1+e^{w^\top x_i})-y_i w^\top x_i\big]+\tfrac{\mu}{2}\|w\|_2^2$.
# $\nabla f(w)=X^\top(\sigma(Xw)-y)+\mu w$,
# $\nabla^2 f(w)=X^\top \mathrm{diag}(\sigma(1-\sigma))X+\mu I$,
# so $\mu I\preceq\nabla^2 f\preceq L I$ with $L=\tfrac14\lambda_{\max}(X^\top X)+\mu$
# (Lecture 11, $L$-smoothness / strong convexity).

# %%
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.sparse.linalg import cg as jax_cg

jax.config.update("jax_enable_x64", True)
FIG = "../figures/"
key = jax.random.PRNGKey(0)

m, d = 1000, 100
kx, kw = jax.random.split(key)
X = jax.random.normal(kx, (m, d))
w_star = jax.random.normal(kw, (d,))
y = (jax.nn.sigmoid(X @ w_star) > 0.5).astype(jnp.float64)

XtX = X.T @ X
L_smooth_0 = 0.25 * float(jnp.linalg.eigvalsh(XtX)[-1])  # L for mu = 0
print(f"lambda_max(X^T X) = {float(jnp.linalg.eigvalsh(XtX)[-1]):.3e}")


def make_f(mu):
    def f(w):
        z = X @ w
        # numerically stable sum log(1+e^z) - y z
        nll = jnp.sum(jax.nn.softplus(z) - y * z)
        return nll + 0.5 * mu * jnp.dot(w, w)

    return f


# %% [markdown]
# ### Optimal points (high-accuracy Newton) for the optimality gap

# %%
def newton(f, w0, iters=100, damped=False, tol=1e-13):
    g_fn = jax.grad(f)
    H_fn = jax.hessian(f)
    w = w0
    hist = [float(f(w))]
    for _ in range(iters):
        g = g_fn(w)
        H = H_fn(w)
        step = jnp.linalg.solve(H, g)
        if damped:
            t = 1.0
            fw = f(w)
            while float(f(w - t * step)) > float(fw) - 0.25 * t * float(g @ step):
                t *= 0.5
                if t < 1e-12:
                    break
            w = w - t * step
        else:
            w = w - step
        hist.append(float(f(w)))
        if jnp.linalg.norm(g) < tol:
            break
    return w, np.array(hist)


f1 = make_f(1.0)
w_opt1, _ = newton(f1, jnp.zeros(d), iters=50)
f_opt1 = float(f1(w_opt1))
print(f"mu=1: f* = {f_opt1:.10f}, ||grad|| = {float(jnp.linalg.norm(jax.grad(f1)(w_opt1))):.2e}")

# %% [markdown]
# ## (a) GD, $\mu=1$ — maximal learning rate $\approx 2/L$

# %%
mu = 1.0
f = make_f(mu)
g_fn = jax.jit(jax.grad(f))
L = L_smooth_0 + mu
lr_max = 2.0 / L
print(f"L = {L:.3e},  theoretical max GD step 2/L = {lr_max:.3e}")


def gd(f, w0, lr, iters):
    gf = jax.jit(jax.grad(f))
    w = w0
    hist = [float(f(w))]
    for _ in range(iters):
        w = w - lr * gf(w)
        hist.append(float(f(w)))
    return w, np.array(hist)


curves_a = {}
for frac, lab in [(0.5, "0.5/L"), (1.0, "1/L"), (1.9, "1.9/L"), (2.05, "2.05/L (divergent)")]:
    _, h = gd(f, jnp.zeros(d), frac / L, 300)
    curves_a[lab] = h

plt.figure(figsize=(7, 4.2))
for lab, h in curves_a.items():
    plt.semilogy(np.maximum(h - f_opt1, 1e-14), label=f"step = {lab}")
plt.xlabel("iteration"); plt.ylabel(r"$f(w_k)-f^*$")
plt.title(r"GD, $\mu=1$: stable for step $<2/L$, divergent above")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "newton_gd_mu1.png", dpi=130)
plt.close()

# %% [markdown]
# ## (b)–(c) Newton vs damped Newton, $\mu=1$ (quadratic local rate, Lecture 11)

# %%
_, h_newton1 = newton(f1, jnp.zeros(d), iters=30)
_, h_damped1 = newton(f1, jnp.zeros(d), iters=30, damped=True)

plt.figure(figsize=(7, 4.2))
plt.semilogy(np.maximum(curves_a["1/L"] - f_opt1, 1e-14), label="GD (step 1/L)")
plt.semilogy(np.maximum(h_newton1 - f_opt1, 1e-14), "o-", label="Newton (pure)")
plt.semilogy(np.maximum(h_damped1 - f_opt1, 1e-14), "s-", label="Newton (damped, backtracking)")
plt.xlabel("iteration"); plt.ylabel(r"$f(w_k)-f^*$")
plt.title(r"$\mu=1$: Newton's quadratic convergence vs GD")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "newton_mu1.png", dpi=130)
plt.close()
print(f"Newton mu=1: f-f* after 6 iters = {h_newton1[6]-f_opt1:.3e}")

# %% [markdown]
# ## (d)–(e) $\mu=0$: separable data, minimiser at infinity
#
# $y_i=\mathbf 1[\sigma(x_i^\top w^*)>0.5]$ makes the data linearly separable, so for
# $\mu=0$ the infimum $f^*=0$ is **not attained** ($\|w_k\|\to\infty$).  GD with step
# $<2/L$ decreases $f$ but only sublinearly toward $0$ (no strong convexity); pure
# Newton takes enormous steps as $\mathrm{diag}(\sigma(1-\sigma))\to0$ (Hessian
# nearly singular); the damped variant stays stable.

# %%
f0 = make_f(0.0)
L0 = L_smooth_0
_, h_gd0 = gd(f0, jnp.zeros(d), 1.0 / L0, 400)
_, h_newton0 = newton(f0, jnp.zeros(d), iters=30)
_, h_damped0 = newton(f0, jnp.zeros(d), iters=30, damped=True)

plt.figure(figsize=(7, 4.2))
plt.semilogy(np.maximum(h_gd0, 1e-12), label="GD step 1/L")
plt.semilogy(np.maximum(h_newton0, 1e-12), "o-", label="Newton (pure)")
plt.semilogy(np.maximum(h_damped0, 1e-12), "s-", label="Newton (damped)")
plt.xlabel("iteration"); plt.ylabel(r"$f(w_k)\ \ (f^*=0,\ \mathrm{not\ attained})$")
plt.title(r"$\mu=0$, separable data: infimum $0$ unreachable")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "newton_mu0.png", dpi=130)
plt.close()
print(f"mu=0: GD min f = {h_gd0.min():.3e}, damped-Newton min f = {h_damped0.min():.3e}")

# %% [markdown]
# ## (f) Newton–CG  and  (g) Hessian-Free Newton  (HVP via autograd, no Hessian stored)

# %%
def hvp(f, w, v):
    """Hessian-vector product via forward-over-reverse autodiff (no n^2 storage)."""
    return jax.jvp(jax.grad(f), (w,), (v,))[1]


def newton_cg(f, w0, iters=20, cg_tol=1e-6, hessian_free=True):
    gfn = jax.jit(jax.grad(f))
    if hessian_free:
        mv = jax.jit(lambda w, v: hvp(f, w, v))
    else:
        Hfn = jax.jit(jax.hessian(f))
    w = w0
    hist = [float(f(w))]
    for _ in range(iters):
        g = gfn(w)
        if hessian_free:
            A = lambda v: mv(w, v)
        else:
            H = Hfn(w)
            A = lambda v: H @ v
        dk, _ = jax_cg(A, -g, tol=cg_tol, maxiter=200)
        w = w + dk
        hist.append(float(f(w)))
        if float(jnp.linalg.norm(g)) < 1e-11:
            break
    return w, np.array(hist)


fb = make_f(1.0)


def timeit(fn):
    fn()  # warm-up / JIT compile
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


(_, h_full), t_full = timeit(lambda: newton(fb, jnp.zeros(d), iters=15))
(_, h_ncg), t_ncg = timeit(lambda: newton_cg(fb, jnp.zeros(d), iters=15, hessian_free=False))
(_, h_hfn), t_hfn = timeit(lambda: newton_cg(fb, jnp.zeros(d), iters=15, hessian_free=True))

print(f"explicit-Hessian Newton : t = {t_full*1e3:7.1f} ms, f-f* = {h_full[-1]-f_opt1:.2e}")
print(f"Newton-CG (dense H)     : t = {t_ncg*1e3:7.1f} ms, f-f* = {h_ncg[-1]-f_opt1:.2e}")
print(f"Hessian-Free Newton-CG  : t = {t_hfn*1e3:7.1f} ms, f-f* = {h_hfn[-1]-f_opt1:.2e}")
print(f"explicit Hessian memory : d^2 = {d*d} floats; HFN stores only O(d) vectors")

plt.figure(figsize=(7, 4.2))
plt.semilogy(np.maximum(h_full - f_opt1, 1e-14), "o-", label=f"Newton (explicit H), {t_full*1e3:.0f} ms")
plt.semilogy(np.maximum(h_ncg - f_opt1, 1e-14), "s-", label=f"Newton-CG dense H, {t_ncg*1e3:.0f} ms")
plt.semilogy(np.maximum(h_hfn - f_opt1, 1e-14), "^-", label=f"Hessian-Free NCG, {t_hfn*1e3:.0f} ms")
plt.xlabel("Newton iteration"); plt.ylabel(r"$f(w_k)-f^*$")
plt.title(r"$\mu=1$: Newton / Newton-CG / Hessian-Free")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "newton_hfn.png", dpi=130)
plt.close()
print("saved newton figures")
