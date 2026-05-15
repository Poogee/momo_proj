# %% [markdown]
# # Proximal gradient — sparse softmax (multinomial logistic) regression
#
# $\varphi(W)=\underbrace{-\sum_i\log P(y_i|x_i;W)}_{f(W)}+\lambda\|W\|_1$,
# $P(y=j|x;W)=\mathrm{softmax}(Wx)_j$.
# $\nabla f(W)=(P-Y)^\top X$ (rows $=$ classes).  Subgradient method:
# $W_{k+1}=W_k-\alpha(\nabla f(W_k)+\lambda\,\mathrm{sign}(W_k))$.
# Proximal gradient (ISTA, Lecture 14):
# $W_{k+1}=\mathcal S_{\alpha\lambda}\!\big(W_k-\alpha\nabla f(W_k)\big)$,
# $\mathcal S_\tau(v)=\mathrm{sign}(v)[\,|v|-\tau\,]_+$ (soft-threshold = $\mathrm{prox}_{\tau\|\cdot\|_1}$).
# Stopping by the gradient mapping $\|G_\alpha(W_k)\|\le\varepsilon$ ($=0\iff$ optimal).

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = "../figures/"
rng = np.random.default_rng(4)

df = pd.read_csv("../data/students.csv", sep=";")
ycol = df.columns[-1]
classes = sorted(df[ycol].unique())
cls_idx = {c: i for i, c in enumerate(classes)}
Yc = df[ycol].map(cls_idx).to_numpy()
Xraw = df.drop(columns=[ycol]).to_numpy(dtype=float)

# standardise + bias column (bias is NOT L1-penalised)
Xs = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-12)
Xall = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
N, d = Xall.shape
C = len(classes)
perm = rng.permutation(N)
ntr = int(0.8 * N)
tr, te = perm[:ntr], perm[ntr:]
Xtr, ytr, Xte, yte = Xall[tr], Yc[tr], Xall[te], Yc[te]
Ytr = np.eye(C)[ytr]
print(f"N={N}, d={d} (incl. bias), C={C}, classes={classes}")

pen = np.ones((C, d)); pen[:, -1] = 0.0          # do not penalise the bias


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def nll(W):
    P = softmax(Xtr @ W.T)
    return -np.sum(np.log(P[np.arange(ntr), ytr] + 1e-15))


def grad_nll(W):
    P = softmax(Xtr @ W.T)
    return (P - Ytr).T @ Xtr


def phi(W, lam):
    return nll(W) + lam * np.sum(np.abs(pen * W))


L = 0.5 * float(np.linalg.eigvalsh(Xtr.T @ Xtr)[-1])   # softmax-CE smoothness bound
alpha = 1.0 / L
print(f"L (smoothness bound) = {L:.3e},  prox/GD step 1/L = {alpha:.3e}")


def soft(V, t):
    return np.sign(V) * np.maximum(np.abs(V) - t, 0.0)


def test_acc(W):
    return float(np.mean(softmax(Xte @ W.T).argmax(1) == yte))


def prox_grad(lam, iters, W0=None, track=False):
    W = np.zeros((C, d)) if W0 is None else W0.copy()
    hist = [phi(W, lam)]
    for _ in range(iters):
        Wn = W - alpha * grad_nll(W)
        Wn = soft(Wn, alpha * lam * pen)              # bias: pen=0 -> untouched
        gmap = np.linalg.norm(Wn - W) / alpha
        W = Wn
        if track:
            hist.append(phi(W, lam))
        if gmap < 1e-9:
            break
    return W, np.array(hist)


def subgrad(lam, iters, track=False):
    W = np.zeros((C, d))
    best = phi(W, lam); hist = [best]
    for k in range(1, iters + 1):
        g = grad_nll(W) + lam * np.sign(W) * pen
        W = W - (alpha / np.sqrt(k)) * g
        best = min(best, phi(W, lam))
        if track:
            hist.append(best)
    return W, np.array(hist)


# %% [markdown]
# ## (b) lambda = 0 : maximal learning rate, convergence and sparsity

# %%
W0_pg, h0_pg = prox_grad(0.0, 4000, track=True)
W0_sg, h0_sg = subgrad(0.0, 4000, track=True)
f0_star = min(h0_pg.min(), h0_sg.min())
print(f"lambda=0: prox/GD step 1/L={alpha:.2e} (>2/L diverges); "
      f"final NLL pg={h0_pg[-1]:.4f}, sg={h0_sg[-1]:.4f}")
print(f"lambda=0 sparsity: pg zeros={np.mean(np.abs(W0_pg)<1e-8):.1%} "
      f"(=0, no L1), sg zeros={np.mean(np.abs(W0_sg)<1e-8):.1%}")
print(f"lambda=0 test accuracy pg={test_acc(W0_pg):.4f}, sg={test_acc(W0_sg):.4f}")

plt.figure(figsize=(7, 4.2))
plt.semilogy(np.maximum(h0_pg - f0_star, 1e-10), label="proximal gradient (= GD, $\\lambda$=0)")
plt.semilogy(np.maximum(h0_sg - f0_star, 1e-10), label=r"subgradient ($\alpha_k\propto1/\sqrt{k}$)")
plt.xlabel("iteration $k$"); plt.ylabel(r"$\varphi(W_k)-\varphi^\star$")
plt.title(r"$\lambda=0$: smooth GD ($O(1/k)$) vs subgradient ($O(1/\sqrt{k})$)")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "prox_lambda0.png", dpi=130)
plt.close()

# %% [markdown]
# ## (c) convergence table:  iterations / sparsity / test accuracy
# Plain proximal gradient (ISTA), stop at relative objective gap
# $(\varphi(W_k)-\varphi^\star)/(\varphi(W_0)-\varphi^\star)\le\varepsilon$.
# Reference $\varphi^\star$ per $\lambda$ from FISTA (accelerated proximal, Lecture 14).

# %%
def fista_ref(lam, iters=40000):
    W = np.zeros((C, d)); Z = W.copy(); t = 1.0
    for _ in range(iters):
        Wn = soft(Z - alpha * grad_nll(Z), alpha * lam * pen)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Z = Wn + ((t - 1) / tn) * (Wn - W)
        W, t = Wn, tn
    return phi(W, lam)


lams = [1e-3, 1e-2, 1e-1, 1.0]
eps_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
MAXIT = 200000
CHECK = 10

rows = []
for lam in lams:
    phi_star = fista_ref(lam)
    W = np.zeros((C, d))
    phi0 = phi(W, lam)
    target = iter(eps_list)
    eps = next(target)
    for k in range(1, MAXIT + 1):
        W = soft(W - alpha * grad_nll(W), alpha * lam * pen)
        if k % CHECK == 0:
            rel = (phi(W, lam) - phi_star) / (phi0 - phi_star)
            while rel <= eps:
                spars = np.mean(np.abs(pen * W) < 1e-8)
                rows.append((lam, eps, k, spars, test_acc(W)))
                try:
                    eps = next(target)
                except StopIteration:
                    eps = -1
                    break
            if eps < 0:
                break
    print(f"lambda={lam:g}: phi*={phi_star:.4f}, "
          f"reached eps>={eps if eps>0 else eps_list[-1]:.0e}")

tab = pd.DataFrame(rows, columns=["lambda", "eps", "iters", "sparsity", "test_acc"])
print(tab.to_string(index=False,
                     formatters={"lambda": "{:.0e}".format, "eps": "{:.0e}".format,
                                 "sparsity": "{:.2%}".format, "test_acc": "{:.4f}".format}))
tab.to_csv("../figures/prox_table.csv", index=False)

# objective curves per lambda
plt.figure(figsize=(7, 4.2))
for lam in lams:
    Wl, hl = prox_grad(lam, 4000, track=True)
    fl = hl.min()
    plt.semilogy(np.maximum(hl - fl, 1e-10), label=f"$\\lambda={lam:g}$")
plt.xlabel("iteration $k$"); plt.ylabel(r"$\varphi(W_k)-\varphi^\star$")
plt.title(r"Proximal gradient (ISTA), sparse softmax — $O(1/k)$")
plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(FIG + "prox_lambdas.png", dpi=130)
plt.close()
print("saved prox figures + table")
