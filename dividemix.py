import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

@torch.no_grad()
def sharpen(p, T=0.5):
    p_pow = p ** (1.0 / T)
    return p_pow / p_pow.sum(dim=1, keepdim=True)

def mixup(x, y, alpha=4.0):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    idx = torch.randperm(x.size(0), device=x.device)
    x2, y2 = x[idx], y[idx]
    x_mix = lam * x + (1.0 - lam) * x2
    y_mix = lam * y + (1.0 - lam) * y2
    return x_mix, y_mix, lam

@torch.no_grad()
def compute_per_sample_loss(model, loader, device):
    model.eval()
    losses = np.zeros(len(loader.dataset), dtype=np.float32)

    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="none")  # per-sample
        losses[idx.numpy()] = ce.detach().cpu().numpy()
    return losses

def gmm_clean_prob(losses):
    # normalize losses to [0,1] for stable GMM fit
    lmin, lmax = losses.min(), losses.max()
    l = (losses - lmin) / (lmax - lmin + 1e-12)
    l = l.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-6)
    gmm.fit(l)

    means = gmm.means_.reshape(-1)
    clean_comp = np.argmin(means)

    prob = gmm.predict_proba(l)[:, clean_comp]  # p(clean | loss)
    return prob.astype(np.float32)

def soft_ce(logits, targets):
    # targets are soft labels (N,C)
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()

def train_dividemix_epoch(
    net, net_other,
    opt,
    labeled_loader, unlabeled_loader,
    device,
    lambda_u=1.0,
    T=0.5,
    alpha=4.0,
):
    net.train()
    net_other.eval()

    unl_it = iter(unlabeled_loader)
    total_Lx, total_Lu, nsteps = 0.0, 0.0, 0

    for x_l, y_l_soft, _ in labeled_loader:
        try:
            x_u, _, _ = next(unl_it)
        except StopIteration:
            unl_it = iter(unlabeled_loader)
            x_u, _, _ = next(unl_it)

        x_l = x_l.to(device)
        y_l_soft = y_l_soft.to(device)  # (N,C)
        x_u = x_u.to(device)

        # 1) unlabeled guess (co-guessing)
        with torch.no_grad():
            pu1 = F.softmax(net(x_u), dim=1)
            pu2 = F.softmax(net_other(x_u), dim=1)
            q_u = sharpen(0.5 * (pu1 + pu2), T=T)

        # 2) mix labeled+unlabeled
        x = torch.cat([x_l, x_u], dim=0)
        y = torch.cat([y_l_soft, q_u], dim=0)
        x_mix, y_mix, _ = mixup(x, y, alpha=alpha)

        # 3) forward
        logits = net(x_mix)
        logits_l = logits[: x_l.size(0)]
        logits_u = logits[x_l.size(0):]

        # 4) losses
        Lx = soft_ce(logits_l, y_mix[: x_l.size(0)])
        Lu = F.mse_loss(F.softmax(logits_u, dim=1), y_mix[x_l.size(0):])

        loss = Lx + lambda_u * Lu

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_Lx += float(Lx.detach().cpu())
        total_Lu += float(Lu.detach().cpu())
        nsteps += 1

    return total_Lx / max(nsteps, 1), total_Lu / max(nsteps, 1)
