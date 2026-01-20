# train.py
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from noisy_datasets import CIFARNoisy

from models import ModelConfig, build_model
from utils import (
    AverageMeter,
    accuracy_top1,
    linear_rampup,
    mixup,
    save_checkpoint,
    set_seed,
    sharpen,
    soft_cross_entropy,
)

# -------------------------
# Dataset
# -------------------------

class NpyPatchDataset(Dataset):
    """
    LIDC patch dataset for .npy files referenced by CSV columns.

    Expected CSV columns (your case):
      patient_id, case_number, nodule_number, malignancy, ...

    Filenames assumed:
      {patient_id}_{nodule_number}_case{case_number}.npy
      e.g. LIDC-IDRI-0003_4_case1.npy
    """
    def __init__(
        self,
        csv_path: str,
        patch_dir: str,
        patient_col: str = "patient_id",
        case_col: str = "case_number",
        nodule_col: str = "nodule_number",
        label_col: str = "malignancy",
        filename_pattern: str = "{patient_id}_{nodule_number}_case{case_number}.npy",
        hw: Optional[int] = None,
        to_binary: bool = True,
        drop_malignancy_3: bool = False,
        bin_threshold: float = 3.0,
        augment: bool = False,
        return_index: bool = True,
        verify_exists: bool = True,
        
    ):
        self.df = pd.read_csv(csv_path)

        for c in (patient_col, case_col, nodule_col, label_col):
            if c not in self.df.columns:
                raise ValueError(f"CSV missing column '{c}'. Columns: {list(self.df.columns)}")

        self.patch_dir = patch_dir
        self.patient_col = patient_col
        self.case_col = case_col
        self.nodule_col = nodule_col
        self.label_col = label_col
        self.filename_pattern = filename_pattern

        self.hw = hw
        self.to_binary = to_binary
        self.bin_threshold = bin_threshold
        self.augment = augment
        self.return_index = return_index

        self.labels_bin: np.ndarray

        # If binary mode and requested: drop malignancy==3
        if to_binary and drop_malignancy_3:
            # handle float/mean malignancy too: drop anything close to 3
            self.df = self.df[~np.isclose(self.df[label_col].astype(float), 3.0)].reset_index(drop=True)


        # Build filenames from pattern
        self.files: List[str] = []
        for _, row in self.df.iterrows():
            fname = self.filename_pattern.format(
                patient_id=str(row[self.patient_col]),
                case_number=int(row[self.case_col]),
                nodule_number=int(row[self.nodule_col]),
            )
            self.files.append(fname)

        self.labels_raw: np.ndarray = self.df[self.label_col].values

        if self.to_binary:
            self.labels_bin: np.ndarray = (
                self.labels_raw.astype(np.float32) >= float(self.bin_threshold)
            ).astype(np.int64)

        if verify_exists:
            # Fail fast on missing files (common issue when pattern mismatch)
            missing = []
            for f in self.files[: min(len(self.files), 2000)]:  # cap check for speed
                p = f if os.path.isabs(f) else os.path.join(self.patch_dir, f)
                if not os.path.isfile(p):
                    missing.append(f)
                    if len(missing) >= 20:
                        break
            if missing:
                example = missing[0]
                raise FileNotFoundError(
                    "Some .npy patches referenced by CSV were not found in patch_dir. "
                    f"Example missing file: {example}\n"
                    f"Expected at: {os.path.join(self.patch_dir, example)}\n"
                    "Check filename_pattern / columns / patch_dir."
                )

    def __len__(self) -> int:
        return len(self.files)

    def _load_patch(self, path: str) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1,H,W)
        elif arr.ndim == 3 and arr.shape[0] != 1:
            if arr.shape[-1] == 1:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                raise ValueError(f"Unexpected patch shape {arr.shape} for {path}")
        return arr.astype(np.float32)

    def _simple_aug(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical
        if torch.rand(1).item() < 0.3:
            x = x + 0.02 * torch.randn_like(x)
        return x

    def _label(self, v) -> int:
        if self.to_binary:
            return int(float(v) >= float(self.bin_threshold))
        v = int(round(float(v)))
        v = max(1, min(5, v))
        return v - 1

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        path = rel if os.path.isabs(rel) else os.path.join(self.patch_dir, rel)

        x = self._load_patch(path)
        if self.hw is not None:
            x = self._resize_to_hw(x, self.hw)
            if x.shape[1] != self.hw or x.shape[2] != self.hw:
                raise ValueError(f"Post-resize patch {path} shape {x.shape} != (1,{self.hw},{self.hw}).")


        x = torch.from_numpy(x)
        if self.augment:
            x = self._simple_aug(x)

        y = torch.tensor(self._label(self.labels_raw[idx]), dtype=torch.long)

        if self.return_index:
            return x, y, idx
        return x, y
    
    def _resize_to_hw(self, x: np.ndarray, hw: int) -> np.ndarray:
        # x: (1,H,W) -> (1,hw,hw)
        if x.shape[1] == hw and x.shape[2] == hw:
            return x.astype(np.float32)

        xt = torch.from_numpy(x).unsqueeze(0)  # (1,1,H,W)
        xt = F.interpolate(xt, size=(hw, hw), mode="bilinear", align_corners=False)
        return xt.squeeze(0).numpy().astype(np.float32)  # (1,hw,hw)


# -------------------------
# DivideMix steps
# -------------------------

@torch.no_grad()
def compute_per_sample_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    losses = np.zeros(len(loader.dataset), dtype=np.float32)
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y, reduction="none")
        losses[np.asarray(idx)] = ce.detach().cpu().numpy()
    return losses


from sklearn.mixture import GaussianMixture

def gmm_clean_prob(losses: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    L = losses.reshape(-1, 1)
    L = (L - L.min()) / (L.max() - L.min() + eps)
    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-6)
    gmm.fit(L)
    prob = gmm.predict_proba(L)
    means = gmm.means_.squeeze()
    clean_comp = np.argmin(means)
    return prob[:, clean_comp].astype(np.float32)


def gmm_clean_prob_classwise(
    losses: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute p(clean | loss) using class-wise GMMs.

    losses: (N,) per-sample CE loss
    labels: (N,) binary labels {0,1}
    returns: (N,) p_clean in [0,1]
    """
    assert losses.ndim == 1
    assert labels.ndim == 1
    assert set(np.unique(labels)).issubset({0, 1})

    N = len(losses)
    p_clean = np.zeros(N, dtype=np.float32)

    for c in (0, 1):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            # not enough samples → treat as clean
            p_clean[idx] = 1.0
            continue

        L = losses[idx].reshape(-1, 1)

        # Normalize losses *within class* (critical)
        L = (L - L.min()) / (L.max() - L.min() + eps)

        gmm = GaussianMixture(
            n_components=n_components,
            max_iter=100,
            tol=1e-3,
            reg_covar=1e-6,
        )
        gmm.fit(L)

        prob = gmm.predict_proba(L)

        # "clean" = component with *smaller mean loss*
        means = gmm.means_.squeeze()
        clean_comp = np.argmin(means)

        p_clean[idx] = prob[:, clean_comp]

    return p_clean



@torch.no_grad()
def build_refined_targets(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    p_clean: np.ndarray,
    num_classes: int,
    T: float,
) -> np.ndarray:
    """
    Label co-refinement:
      y_ref = p_clean * onehot(y) + (1 - p_clean) * p_model
      then sharpen
    """
    model.eval()
    refined = np.zeros((len(eval_loader.dataset), num_classes), dtype=np.float32)

    for x, y, idx in eval_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        p = F.softmax(logits, dim=1).detach().cpu().numpy()  # (N,C)

        y_oh = F.one_hot(y, num_classes=num_classes).float().cpu().numpy()
        w = p_clean[np.asarray(idx)].reshape(-1, 1)  # (N,1)
        y_ref = w * y_oh + (1.0 - w) * p

        y_ref_t = torch.from_numpy(y_ref)
        y_ref_t = sharpen(y_ref_t, T=T).cpu().numpy()
        refined[np.asarray(idx)] = y_ref_t

    return refined


class LabeledView(Dataset):
    def __init__(self, base: NpyPatchDataset, indices: np.ndarray, soft_targets: np.ndarray):
        self.base = base
        self.indices = indices.astype(np.int64)
        self.soft_targets = soft_targets  # (N,C) for full dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x, _, _ = self.base[idx]  # base returns (x,y,idx)
        y_soft = torch.from_numpy(self.soft_targets[idx]).float()
        return x, y_soft, idx


class UnlabeledView(Dataset):
    def __init__(self, base: NpyPatchDataset, indices: np.ndarray):
        self.base = base
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x, _, _ = self.base[idx]
        # dummy placeholder
        return x, torch.tensor(-1, dtype=torch.long), idx


def warmup_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("warmup_loss")
    acc_meter = AverageMeter("warmup_acc")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits.detach(), y), n=x.size(0))

    return loss_meter.avg, acc_meter.avg


def divmix_epoch(
    net: torch.nn.Module,
    net_other: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    device: torch.device,
    lambda_u: float,
    T: float,
    alpha: float,
) -> Tuple[float, float]:
    net.train()
    net_other.eval()

    Lx_meter = AverageMeter("Lx")
    Lu_meter = AverageMeter("Lu")

    unl_it = iter(unlabeled_loader)

    for x_l, y_l_soft, _ in labeled_loader:
        try:
            x_u, _, _ = next(unl_it)
        except StopIteration:
            unl_it = iter(unlabeled_loader)
            x_u, _, _ = next(unl_it)

        x_l = x_l.to(device, non_blocking=True)
        y_l_soft = y_l_soft.to(device, non_blocking=True)
        x_u = x_u.to(device, non_blocking=True)

        # Co-guessing on unlabeled: average both nets, sharpen
        with torch.no_grad():
            pu1 = F.softmax(net(x_u), dim=1)
            pu2 = F.softmax(net_other(x_u), dim=1)
            q_u = sharpen(0.5 * (pu1 + pu2), T=T)

        # MixUp across labeled + unlabeled
        x_all = torch.cat([x_l, x_u], dim=0)
        y_all = torch.cat([y_l_soft, q_u], dim=0)
        x_mix, y_mix = mixup(x_all, y_all, alpha=alpha)

        logits = net(x_mix)
        n_l = x_l.size(0)
        logits_l = logits[:n_l]
        logits_u = logits[n_l:]

        Lx = soft_cross_entropy(logits_l, y_mix[:n_l])
        Lu = F.mse_loss(F.softmax(logits_u, dim=1), y_mix[n_l:])

        loss = Lx + lambda_u * Lu

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        Lx_meter.update(Lx.item(), n=n_l)
        Lu_meter.update(Lu.item(), n=x_u.size(0))

    return Lx_meter.avg, Lu_meter.avg


@torch.no_grad()
def evaluate_with_metrics(model, loader, device, num_classes: int):
    model.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter  = AverageMeter("val_acc")

    all_probs = []
    all_y = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy_top1(logits, y), n=x.size(0))

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_y.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    # Confusion matrix + derived stats
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    # AUC
    auc = None
    if num_classes == 2:
        # use P(class=1)
        try:
            auc = float(roc_auc_score(y_true, probs[:, 1]))
        except ValueError:
            auc = None
    else:
        # multiclass one-vs-rest AUC
        try:
            auc = float(roc_auc_score(y_true, probs, multi_class="ovr"))
        except ValueError:
            auc = None

    metrics = {
        "loss": float(loss_meter.avg),
        "acc": float(acc_meter.avg),
        "bacc": float(bacc),
        "auc": auc,
        "cm": cm,
    }
    return metrics



# -------------------------
# Main
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("DivideMix for LIDC 2D npy patches")

    # Data
    p.add_argument("--csv", type=str, default=None, help="CSV with filename + malignancy (LIDC only)")
    p.add_argument("--patch_dir", type=str, default=None, help="Root folder containing .npy patches (LIDC only)")
    p.add_argument("--hw", type=int, default=None, help="Optional: assert patch size (e.g., 128)")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patient_col", type=str, default="patient_id")
    p.add_argument("--case_col", type=str, default="case_number")
    p.add_argument("--nodule_col", type=str, default="nodule_number")
    p.add_argument("--label_col", type=str, default="malignancy")

    p.add_argument("--dataset", type=str, default="lidc", choices=["lidc", "cifar10", "cifar100"])
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--noise_mode", type=str, default="sym", choices=["sym", "asym"])
    p.add_argument("--noise_ratio", type=float, default=0.5)   # r in DivideMix


    p.add_argument(
        "--filename_pattern",
        type=str,
        default="{patient_id}_{nodule_number}_case{case_number}.npy",
        help="Pattern to construct patch filename from CSV columns",
    )

    # Label mode
    p.add_argument("--binary", action="store_true", help="Binary classification (default)")
    p.add_argument("--no_binary", dest="binary", action="store_false")
    p.set_defaults(binary=True)
    p.add_argument("--bin_threshold", type=float, default=3.0, help=">= threshold => malignant (binary mode)")
    p.add_argument(
    "--drop_malignancy_3",
    action="store_true",
    help="Only in binary mode: drop samples with malignancy==3 from the dataset",)


    # Model
    p.add_argument("--model", type=str, default="smallcnn", choices=["smallcnn", "resnet18", "resnet34"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--width", type=int, default=32, help="SmallCNN width")

    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    # DivideMix hyperparams
    p.add_argument("--tau", type=float, default=0.5, help="p(clean) threshold for labeled set")
    p.add_argument("--T", type=float, default=0.5, help="Sharpen temperature")
    p.add_argument("--alpha", type=float, default=4.0, help="MixUp beta(alpha, alpha)")
    p.add_argument("--lambda_u", type=float, default=25.0, help="Max weight for unsup loss")
    p.add_argument("--lambda_u_ramp", type=int, default=10, help="Epochs to ramp up lambda_u after warmup")

    # Output
    p.add_argument("--outdir", type=str, default="./runs/dividemix_lidc")
    p.add_argument("--save_every", type=int, default=5)

    return p.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    if args.dataset in ["cifar10", "cifar100"]:
        is_c100 = (args.dataset == "cifar100")

        # CIFAR transforms (standard-ish)
        tf_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        # Train dataset uses noisy labels, eval-train uses same noisy labels but no aug
        base_train = CIFARNoisy(
            root=args.data_root,
            train=True,
            cifar100=is_c100,
            transform=tf_train,
            noise_ratio=args.noise_ratio,
            noise_mode=args.noise_mode,
            seed=args.seed,
        )
        base_eval = CIFARNoisy(
            root=args.data_root,
            train=True,
            cifar100=is_c100,
            transform=tf_test,
            noise_ratio=args.noise_ratio,
            noise_mode=args.noise_mode,
            seed=args.seed,
        )

        # For CIFAR, validate on the official test set (common for DivideMix)
        test_set = CIFARNoisy(
            root=args.data_root,
            train=False,
            cifar100=is_c100,
            transform=tf_test,
            noise_ratio=0.0,
            noise_mode="sym",
            seed=args.seed,
        )

        num_classes = 100 if is_c100 else 10

        # Use full noisy train set as train_idx
        N = len(base_eval)
        train_idx = np.arange(N, dtype=np.int64)

        # Val loader uses test set
        val_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # IMPORTANT: your evaluate_with_metrics currently prints acc/bacc/auc/confmat.
        # For CIFAR, you'll likely only want acc; auc isn't meaningful for 10-way unless you implemented multiclass AUC.
        # You can keep it if you implemented multiclass AUC; otherwise, just print acc/loss.

    else:
        # ---------------- LIDC path (your original code) ----------------
        base_eval = NpyPatchDataset(
            csv_path=args.csv,
            patch_dir=args.patch_dir,
            patient_col=args.patient_col,
            case_col=args.case_col,
            nodule_col=args.nodule_col,
            label_col=args.label_col,
            filename_pattern=args.filename_pattern,
            hw=args.hw,
            to_binary=args.binary,
            bin_threshold=args.bin_threshold,
            drop_malignancy_3=args.drop_malignancy_3,
            augment=False,
            return_index=True,
            verify_exists=True,
        )

        N = len(base_eval)
        idx_all = np.arange(N)
        rng = np.random.RandomState(args.seed)
        rng.shuffle(idx_all)
        n_val = int(round(args.val_ratio * N))
        val_idx = idx_all[:n_val]
        train_idx = idx_all[n_val:]

        base_train = NpyPatchDataset(
            csv_path=args.csv,
            patch_dir=args.patch_dir,
            patient_col=args.patient_col,
            case_col=args.case_col,
            nodule_col=args.nodule_col,
            label_col=args.label_col,
            filename_pattern=args.filename_pattern,
            hw=args.hw,
            to_binary=args.binary,
            bin_threshold=args.bin_threshold,
            drop_malignancy_3=args.drop_malignancy_3,
            augment=True,
            return_index=True,
            verify_exists=False,
        )

        num_classes = 2 if args.binary else 5

        # your existing LIDC val_loader using val_idx subset
        val_loader = DataLoader(
            Subset(base_eval, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )


    # Dataloader Helper
    def make_subset_loader(base: Dataset, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        subset = torch.utils.data.Subset(base, indices.tolist())
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    # Dataloaders
    # Warmup loaders (always over train_idx)
    warmup_loader_a = make_subset_loader(base_train, train_idx, args.batch_size, shuffle=True)
    warmup_loader_b = make_subset_loader(base_train, train_idx, args.batch_size, shuffle=True)

    # Eval-on-train loader (for per-sample loss & GMM)
    eval_train_loader = make_subset_loader(base_eval, train_idx, args.batch_size, shuffle=False)

    if args.dataset in ["cifar10", "cifar100"]:
        # CIFAR: val_loader already defined as test_set loader
        pass
    else:
        # LIDC: validation split from base_eval
        val_loader = make_subset_loader(base_eval, val_idx, args.batch_size, shuffle=False)

    # Build two models
    in_ch = 1 if args.dataset == "lidc" else 3

    cfg = ModelConfig(
        name=args.model,
        in_channels=in_ch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
        width=args.width,
        return_features=False,
    )
    netA = build_model(cfg).to(device)
    netB = build_model(cfg).to(device)


    optA = torch.optim.AdamW(netA.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optB = torch.optim.AdamW(netB.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        if epoch <= args.warmup_epochs:
            # Warmup: standard supervised on all training data
            la, acca = warmup_epoch(netA, optA, warmup_loader_a, device)
            lb, accb = warmup_epoch(netB, optB, warmup_loader_b, device)
            print(f"  Warmup A: loss={la:.4f} acc={acca:.4f}")
            print(f"  Warmup B: loss={lb:.4f} acc={accb:.4f}")

        else:
            
            # These losses arrays are sized to len(subset), but we need per full dataset indices.
            # Because eval_train_loader is a Subset, indices are re-mapped; easiest fix:
            # We'll rebuild losses over FULL base_eval then slice by train_idx.
            # This avoids subtle index mapping issues.
            full_eval_loader = DataLoader(
                base_eval,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            full_lossesA = compute_per_sample_loss(netA, full_eval_loader, device)
            full_lossesB = compute_per_sample_loss(netB, full_eval_loader, device)

            # only train part is relevant for split
            pA_full = np.zeros(N, dtype=np.float32)
            pB_full = np.zeros(N, dtype=np.float32)

            # labels aligned with base_eval indices
            if args.dataset in ["cifar10", "cifar100"]:
                if hasattr(base_eval, "targets"):
                    y_np = np.asarray(base_eval.targets, dtype=np.int64)
                elif hasattr(base_eval, "labels"):
                    y_np = np.asarray(base_eval.labels, dtype=np.int64)
                else:
                    raise AttributeError("CIFARNoisy must expose labels as .targets or .labels")

            else:
                if args.binary:
                    y_np = np.asarray(base_eval.labels_bin, dtype=np.int64)
                else:
                    # if you kept multiclass mapping in the dataset
                    y_np = np.asarray(base_eval.labels_mc, dtype=np.int64)  # or map from labels_raw if needed

            y_train = y_np[train_idx]

            if args.dataset in ["cifar10", "cifar100"]:
                pA_full[train_idx] = gmm_clean_prob(full_lossesA[train_idx])
                pB_full[train_idx] = gmm_clean_prob(full_lossesB[train_idx])
            else:
                pA_full[train_idx] = gmm_clean_prob_classwise(full_lossesA[train_idx], y_train)
                pB_full[train_idx] = gmm_clean_prob_classwise(full_lossesB[train_idx], y_train)

            # 2) Build refined targets for each net based on its own p(clean)
            refinedA = build_refined_targets(netA, full_eval_loader, device, pA_full, num_classes, T=args.T)
            refinedB = build_refined_targets(netB, full_eval_loader, device, pB_full, num_classes, T=args.T)

            # 3) Co-divide: netA uses netB's split, netB uses netA's split
            labeled_idx_for_A = train_idx[pB_full[train_idx] > args.tau]
            unlabeled_idx_for_A = train_idx[pB_full[train_idx] <= args.tau]

            labeled_idx_for_B = train_idx[pA_full[train_idx] > args.tau]
            unlabeled_idx_for_B = train_idx[pA_full[train_idx] <= args.tau]

            if args.dataset == "lidc":
                if args.binary:
                    y_dbg = (np.asarray(base_eval.labels_raw, dtype=np.float32) >= float(args.bin_threshold)).astype(np.int64)
                else:
                    y_dbg = np.round(np.asarray(base_eval.labels_raw, dtype=np.float32)).astype(np.int64)
                    y_dbg = np.clip(y_dbg, 1, 5) - 1

                def pct_pos(indices):
                    if len(indices) == 0:
                        return float("nan")
                    return float(y_dbg[indices].mean())

                print(f"  A labeled pos%={pct_pos(labeled_idx_for_A):.3f} unlabeled pos%={pct_pos(unlabeled_idx_for_A):.3f}")
                print(f"  B labeled pos%={pct_pos(labeled_idx_for_B):.3f} unlabeled pos%={pct_pos(unlabeled_idx_for_B):.3f}")


            def pct_pos(indices):
                if len(indices) == 0:
                    return float("nan")
                return float(y_np[indices].mean())

            print(
                f"  A labeled pos%={pct_pos(labeled_idx_for_A):.3f} "
                f"unlabeled pos%={pct_pos(unlabeled_idx_for_A):.3f}"
            )
            print(
                f"  B labeled pos%={pct_pos(labeled_idx_for_B):.3f} "
                f"unlabeled pos%={pct_pos(unlabeled_idx_for_B):.3f}"
            )

            print(f"  Split for A: labeled={len(labeled_idx_for_A)} unlabeled={len(unlabeled_idx_for_A)}")
            print(f"  Split for B: labeled={len(labeled_idx_for_B)} unlabeled={len(unlabeled_idx_for_B)}")

            # Edge case guard: if split collapses, relax threshold
            if len(labeled_idx_for_A) < 8 or len(labeled_idx_for_B) < 8:
                print("  [WARN] Too few labeled samples; consider lowering --tau or increasing warmup.")
            if len(unlabeled_idx_for_A) < 8 or len(unlabeled_idx_for_B) < 8:
                print("  [WARN] Too few unlabeled samples; consider raising --tau.")

            # 4) Build loaders for DivideMix epoch
            labeledA = LabeledView(base_train, labeled_idx_for_A, refinedA)
            unlabeledA = UnlabeledView(base_train, unlabeled_idx_for_A)

            labeledB = LabeledView(base_train, labeled_idx_for_B, refinedB)
            unlabeledB = UnlabeledView(base_train, unlabeled_idx_for_B)

            labeled_loader_A = DataLoader(
                labeledA,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            unlabeled_loader_A = DataLoader(
                unlabeledA,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            labeled_loader_B = DataLoader(
                labeledB,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            unlabeled_loader_B = DataLoader(
                unlabeledB,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )

            # 5) Ramp up lambda_u after warmup
            ramp_epoch = epoch - args.warmup_epochs
            w_u = args.lambda_u * linear_rampup(ramp_epoch, args.lambda_u_ramp)

            # 6) DivideMix epoch (co-training)
            LxA, LuA = divmix_epoch(netA, netB, optA, labeled_loader_A, unlabeled_loader_A, device,
                                   lambda_u=w_u, T=args.T, alpha=args.alpha)
            LxB, LuB = divmix_epoch(netB, netA, optB, labeled_loader_B, unlabeled_loader_B, device,
                                   lambda_u=w_u, T=args.T, alpha=args.alpha)

            print(f"  DivMix A: Lx={LxA:.4f} Lu={LuA:.4f} (lambda_u={w_u:.2f})")
            print(f"  DivMix B: Lx={LxB:.4f} Lu={LuB:.4f} (lambda_u={w_u:.2f})")

            # -------------------------
            # Validation (every epoch)
            # -------------------------
            mA = evaluate_with_metrics(netA, val_loader, device, num_classes=num_classes)
            mB = evaluate_with_metrics(netB, val_loader, device, num_classes=num_classes)

            if args.dataset in ["cifar10", "cifar100"]:
                # CIFAR reporting: loss + acc (paper-style)
                print(f"  Test A: loss={mA['loss']:.4f} acc={mA['acc']:.4f}")
                print(f"  Test B: loss={mB['loss']:.4f} acc={mB['acc']:.4f}")
                print(f"  ConfMat A:\n{mA['cm']}")
                print(f"  ConfMat B:\n{mB['cm']}")
            else:
                # LIDC reporting
                print(f"  Val A: loss={mA['loss']:.4f} acc={mA['acc']:.4f} bacc={mA['bacc']:.4f} auc={mA['auc']}")
                print(f"  ConfMat A:\n{mA['cm']}")
                print(f"  Val B: loss={mB['loss']:.4f} acc={mB['acc']:.4f} bacc={mB['bacc']:.4f} auc={mB['auc']}")
                print(f"  ConfMat B:\n{mB['cm']}")

            # -------------------------
            # Pick best metric for checkpointing
            # -------------------------
            if args.dataset in ["cifar10", "cifar100"]:
                # Paper-style: best by test accuracy
                scoreA = float(mA["acc"])
                scoreB = float(mB["acc"])
                current_best = max(scoreA, scoreB)
                best_name = "best_test_acc"
            else:
                # LIDC: best by AUC if available, else fallback to acc
                scoreA = float(mA["auc"]) if mA["auc"] is not None else float(mA["acc"])
                scoreB = float(mB["auc"]) if mB["auc"] is not None else float(mB["acc"])
                current_best = max(scoreA, scoreB)
                best_name = "best_val_auc" if (mA["auc"] is not None or mB["auc"] is not None) else "best_val_acc"

            # -------------------------
            # Best checkpoint
            # -------------------------
            if current_best > best_val:
                best_val = current_best
                save_checkpoint(
                    os.path.join(args.outdir, "best.pt"),
                    epoch=epoch,
                    model_a=netA,
                    model_b=netB,
                    opt_a=optA,
                    opt_b=optB,
                    extra={best_name: best_val, "args": vars(args)},
                )

            # -------------------------
            # Periodic checkpoint
            # -------------------------
            if (epoch % args.save_every) == 0 or epoch == args.epochs:
                save_checkpoint(
                    os.path.join(args.outdir, f"epoch_{epoch:03d}.pt"),
                    epoch=epoch,
                    model_a=netA,
                    model_b=netB,
                    opt_a=optA,
                    opt_b=optB,
                    extra={"best_score": best_val, "best_metric": best_name, "args": vars(args)},
                )

    print(f"\nDone. Best score: {best_val:.4f}")
    print(f"Saved to: {args.outdir}")

if __name__ == "__main__":
    main()
