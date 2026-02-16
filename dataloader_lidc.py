# dataloader_lidc.py
# Drop-in "cifar_dataloader"-style loader for LIDC .npy patches and CSV labels.

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _ensure_chw_float(arr: np.ndarray, path: str) -> np.ndarray:
    """Ensure arr is float32 CHW with C=1."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        # allow HWC with last channel==1
        if arr.shape[-1] == 1 and arr.shape[0] != 1:
            arr = np.transpose(arr, (2, 0, 1))
        if arr.shape[0] != 1:
            raise ValueError(f"Expected (1,H,W) or (H,W) or (H,W,1), got {arr.shape} for {path}")
    else:
        raise ValueError(f"Unexpected npy shape {arr.shape} for {path}")
    return arr


def _resize_chw(x: torch.Tensor, hw: int) -> torch.Tensor:
    """
    True resize (interpolation) to (1, hw, hw).
    x: (1,H,W) float tensor
    """
    # x -> (N=1,C=1,H,W)
    x4 = x.unsqueeze(0)
    x4 = F.interpolate(x4, size=(hw, hw), mode="bilinear", align_corners=False)
    return x4.squeeze(0)


class LIDCPatchDataset(Dataset):
    """
    mode signatures (must match DivideMix expectations):
      - mode="all"      -> (img, target, index)
      - mode="labeled"  -> (img1, img2, target, prob)
      - mode="unlabeled"-> (img1, img2)
      - mode="test"     -> (img, target)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        data_path: str,
        mode: str,
        hw: int = 32,
        pred: np.ndarray | None = None,
        prob: np.ndarray | None = None,
        num_class: int = 5,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.mode = mode
        self.hw = int(hw)
        self.num_class = int(num_class)

        # map malignancy 1..5 -> 0..4
        if "malignancy" not in self.df.columns:
            raise ValueError("CSV must contain column: malignancy")

        mal = self.df["malignancy"].astype(int).to_numpy()
        if mal.min() < 1 or mal.max() > 5:
            raise ValueError(f"Expected malignancy in [1,5], got min={mal.min()}, max={mal.max()}")
        self.targets = (mal - 1).astype(np.int64)

        # build filenames: "{patient_id}_{nodule_number}_case{case_number}.npy"
        for col in ["patient_id", "case_number", "nodule_number"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV must contain column: {col}")

        self.paths = []
        for i in range(len(self.df)):
            pid = str(self.df.loc[i, "patient_id"])
            case = int(self.df.loc[i, "case_number"])
            nod = int(self.df.loc[i, "nodule_number"])
            fname = f"{pid}_{nod}_case{case}.npy"
            fpath = os.path.join(self.data_path, fname)
            self.paths.append(fpath)

        # For labeled/unlabeled split (co-divide)
        self.pred = pred
        self.prob = prob

        # RNG for deterministic augmentations if you want
        self.rng = np.random.RandomState(seed)

        if self.mode in ("labeled", "unlabeled"):
            if self.pred is None or self.prob is None:
                raise ValueError("pred and prob must be provided for labeled/unlabeled modes")
            self.pred = np.asarray(self.pred).astype(bool)
            self.prob = np.asarray(self.prob).astype(np.float32)
            if len(self.pred) != len(self.df) or len(self.prob) != len(self.df):
                raise ValueError("pred/prob must match dataset length")

            if self.mode == "labeled":
                self.indices = np.where(self.pred)[0]
            else:
                self.indices = np.where(~self.pred)[0]
        else:
            self.indices = np.arange(len(self.df))

    def __len__(self):
        return len(self.indices)

    def _load_tensor(self, idx0: int) -> torch.Tensor:
        path = self.paths[idx0]
        arr = np.load(path)
        arr = _ensure_chw_float(arr, path)
        x = torch.from_numpy(arr).float().contiguous()
        x = _resize_chw(x, self.hw)
        # Optional: normalize. For now, assume patches are already scaled sensibly.
        return x

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Keep augmentation minimal and domain-safe.
        DivideMix CIFAR uses random crop/flip.
        Here: small random horizontal flip (optional) + small noise.
        """
        # random horizontal flip
        if self.rng.rand() < 0.5:
            x = torch.flip(x, dims=[2])  # flip W

        # light gaussian noise
        if self.rng.rand() < 0.2:
            x = x + 0.01 * torch.randn_like(x)
        return x

    def __getitem__(self, i):
        idx0 = int(self.indices[i])
        y = int(self.targets[idx0])

        if self.mode == "test":
            x = self._load_tensor(idx0)
            return x, y

        if self.mode == "all":
            x = self._load_tensor(idx0)
            return x, y, idx0

        if self.mode == "labeled":
            x = self._load_tensor(idx0)
            x1 = self._augment(x.clone())
            x2 = self._augment(x.clone())
            p = float(self.prob[idx0])
            return x1, x2, y, p

        if self.mode == "unlabeled":
            x = self._load_tensor(idx0)
            x1 = self._augment(x.clone())
            x2 = self._augment(x.clone())
            return x1, x2

        raise ValueError(f"Unknown mode: {self.mode}")


class lidc_dataloader:
    """
    CIFAR loader-compatible wrapper:
      loader.run('warmup') -> loader for warmup (img, label, index)
      loader.run('test') -> (img, label)
      loader.run('eval_train') -> (img, label, index)
      loader.run('train', pred, prob) -> (labeled_loader, unlabeled_loader)
    """
    def __init__(
        self,
        csv_path: str,
        data_path: str,
        batch_size: int,
        num_workers: int,
        test_ratio: float = 0.2,
        split_seed: int = 123,
        hw: int = 32,
        num_class: int = 5,
        log=None,
    ):
        self.csv_path = csv_path
        self.data_path = data_path
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.hw = int(hw)
        self.num_class = int(num_class)
        self.log = log

        df = pd.read_csv(self.csv_path)

        # Basic sanity
        n = len(df)
        if n < 10:
            raise ValueError(f"CSV too small: {n} rows")

        # Split indices (random, not patient-wise; matches your preference)
        rng = np.random.RandomState(self.split_seed)
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        n_test = int(round(self.test_ratio * n))
        self.test_idx = all_idx[:n_test]
        self.train_idx = all_idx[n_test:]

        self.df_train = df.iloc[self.train_idx].reset_index(drop=True)
        self.df_test = df.iloc[self.test_idx].reset_index(drop=True)

        # Log split stats
        if self.log is not None:
            mal_train = self.df_train["malignancy"].astype(int).to_numpy()
            mal_test = self.df_test["malignancy"].astype(int).to_numpy()
            # class distribution in 1..5
            def dist(mal):
                vals, cnt = np.unique(mal, return_counts=True)
                return {int(v): int(c) for v, c in zip(vals, cnt)}
            self.log.write(f"LIDC split: train={len(self.df_train)} test={len(self.df_test)}\n")
            self.log.write(f"Train malignancy dist: {dist(mal_train)}\n")
            self.log.write(f"Test  malignancy dist: {dist(mal_test)}\n")
            self.log.flush()

    def run(self, mode, pred=None, prob=None):
        if mode == "warmup":
            ds = LIDCPatchDataset(
                df=self.df_train,
                data_path=self.data_path,
                mode="all",        # warmup expects (img,label,index) in Train script
                hw=self.hw,
                num_class=self.num_class,
                seed=self.split_seed,
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                              num_workers=self.num_workers, pin_memory=True, drop_last=True)

        if mode == "test":
            ds = LIDCPatchDataset(
                df=self.df_test,
                data_path=self.data_path,
                mode="test",
                hw=self.hw,
                num_class=self.num_class,
                seed=self.split_seed,
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True)

        if mode == "eval_train":
            ds = LIDCPatchDataset(
                df=self.df_train,
                data_path=self.data_path,
                mode="all",        # eval_train expects (img,label,index)
                hw=self.hw,
                num_class=self.num_class,
                seed=self.split_seed,
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=True)

        if mode == "train":
            if pred is None or prob is None:
                raise ValueError("train mode requires pred and prob")

            # pred/prob are over TRAIN set length (len(df_train))
            labeled_ds = LIDCPatchDataset(
                df=self.df_train,
                data_path=self.data_path,
                mode="labeled",
                hw=self.hw,
                pred=pred,
                prob=prob,
                num_class=self.num_class,
                seed=self.split_seed + 1,
            )
            unlabeled_ds = LIDCPatchDataset(
                df=self.df_train,
                data_path=self.data_path,
                mode="unlabeled",
                hw=self.hw,
                pred=pred,
                prob=prob,
                num_class=self.num_class,
                seed=self.split_seed + 2,
            )

            labeled_loader = DataLoader(labeled_ds, batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.num_workers, pin_memory=True, drop_last=True)
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

            print(f"labeled data has a size of {len(labeled_ds)}")
            print(f"unlabeled data has a size of {len(unlabeled_ds)}")

            return labeled_loader, unlabeled_loader

        raise ValueError(f"Unknown mode: {mode}")
