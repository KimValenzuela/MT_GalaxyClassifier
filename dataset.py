import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits

CLASS_COLS = [
    "p_smooth",
    "p_features",
    "p_irregular",
    "p_point_source",
    "p_unclassifiable",
]

class GalaxyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, log_scale=True, eps=1e-8):
        """
        df: DataFrame con columnas [ID] + CLASS_COLS
        img_dir: carpeta donde están los .fits
        transform: transforms de torchvision (que acepten tensores CHW)
        log_scale: si True, aplica log(1 + x) a la imagen
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.log_scale = log_scale
        self.eps = eps

    def __len__(self):
        return len(self.df)

    def _load_fits(self, img_id):
        path = os.path.join(self.img_dir, f"{img_id}.fits")
        with fits.open(path) as hdul:
            img = hdul[0].data.astype("float32")
        return img

    def _preprocess_image(self, img):
        img = np.nan_to_num(img, nan=0.0)

        # Quita outliers
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin + self.eps)  # [0,1]

        if self.log_scale:
            img = np.log1p(img)  # resalta estructura débil

        # (H, W) -> (1, H, W)
        img = img.astype("float32")
        img = torch.from_numpy(img).unsqueeze(0)  # 1 canal
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["ID"]

        # Imagen
        img = self._load_fits(img_id)
        img = self._preprocess_image(img)

        if self.transform is not None:
            img = self.transform(img)

        # Soft label
        probs = row[CLASS_COLS].astype(float).to_numpy(dtype="float32")
        y = torch.from_numpy(probs)

        # Hard label (aux)
        hard_label = int(torch.argmax(y).item())

        return img, y, hard_label

