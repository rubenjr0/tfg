from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from uncertainty_estimation.utils import process_depth


class ImageDepthDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        all = glob(f"{root}/**/*final_preview/**", recursive=True)
        self.paths = [
            (
                r,
                r.replace("final_preview", "geometry_hdf5").replace(
                    ".color.jpg", ".depth_meters.hdf5"
                ),
            )
            for r in all
            if ".color.jpg" in r
        ]
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float, scale=False),
                T.Resize((256, 256)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.paths[idx]
        image = Image.open(rgb_path).convert("RGB")
        depth = h5py.File(depth_path, "r")["dataset"][:]

        image = self.transform(image)
        est = np.load(rgb_path.replace("color.jpg", "est.npy"))

        depth = self.transform(depth)
        depth_edges, depth_laplacian, _ = process_depth(depth.permute(1, 2, 0))
        depth_edges = self.transform(depth_edges)
        depth_laplacian = self.transform(depth_laplacian)

        est = self.transform(est)
        est_edges, est_laplacian, _ = process_depth(est.permute(1, 2, 0))
        est_edges = self.transform(est_edges)
        est_laplacian = self.transform(est_laplacian)
        return {
            "image": image,
            "depth": depth,
            "depth_edges": depth_edges,
            "depth_laplacian": depth_laplacian,
            "est": est,
            "est_edges": est_edges,
            "est_laplacian": est_laplacian,
        }


if __name__ == "__main__":
    dataset = ImageDepthDataset(root="data/ai_001_001/images")
    entry = dataset[0]

    for k in entry.keys():
        d = entry[k].permute(1, 2, 0)
        print(f"{k:<20} {entry[k].shape} {entry[k].dtype}")
