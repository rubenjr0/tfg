from pathlib import Path

import numpy as np
import torch
from geosynth import GeoSynth
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from uncertainty_estimation.utils import Pipeline


class ImageDepthDataset(Dataset):
    def __init__(self, root: str):
        super().__init__()
        self.data = GeoSynth(root, variant="demo")
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float, scale=False),
                T.Resize((256, 256)),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        path = Path(scene.path)
        image = scene.rgb.read()
        depth = scene.depth.read()
        est = np.load(path / "est.npy")

        image = self.transform(image)

        depth = self.transform(depth)
        depth_edges, depth_laplacian, _ = Pipeline.process_depth(depth.permute(1, 2, 0))
        depth_edges = self.transform(depth_edges)
        depth_laplacian = self.transform(depth_laplacian)

        est = self.transform(est)
        est_edges, est_laplacian, _ = Pipeline.process_depth(est.permute(1, 2, 0))
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
    import rerun as rr

    rr.init("depth-pipeline", spawn=True)
    dataset = ImageDepthDataset(root="data/geosynth")
    batch = dataset[0]

    for k in batch.keys():
        d = batch[k].permute(1, 2, 0)
        print(f"{k:<20} {batch[k].shape} {batch[k].dtype}")
        rr.log(
            k,
            rr.Image(d)
            if k == "image"
            else rr.DepthImage(d)
            if k == "est" or k == "depth"
            else rr.Tensor(d),
        )
