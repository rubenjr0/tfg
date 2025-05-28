import numpy as np
import os
from uncertainty_estimation.utils import Pipeline
from uncertainty_estimation.data import ImageDepthDataset
from tqdm import tqdm


def preprocess():
    pipeline = Pipeline(None, None)
    dataset = ImageDepthDataset(root="data/", preprocess=True)
    for k in tqdm(range(len(dataset))):
        entry = dataset[k]
        rgb_path = entry["image_path"]
        path = rgb_path.replace("color.jpg", "est.npy")
        # if path existrs, skip
        if os.path.exists(path):
            continue
        image = entry["image"]
        estimated = pipeline.estimate_depth(image)
        estimated = estimated.cpu().numpy()
        np.save(path, estimated)
