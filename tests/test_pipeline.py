import torch
from uncertainty_estimation.utils.pipeline import Pipeline


def test_refine():
    mu1 = torch.rand(1, 1, 128, 128)
    var1 = torch.rand(1, 1, 128, 128)
    mu2 = torch.rand(1, 1, 128, 128)
    var2 = torch.rand(1, 1, 128, 128)
    mu_refined, var_refined = Pipeline.refine(mu1, var1, mu2, var2)
    assert mu_refined.shape == mu1.shape
    assert var_refined.shape == var1.shape
    assert var_refined.mean() < var1.mean()
    assert var_refined.mean() < var2.mean()
