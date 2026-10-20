"""Deterministic core tests; no pretrained checkpoint needed."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from nca import GrowingNCA, flower_target, render_rgba
from train import damage_random_rectangle

def test_seed_shape_and_center_alpha():
    model = GrowingNCA(channels=12,hidden=16)
    x = model.seed(16,batch=2)
    assert x.shape == (2,12,16,16)
    assert torch.all(x[:,3:,8,8] == 1)
    assert x[:,3,:].count_nonzero().item() == 2

def test_zero_initialized_model_preserves_seed():
    torch.manual_seed(0)
    model = GrowingNCA(channels=8,hidden=16)
    x = model.seed(16)
    assert torch.equal(x,model.rollout(x,4))

def test_trainable_gradients():
    torch.manual_seed(2)
    model = GrowingNCA(channels=8,hidden=16)
    x = model.rollout(model.seed(16,batch=2),3)
    loss = (model.rgba(x) - flower_target(16)).square().mean()
    loss.backward()
    assert model.fc1.weight.grad is not None
    assert torch.isfinite(model.fc1.weight.grad).all()
    assert model.fc1.weight.grad.abs().sum() > 0

def test_damage_and_render_shapes():
    torch.manual_seed(3)
    x = GrowingNCA(channels=8).seed(16,batch=2)
    x[:] = 1
    y = damage_random_rectangle(x.clone(),count=1)
    assert (y[0] == 0).any()
    assert (y[1] == 1).all()
    image = render_rgba(y[0])
    assert image.shape == (16,16,3)
    assert image.dtype == np.uint8
