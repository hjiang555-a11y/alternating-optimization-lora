"""Tests for stochastic perturbation mechanisms."""

import pytest
import torch
import torch.nn as nn

from altopt.perturbation import PerturbationScheduler


@pytest.fixture
def simple_model():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.attn_fc = nn.Linear(16, 16)
            self.ffn_fc = nn.Linear(16, 16)

    return SimpleModel()


@pytest.fixture
def frozen_model():
    model = nn.Sequential(
        nn.Linear(8, 8),
        nn.Linear(8, 4),
    )
    model[0].weight.requires_grad = False
    model[0].bias.requires_grad = False
    return model


class TestPerturbationGaussian:
    def test_apply_gaussian_noise(self, simple_model):
        scheduler = PerturbationScheduler(simple_model, noise_type="gaussian", initial_scale=0.01)
        params_before = {n: p.clone() for n, p in simple_model.named_parameters() if p.requires_grad}

        energy = scheduler.apply_noise(scale=0.01)
        assert energy > 0.0

        for n, p in simple_model.named_parameters():
            if p.requires_grad:
                assert not torch.equal(p, params_before[n])

    def test_gaussian_mean_near_zero(self, simple_model):
        scheduler = PerturbationScheduler(simple_model, noise_type="gaussian", initial_scale=0.1)
        params_before = {n: p.clone() for n, p in simple_model.named_parameters() if p.requires_grad}

        scheduler.apply_noise(scale=0.1)
        param_deltas = []
        for n, p in simple_model.named_parameters():
            if p.requires_grad:
                param_deltas.append((p - params_before[n]).mean().item())

        import numpy as np
        assert abs(np.mean(param_deltas)) < 0.05


class TestPerturbationUniform:
    def test_apply_uniform_noise(self, simple_model):
        scheduler = PerturbationScheduler(simple_model, noise_type="uniform", initial_scale=0.01)
        params_before = {n: p.clone() for n, p in simple_model.named_parameters() if p.requires_grad}

        energy = scheduler.apply_noise(scale=0.01)
        assert energy > 0.0

        for n, p in simple_model.named_parameters():
            if p.requires_grad:
                assert not torch.equal(p, params_before[n])


class TestFrozenParameters:
    def test_frozen_params_unchanged(self, frozen_model):
        scheduler = PerturbationScheduler(frozen_model, initial_scale=0.1)
        before = frozen_model[0].weight.clone()

        scheduler.apply_noise(scale=0.1)
        assert torch.equal(frozen_model[0].weight, before)

    def test_trainable_params_changed(self, frozen_model):
        scheduler = PerturbationScheduler(frozen_model, initial_scale=0.1)
        before = frozen_model[1].weight.clone()

        scheduler.apply_noise(scale=0.1)
        assert not torch.equal(frozen_model[1].weight, before)


class TestDecaySchedules:
    def test_cosine_decay(self, simple_model):
        scheduler = PerturbationScheduler(
            simple_model, decay_schedule="cosine", initial_scale=0.1
        )
        scale_0 = scheduler._compute_scale(None)
        scheduler.cycle = 5
        scale_5 = scheduler._compute_scale(None)
        scheduler.cycle = 10
        scale_10 = scheduler._compute_scale(None)
        assert scale_10 < scale_0
        assert scale_10 < scale_5

    def test_exponential_decay(self, simple_model):
        scheduler = PerturbationScheduler(
            simple_model, decay_schedule="exponential", initial_scale=0.1
        )
        scale_0 = scheduler._compute_scale(None)
        scheduler.cycle = 3
        scale_3 = scheduler._compute_scale(None)
        assert scale_3 < scale_0

    def test_constant_schedule(self, simple_model):
        scheduler = PerturbationScheduler(
            simple_model, decay_schedule="constant", initial_scale=0.1
        )
        scale_0 = scheduler._compute_scale(None)
        scheduler.cycle = 10
        scale_10 = scheduler._compute_scale(None)
        assert scale_0 == scale_10

    def test_override_scale(self, simple_model):
        scheduler = PerturbationScheduler(simple_model, initial_scale=0.1)
        scale = scheduler._compute_scale(override_scale=0.05)
        assert scale == 0.05

    def test_min_scale_bound(self, simple_model):
        scheduler = PerturbationScheduler(
            simple_model, initial_scale=1e-3, min_scale=1e-6, decay_schedule="exponential"
        )
        scheduler.cycle = 100
        scale = scheduler._compute_scale(None)
        assert scale >= 1e-6


class TestLayerMultipliers:
    def test_embedding_multiplier(self):
        assert PerturbationScheduler._layer_multiplier("model.embed_tokens.weight") == 0.1

    def test_attention_multiplier(self):
        assert PerturbationScheduler._layer_multiplier("model.layers.0.self_attn.q_proj") == 0.5

    def test_ffn_multiplier(self):
        assert PerturbationScheduler._layer_multiplier("model.layers.0.mlp.fc1") == 1.0

    def test_default_multiplier(self):
        assert PerturbationScheduler._layer_multiplier("model.layers.0.layer_norm") == 0.5


class TestReset:
    def test_reset_cycle(self, simple_model):
        scheduler = PerturbationScheduler(simple_model)
        scheduler.cycle = 5
        scheduler.reset(cycle=0)
        assert scheduler.cycle == 0

    def test_reset_default(self, simple_model):
        scheduler = PerturbationScheduler(simple_model)
        scheduler.cycle = 5
        scheduler.reset()
        assert scheduler.cycle == 0


class TestEnergyComputation:
    def test_energy_positive(self, simple_model):
        scheduler = PerturbationScheduler(simple_model, initial_scale=0.01)
        energy = scheduler.apply_noise(scale=0.01)
        assert energy > 0.0

    def test_larger_scale_gives_larger_energy(self, simple_model):
        model1 = simple_model.__class__()
        model2 = simple_model.__class__()

        s1 = PerturbationScheduler(model1, initial_scale=0.01)
        s2 = PerturbationScheduler(model2, initial_scale=0.1)

        e1 = s1.apply_noise(scale=0.01)
        e2 = s2.apply_noise(scale=0.1)
        assert e2 > e1
