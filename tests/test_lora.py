"""Tests for the LoRA baseline implementation."""

import pytest
import torch
import torch.nn as nn

from altopt.lora import LoRABaseline, LoRAConfig, LoRALayer


class TinyTransformer(nn.Module):
    """Minimal transformer-like model for testing LoRA injection."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Linear(d_model, d_model * 4)

    def forward(self, x: torch.Tensor, labels=None) -> dict:
        x = self.o_proj(self.v_proj(x))
        x = self.ffn(x)
        loss = x.mean() if labels is None else ((x - labels) ** 2).mean()
        # Return an object with .loss attribute to match HuggingFace convention
        output = type("Output", (), {"loss": loss, "logits": x})()
        return output


@pytest.fixture
def model():
    return TinyTransformer()


@pytest.fixture
def batch():
    x = torch.randn(2, 64)
    return {"x": x}


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_config(self):
        config = LoRAConfig()
        assert config.r == 8
        assert config.alpha == 16.0
        assert config.scaling == 2.0  # alpha / r = 16/8

    def test_custom_config(self):
        config = LoRAConfig(r=4, alpha=8.0)
        assert config.scaling == 2.0


class TestLoRALayer:
    """Tests for the LoRALayer wrapper."""

    def test_lora_params_on_base_device(self):
        """LoRA parameters must be on the same device as the base layer."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        base = nn.Linear(64, 32).to("cuda")
        config = LoRAConfig(r=4)
        lora = LoRALayer(base, config)

        assert lora.lora_A.device == base.weight.device, \
            f"lora_A on {lora.lora_A.device}, expected {base.weight.device}"
        assert lora.lora_B.device == base.weight.device, \
            f"lora_B on {lora.lora_B.device}, expected {base.weight.device}"

    def test_forward_cuda_no_device_mismatch(self):
        """Forward pass must work when base layer is on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        base = nn.Linear(64, 32).to("cuda")
        config = LoRAConfig(r=4)
        lora = LoRALayer(base, config)

        x = torch.randn(8, 64, device="cuda")
        out = lora(x)
        assert out.shape == (8, 32)

    def test_forward_shape_preserved(self):
        base = nn.Linear(64, 32)
        config = LoRAConfig(r=4, alpha=8.0)
        lora = LoRALayer(base, config)

        x = torch.randn(8, 64)
        out = lora(x)

        assert out.shape == (8, 32)

    def test_base_weights_frozen(self):
        base = nn.Linear(64, 32)
        config = LoRAConfig(r=4)
        lora = LoRALayer(base, config)

        assert not base.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_b_initialized_to_zero(self):
        base = nn.Linear(64, 32)
        config = LoRAConfig(r=4)
        lora = LoRALayer(base, config)

        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_lora_a_initialized_nonzero(self):
        base = nn.Linear(64, 32)
        config = LoRAConfig(r=4)
        lora = LoRALayer(base, config)

        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))


class TestLoRABaseline:
    """Tests for the LoRA baseline wrapper."""

    def test_applies_lora_to_target_modules(self, model):
        config = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
        baseline = LoRABaseline(model, config)

        # Check that target modules were replaced
        assert isinstance(model.q_proj, LoRALayer)
        assert isinstance(model.v_proj, LoRALayer)
        # Non-target modules should remain unchanged
        assert isinstance(model.k_proj, nn.Linear)
        assert isinstance(model.o_proj, nn.Linear)

    def test_num_trainable_params_is_reduced(self, model):
        full_params = sum(p.numel() for p in model.parameters())

        config = LoRAConfig(r=4)
        baseline = LoRABaseline(model, config)

        trainable = baseline.num_trainable_params()
        assert trainable < full_params, "LoRA should reduce trainable parameters"
        assert trainable > 0, "Should have some trainable parameters"

    def test_step_reduces_loss(self, model, batch):
        config = LoRAConfig(r=4, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "ffn"])
        baseline = LoRABaseline(model, config, lr=0.1)

        loss_before = model(batch["x"]).loss.item()
        loss = baseline.step(batch)

        assert isinstance(loss, float)
        assert len(baseline.losses) == 1
        assert len(baseline.grad_norms) == 1

    def test_merge_and_unload(self, model):
        config = LoRAConfig(r=4, target_modules=["q_proj"])
        baseline = LoRABaseline(model, config)

        # Record pre-merge state
        was_lora = isinstance(model.q_proj, LoRALayer)

        merged = baseline.merge_and_unload()

        # After merge, LoRA layer should be removed
        assert isinstance(merged.q_proj, nn.Linear)
        assert was_lora, "q_proj should have been LoRA-injected before merge"

    def test_get_lora_params(self, model):
        config = LoRAConfig(r=4, target_modules=["q_proj"])
        baseline = LoRABaseline(model, config)

        lora_params = baseline.get_lora_params()

        assert len(lora_params) > 0
        for name in lora_params:
            assert "lora" in name.lower() or model.q_proj.lora_A is not None
