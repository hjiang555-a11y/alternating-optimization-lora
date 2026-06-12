"""Tests for model_utils — model loading, memory estimation, dtype resolution."""

import pytest
import torch
import torch.nn as nn

from altopt.model_utils import (
    ModelLoadConfig,
    load_model_and_tokenizer,
    enable_gradient_checkpointing,
    get_model_size_gb,
    estimate_training_memory_gb,
    _format_params,
)


class TestModelLoadConfig:
    def test_dtype_bf16(self):
        cfg = ModelLoadConfig("gpt2", dtype="bf16")
        assert cfg._torch_dtype == torch.bfloat16

    def test_dtype_fp16(self):
        cfg = ModelLoadConfig("gpt2", dtype="fp16")
        assert cfg._torch_dtype == torch.float16

    def test_dtype_fp32(self):
        cfg = ModelLoadConfig("gpt2", dtype="fp32")
        assert cfg._torch_dtype == torch.float32

    def test_dtype_int8(self):
        cfg = ModelLoadConfig("gpt2", dtype="int8")
        assert cfg._torch_dtype is None

    def test_dtype_int4(self):
        cfg = ModelLoadConfig("gpt2", dtype="int4")
        assert cfg._torch_dtype is None

    def test_default_device_map(self):
        cfg = ModelLoadConfig("gpt2")
        assert cfg.device_map == "auto"

    def test_gradient_checkpointing_default(self):
        cfg = ModelLoadConfig("gpt2")
        assert cfg.gradient_checkpointing is True

    def test_flash_attention_default(self):
        cfg = ModelLoadConfig("gpt2")
        assert cfg.use_flash_attention is True


class TestModelLoading:
    def test_load_gpt2(self):
        cfg = ModelLoadConfig("gpt2", dtype="fp32", device_map=None, use_flash_attention=False)
        model, tokenizer = load_model_and_tokenizer(cfg)
        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None

    def test_load_gpt2_bf16(self):
        cfg = ModelLoadConfig("gpt2", dtype="bf16", device_map=None, use_flash_attention=False)
        model, tokenizer = load_model_and_tokenizer(cfg)
        assert model is not None
        assert tokenizer is not None


class TestFormatParams:
    def test_billions(self):
        assert _format_params(7_000_000_000) == "7.00B"

    def test_millions(self):
        assert _format_params(125_000_000) == "125.00M"

    def test_thousands(self):
        assert _format_params(7_500) == "7.5K"

    def test_small(self):
        assert _format_params(42) == "42"


class TestGetModelSize:
    def test_simple_model(self):
        model = nn.Linear(1024, 1024)
        size = get_model_size_gb(model)
        assert 0.0 < size < 0.1

    def test_small_network(self):
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        size = get_model_size_gb(model)
        assert 0.0 < size < 0.01


class TestEstimateTrainingMemory:
    def test_basic_estimate(self):
        model = nn.Sequential(nn.Linear(1024, 1024))
        mem = estimate_training_memory_gb(model, batch_size=4, seq_length=512, optimizer_type="adamw")
        assert "model_params_gb" in mem
        assert "optimizer_states_gb" in mem
        assert "gradients_gb" in mem
        assert "activations_gb" in mem
        assert "total_estimated_gb" in mem
        for k, v in mem.items():
            assert v >= 0.0

    def test_adamw_vs_sgd(self):
        model = nn.Sequential(nn.Linear(1024, 1024))
        mem_adamw = estimate_training_memory_gb(model, batch_size=4, seq_length=512, optimizer_type="adamw")
        mem_sgd = estimate_training_memory_gb(model, batch_size=4, seq_length=512, optimizer_type="sgd")
        assert mem_adamw["optimizer_states_gb"] > mem_sgd["optimizer_states_gb"]

    def test_gradient_checkpointing_reduces_memory(self):
        model = nn.Sequential(nn.Linear(1024, 1024))
        mem_with = estimate_training_memory_gb(model, batch_size=4, seq_length=512,
                                                optimizer_type="adamw", use_gradient_checkpointing=True)
        mem_without = estimate_training_memory_gb(model, batch_size=4, seq_length=512,
                                                   optimizer_type="adamw", use_gradient_checkpointing=False)
        assert mem_with["activations_gb"] <= mem_without["activations_gb"]


class TestGradientCheckpointing:
    def test_enable_on_model_with_support(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        enable_gradient_checkpointing(model)
        assert model.is_gradient_checkpointing
