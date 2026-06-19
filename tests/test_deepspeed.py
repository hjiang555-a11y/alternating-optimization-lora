"""Tests for DeepSpeed engine configuration (no GPU needed)."""
import json
import torch
import torch.nn as nn
from altopt.deepspeed_engine import DeepSpeedConfig


class TestDeepSpeedConfig:
    def test_zero_stage_2_bf16(self):
        cfg = DeepSpeedConfig(zero_stage=2, bf16_enabled=True)
        d = cfg.to_dict()
        assert d["bf16"]["enabled"] is True
        assert d["zero_optimization"]["stage"] == 2
        assert d["zero_optimization"]["reduce_scatter"] is True

    def test_zero_stage_3_fp16(self):
        cfg = DeepSpeedConfig(zero_stage=3, bf16_enabled=False, fp16_enabled=True)
        d = cfg.to_dict()
        assert "fp16" in d
        assert d["fp16"]["enabled"] is True
        assert d["zero_optimization"]["stage"] == 3

    def test_zero_stage_0_no_zero_section(self):
        cfg = DeepSpeedConfig(zero_stage=0)
        d = cfg.to_dict()
        assert "zero_optimization" not in d

    def test_save_load_roundtrip(self, tmp_path):
        cfg = DeepSpeedConfig(zero_stage=2)
        path = tmp_path / "ds_config.json"
        cfg.save(str(path))
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["zero_optimization"]["stage"] == 2

    def test_defaults(self):
        cfg = DeepSpeedConfig()
        assert cfg.zero_stage == 2
        assert cfg.bf16_enabled is True
        assert cfg.gradient_clipping == 1.0
