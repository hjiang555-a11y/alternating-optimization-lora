"""Tests for PeftBridge architecture detection and graceful degradation."""

import pytest
import torch
import torch.nn as nn


class MockGPT2Config:
    model_type = "gpt2"


class MockOPTConfig:
    model_type = "opt"


class MockLlamaConfig:
    model_type = "llama"


class MockGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockGPT2Config()
        self.transformer = nn.Sequential(nn.Linear(16, 16))

    def named_modules(self):
        yield from super().named_modules()


class MockOPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockOPTConfig()
        self.q_proj = nn.Linear(16, 16)

    def named_modules(self):
        yield from super().named_modules()


class MockLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockLlamaConfig()

    def named_modules(self):
        yield from super().named_modules()


class TestArchitectureDetection:
    def test_detect_gpt2(self):
        from altopt.peft_bridge import detect_target_modules
        model = MockGPT2Model()
        targets = detect_target_modules(model)
        assert targets == ["c_attn", "c_proj"]

    def test_detect_opt(self):
        from altopt.peft_bridge import detect_target_modules
        model = MockOPTModel()
        targets = detect_target_modules(model)
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert "k_proj" in targets

    def test_detect_llama(self):
        from altopt.peft_bridge import detect_target_modules
        model = MockLlamaModel()
        targets = detect_target_modules(model)
        assert targets == ["q_proj", "v_proj", "k_proj", "o_proj"]

    def test_detect_fallback_by_module_names(self):
        from altopt.peft_bridge import detect_target_modules

        class UnknownModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"model_type": "unknown"})()

            def named_modules(self):
                return [("model.layers.0.self_attn.q_proj", nn.Linear(16, 16))]

        model = UnknownModel()
        targets = detect_target_modules(model)
        assert "q_proj" in targets

    def test_detect_fallback_default(self):
        from altopt.peft_bridge import detect_target_modules

        class BareModel(nn.Module):
            def __init__(self):
                super().__init__()

        model = BareModel()
        if not hasattr(model, "config"):
            targets = detect_target_modules(model)
            assert isinstance(targets, list)
            assert len(targets) > 0


class TestModelSupportsLora:
    def test_gpt2_not_supported(self):
        from altopt.peft_bridge import model_supports_lora
        model = MockGPT2Model()
        assert model_supports_lora(model) is False

    def test_opt_supported(self):
        from altopt.peft_bridge import model_supports_lora
        model = MockOPTModel()
        assert model_supports_lora(model) is True

    def test_has_linear_detected(self):
        from altopt.peft_bridge import model_supports_lora

        class ModelWithLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 16)

        model = ModelWithLinear()
        assert model_supports_lora(model) is True


class TestTrainerProtocolC_Fallback:
    def test_gpt2_protocol_c_does_not_crash(self):
        """Protocol C on GPT-2 should fallback gracefully, not crash."""
        import logging
        logging.disable(logging.CRITICAL)

        from altopt.trainer import AltOptTrainer, TrainerConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        import torch
        dummy_batch = {
            "input_ids": torch.randint(0, 50000, (1, 16)),
            "attention_mask": torch.ones(1, 16),
            "labels": torch.randint(0, 50000, (1, 16)),
        }

        class DummyDL:
            def __iter__(self):
                yield dummy_batch

        cfg = TrainerConfig(
            protocol="C", optimizer_type="altopt", parameter_form="lora",
            max_steps=1, run_dir="/tmp/test_pc_fallback", seed=42,
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=DummyDL(), tokenizer=tokenizer)

        state = trainer.train(DummyDL())
        assert state.step >= 1
        logging.disable(logging.NOTSET)

    def test_gpt2_protocol_d_does_not_crash(self):
        """Protocol D on GPT-2 should fallback gracefully."""
        import logging
        logging.disable(logging.CRITICAL)

        from altopt.trainer import AltOptTrainer, TrainerConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        import torch
        dummy_batch = {
            "input_ids": torch.randint(0, 50000, (1, 16)),
            "attention_mask": torch.ones(1, 16),
            "labels": torch.randint(0, 50000, (1, 16)),
        }

        class DummyDL:
            def __iter__(self):
                yield dummy_batch

        cfg = TrainerConfig(
            protocol="D", optimizer_type="adamw", parameter_form="lora",
            max_steps=1, run_dir="/tmp/test_pd_fallback", seed=42,
        )
        trainer = AltOptTrainer(model, cfg, eval_dataloader=DummyDL(), tokenizer=tokenizer)

        state = trainer.train(DummyDL())
        assert state.step >= 1
        logging.disable(logging.NOTSET)
