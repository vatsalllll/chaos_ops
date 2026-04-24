"""LLM-backed :class:`Policy` driven by a LoRA-tuned Qwen checkpoint.

This is the counterpart to ``random_policy`` / ``heuristic_policy`` /
``oracle_policy`` in :mod:`chaosops.agents.policies`. It loads a LoRA
adapter (produced by :mod:`chaosops.train.grpo_train`) on top of a base
Qwen model and serves the ``(obs, role) -> ChaosOpsAction`` interface the
runner + evaluator expect.

Kept in its own module so importing :mod:`chaosops.agents.policies`
(which the scripted baselines do) never drags in torch / transformers /
peft. The import cost is ~4 s cold — only paid when a caller explicitly
constructs a :class:`TrainedPolicy`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chaosops.agents.llm_adapter import build_prompt, parse_action
from chaosops.agents.policies import Policy
from chaosops.env.models import (
    ActionType,
    AgentRole,
    ChaosOpsAction,
    ChaosOpsObservation,
)


_LOG = logging.getLogger(__name__)


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class TrainedPolicyConfig:
    adapter_path: Path
    base_model: str = DEFAULT_BASE_MODEL
    device: str | None = None  # auto-detect if None
    max_new_tokens: int = 96
    temperature: float = 0.7
    max_seq_length: int = 1024


class TrainedPolicy:
    """Callable policy that wraps a LoRA-adapted Qwen model.

    Usage::

        policy = TrainedPolicy.from_adapter("artifacts/chaosops-grpo/lora_adapter")
        action = policy(observation, role)

    The instance caches the loaded model + tokenizer across calls so a full
    evaluation sweep (~100+ episodes) pays the load cost exactly once.
    """

    def __init__(self, config: TrainedPolicyConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._device = None

    # ------------------------------------------------------------------ loaders

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 96,
        temperature: float = 0.7,
    ) -> "TrainedPolicy":
        """Load a LoRA adapter; infer the base model from ``adapter_config.json``
        if the caller doesn't supply one explicitly.
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"adapter path not found: {adapter_path}. Did you sync the "
                "Colab artifacts/chaosops-grpo/lora_adapter/ folder?"
            )
        resolved_base = base_model or _infer_base_model(adapter_path) or DEFAULT_BASE_MODEL
        cfg = TrainedPolicyConfig(
            adapter_path=adapter_path,
            base_model=resolved_base,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return cls(cfg)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Lazy imports so scripted policies stay torch-free.
        import torch  # type: ignore[import-not-found]
        from peft import PeftModel  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        _LOG.info(
            "loading TrainedPolicy base=%s adapter=%s device=%s",
            self.config.base_model,
            self.config.adapter_path,
            device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.adapter_path)
            if (self.config.adapter_path / "tokenizer_config.json").exists()
            else self.config.base_model
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.config.base_model, torch_dtype=dtype
        )
        model = PeftModel.from_pretrained(base, str(self.config.adapter_path))
        model.to(device)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    # ------------------------------------------------------------------ inference

    def __call__(self, obs: ChaosOpsObservation, role: AgentRole) -> ChaosOpsAction:
        self._ensure_loaded()
        prompt = build_prompt(obs)
        completion = self._generate(prompt, role)
        return parse_action(completion, role=role, fallback=ActionType.NOOP)

    def _generate(self, prompt: str, role: AgentRole) -> str:
        assert self._model is not None and self._tokenizer is not None
        import torch  # type: ignore[import-not-found]

        messages = [
            {
                "role": "system",
                "content": f"You are the {role.value.upper()} agent in ChaosOps AI.",
            },
            {"role": "user", "content": prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------ Policy interface

    def as_policy(self) -> Policy:
        """Return a plain ``(obs, role) -> action`` callable for APIs that
        type-check the :data:`Policy` alias instead of a class instance."""

        def _policy(obs: ChaosOpsObservation, role: AgentRole) -> ChaosOpsAction:
            return self(obs, role)

        return _policy


def _infer_base_model(adapter_path: Path) -> str | None:
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(config_file.read_text())
    except json.JSONDecodeError:
        return None
    return payload.get("base_model_name_or_path")


__all__ = ["TrainedPolicy", "TrainedPolicyConfig", "DEFAULT_BASE_MODEL"]
