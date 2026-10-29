#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO fine-tuning for Muse Glimmer on Modal (RTX PRO 6000) with Langfuse eval + Meta judge.

Pipeline overview
-----------------
1. **Train / infer on Modal.** ``train_grpo`` and ``generate`` are Modal
   functions pinned to ``RTX-PRO-6000``. They load Muse Glimmer via Unsloth
   (4-bit) and run TRL GRPO. Nothing here needs a local GPU.
2. **Evaluate on AFM-WebAgent-RL-Dataset.**
   ``PersonalAILab/AFM-WebAgent-RL-Dataset`` (10.4k Chain-of-Agents web-agent
   trajectories) is the eval / RL prompt source. Each row carries a CoA system
   prompt (think/plan/tool/.../answer), a ``reward_model.ground_truth`` target
   list, and ``extra_info`` with the question.
3. **Judge with Muse Wonder via Meta's API.** ``MetaJudgeClient`` calls Meta's
   OpenAI-compatible endpoint using exactly two env keys:
   ``JUDGE_MODEL`` and ``JUDGE_API_KEY`` (plus optional ``JUDGE_BASE_URL``).
4. **Langfuse testing harness closes the loop.** Every judged completion is
   logged to Langfuse as a trace + score. The GRPO ``llm_judge_reward`` reads
   that score back (or uses it directly) so Langfuse output *is* the RL signal.

Separation of concerns (read top to bottom, extend without cross-edits):

- ``*_Config`` dataclasses .... all tunables, all env-overridable.
- ``modal_infra`` ........... Modal App / Image / Volume / Secrets (only Modal dep at import).
- ``afm_dataset`` ........... AFM row -> GRPO prompt / ground truth helpers (pure python).
- ``glimmer_channels`` ...... Muse Glimmer ``to=self`` / ``to=user`` splitting (pure python).
- ``verifiable`` ............ exact-match / token-F1 reward against ground truth (pure python).
- ``MetaJudgeClient`` ....... Muse Wonder via Meta API (lazy ``openai`` import).
- ``LangfuseHarness`` ....... eval logging + score fetch (lazy ``langfuse`` import).
- ``grpo_rewards`` .......... the three TRL reward functions (format, verifiable, judge).
- ``modal_fns`` ............. ``@app.function`` training / inference / eval (heavy imports inside).
- ``local_entrypoint`` ...... ``modal run starter.py --action train`` etc.

Environment variables
---------------------
Required for judged runs:

- ``JUDGE_MODEL`` .... e.g. ``muse-wonder`` (Muse Wonder judge id on Meta's API).
- ``JUDGE_API_KEY`` .. Meta API key for the judge model.

Optional:

- ``JUDGE_BASE_URL`` ......... Meta OpenAI-compatible base URL
  (default ``https://api.meta.ai/v1`` — confirmed Meta endpoint).
- ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST``
  for the Langfuse instance. Training still runs without them; judge scores
  are then returned directly instead of round-tripping through Langfuse.
- ``MODAL_VOLUME`` ........... persistent volume name (default ``glimmer-vol``).
- ``HF_DATASET`` ............. override AFM dataset id (default below).
- ``HF_TOKEN_NEW`` ........... Hugging Face token (mirrored to ``HF_TOKEN``
  locally and in Modal secrets; used for gated repos and rate limits).

Local usage (no GPU needed)::

    modal run starter.py --action train            # full GRPO run on RTX-PRO-6000
    modal run starter.py --action evaluate         # eval current LoRA via Langfuse + judge
    python starter.py --dry-run                    # validate configs/rewards locally

Original DSMath/Unsloth logic preserved: transformers 5.15.0 pin, ``<|eot|>``
eos fix, head-aware device-map planner, no vLLM (unsupported arch), GSPO
sequence-level importance sampling, ``mask_truncated_completions=True``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Only hard dependency at import time is `modal` (already installed per spec).
# Everything else (torch / unsloth / trl / datasets / langfuse / openai) is
# imported lazily inside the function that needs it so `python starter.py
# --dry-run` works on a plain laptop.
# ---------------------------------------------------------------------------

import modal  # noqa: E402  (Modal SDK — training & inference transport)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("glimmer-grpo")

APP_NAME = "muse-glimmer-grpo"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    """Read env var, returning ``default`` when unset/empty."""
    value = os.environ.get(name, default)
    return value if value != "" else default


def hf_token() -> str:
    """Hugging Face token, read fresh from env (``HF_TOKEN_NEW`` first).

    Never logged or printed — only passed to HF APIs / Modal secrets.
    """
    return _env("HF_TOKEN_NEW", _env("HF_TOKEN", ""))


# Mirror HF_TOKEN_NEW -> HF_TOKEN locally so huggingface_hub (which only
# reads HF_TOKEN) authenticates for gated repos and higher rate limits.
if hf_token() and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token()

# Xet high-performance transfers for model/dataset downloads (chunked,
# resumable CAS fetches instead of single-stream HTTP). huggingface_hub
# picks this up automatically wherever it runs.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", _env("HF_XET_HIGH_PERFORMANCE", "1"))


@dataclass(frozen=True)
class ModelConfig:
    """Which Muse Glimmer weights to load."""

    base_repo: str = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
    full_precision_repo: str = "unsloth/Muse-Glimmer-30B"
    lora_rank: int = 8
    finetune_vision: bool = False  # text-only task; keep vision tower frozen
    transformers_pin: str = "transformers==5.15.0"  # first release with muse-glimmer arch
    # Hub destination for the trained LoRA (HF_TOKEN_NEW auth).
    hub_repo: str = field(
        default_factory=lambda: _env("HF_REPO", "pranavupadhyaya52/muse_glimmer_agentFT")
    )
    hub_private: bool = field(default_factory=lambda: _env("HF_PRIVATE", "1") == "1")


@dataclass(frozen=True)
class DataConfig:
    """AFM WebAgent RL dataset → GRPO prompt mapping."""

    hf_id: str = field(default_factory=lambda: _env("HF_DATASET", "PersonalAILab/AFM-WebAgent-RL-Dataset"))
    split: str = field(default_factory=lambda: _env("HF_SPLIT", "train"))
    eval_size: int = 256  # rows held out for the Langfuse eval harness
    max_prompt_tokens: int = 1024  # CoA prompts are longer than GSM8K; sized generously
    reasoning_strength: str = field(default_factory=lambda: _env("REASONING_STRENGTH", "low"))
    system_overlay: str = (
        "You are Muse Glimmer. Work the problem step by step in your private "
        "to=self reasoning (you may use the Chain-of-Agents functions think, "
        "plan, tool, observation, reflection there). You MUST finish with a "
        "reply on the user channel: open it with `to=user` and put the final "
        "answer there and nothing else. Copy your Chain-of-Agents `answer:` "
        "content verbatim into the to=user reply."
    )


@dataclass(frozen=True)
class TrainConfig:
    """GRPO hyper-parameters sized for a single RTX PRO 6000 (96GB)."""

    # RTX-PRO-6000 comfortably fits num_generations=8 @ 3072 tokens; keep 4 as the
    # safe default and raise to 8 once `frac_reward_zero_std` ≈ 1 (see docstring
    # in `train_grpo`). Every +1 generation costs linear memory/step-time.
    num_generations: int = field(default_factory=lambda: int(_env("NUM_GENERATIONS", "4")))
    per_device_batch: int = field(default_factory=lambda: int(_env("BATCH_SIZE", "4")))
    max_seq_length: int = field(default_factory=lambda: int(_env("MAX_SEQ_LENGTH", "3072")))
    learning_rate: float = 5e-6
    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "200")))
    # Checkpoint every 5 steps (first at step 5) so interrupted runs lose
    # almost nothing. Override with SAVE_STEPS. Only ONE checkpoint is kept:
    # each new save deletes the previous (same-slot rotation), so the volume
    # never accumulates checkpoint-5, checkpoint-10, ...
    save_steps: int = field(default_factory=lambda: int(_env("SAVE_STEPS", "5")))
    save_total_limit: int = field(default_factory=lambda: int(_env("SAVE_TOTAL_LIMIT", "1")))
    output_dir: str = "/vol/outputs"
    lora_out: str = "/vol/muse_glimmer_lora"
    # GRPO log-softmax chunking: must match UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK.
    grpo_rows_per_chunk: int = 128


@dataclass(frozen=True)
class JudgeConfig:
    """Muse Wonder via Meta's API. Env keys are exactly JUDGE_MODEL / JUDGE_API_KEY."""

    model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "muse-wonder"))
    api_key: str = field(default_factory=lambda: _env("JUDGE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: _env("JUDGE_BASE_URL", "https://api.meta.ai/v1"))
    timeout_s: float = 60.0
    max_tokens: int = 512
    temperature: float = 0.0
    # Exception-retry budget for transient gateway errors (e.g. 404 during
    # model propagation, 429/5xx): immediate retries — a blip is retried at
    # once, and only a persistent failure falls back. No post-success pause:
    # pacing is handled purely via exceptions.
    max_retries: int = field(default_factory=lambda: int(_env("JUDGE_MAX_RETRIES", "6")))


@dataclass(frozen=True)
class LangfuseConfig:
    """Langfuse instance used as the testing harness + reward store."""

    public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY", ""))
    secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY", ""))
    host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", ""))
    session_id: str = field(default_factory=lambda: _env("LANGFUSE_SESSION", "glimmer-grpo"))
    enabled: bool = field(default_factory=lambda: bool(_env("LANGFUSE_HOST", "") and _env("LANGFUSE_PUBLIC_KEY", "")))

    def as_modal_secrets(self) -> list[modal.Secret]:
        """Secrets injected into Modal functions (missing secrets are skipped)."""
        secrets: list[modal.Secret] = []
        if self.enabled or True:  # from_dotenv never fails; values may be empty
            secrets.append(modal.Secret.from_dict({}))
        return secrets


@dataclass(frozen=True)
class InfraConfig:
    # RTX PRO 6000 (Blackwell, 96GB): native FP4 via Unsloth/Triton kernels.
    gpu: str = field(default_factory=lambda: _env("MODAL_GPU", "RTX-PRO-6000"))
    timeout_s: int = 86400  # 24h; long GRPO runs on 28B need it
    volume_name: str = field(default_factory=lambda: _env("MODAL_VOLUME", "glimmer-vol"))


MODEL = ModelConfig()
DATA = DataConfig()
TRAIN = TrainConfig()
JUDGE = JudgeConfig()
LANGFUSE = LangfuseConfig()
INFRA = InfraConfig()


def fresh_judge_config() -> JudgeConfig:
    """Re-read JUDGE_* env vars (picks up keys set after process start)."""
    return JudgeConfig()


def fresh_langfuse_config() -> LangfuseConfig:
    """Re-read LANGFUSE_* env vars (picks up keys set after process start)."""
    return LangfuseConfig()

# ---------------------------------------------------------------------------
# Modal infrastructure (training + inference transport)
# ---------------------------------------------------------------------------

# Split into layers so a builder failure names the guilty package instead of
# failing one giant `pip install`. `git` must come first: the Unsloth installs
# below are `git+https://...` and fail with exit 1 when git is missing.
BASE_DEPS = [
    "torch>=2.8.0",
    "triton>=3.4.0",
    "numpy",
    "safetensors",
    "bitsandbytes",
    "torchao>=0.16.0",
]

HF_DEPS = [
    "transformers==5.15.0",
    "tokenizers>=0.22.0,<=0.23.0",
    "datasets",
    "trl==1.9.2",
    "hf_xet",  # Xet-native transfers for model/dataset downloads (chunked CAS)
]

UNSLOTH_DEPS = [
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
    "unsloth[base] @ git+https://github.com/unslothai/unsloth",
]

HARNESS_DEPS = [  # eval / judge harness (CPU-only, tiny)
    "langfuse>=2.0",
    "openai>=1.0",
    "fastapi[standard]",  # required explicitly for @modal.fastapi_endpoint
]

EVAL_DEPS = [  # EleutherAI LM Evaluation Harness (reasoning / multidomain)
    "lm-eval>=0.4",  # PyPI name is lm-eval (import lm_eval)
    "peft",  # explicit: lm-eval loads Hub LoRA via model_args peft=
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(*BASE_DEPS)
    .pip_install(*HF_DEPS)
    .pip_install(*UNSLOTH_DEPS)
    .pip_install(*HARNESS_DEPS)
    .pip_install(*EVAL_DEPS)
    .run_commands("python -c 'import transformers; print(transformers.__version__)'")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(INFRA.volume_name, create_if_missing=True)

# Env passthrough: Modal functions see the same JUDGE_* / LANGFUSE_* / HF_* keys.
# HF_TOKEN is set from HF_TOKEN_NEW because huggingface_hub only reads HF_TOKEN.
MODAL_SECRETS = [
    modal.Secret.from_dict(
        {
            "JUDGE_MODEL": JUDGE.model,
            "JUDGE_API_KEY": JUDGE.api_key,
            "JUDGE_BASE_URL": JUDGE.base_url,
            "LANGFUSE_PUBLIC_KEY": LANGFUSE.public_key,
            "LANGFUSE_SECRET_KEY": LANGFUSE.secret_key,
            "LANGFUSE_HOST": LANGFUSE.host,
            "LANGFUSE_SESSION": LANGFUSE.session_id,
            "HF_DATASET": DATA.hf_id,
            "HF_TOKEN": hf_token(),
            "HF_TOKEN_NEW": hf_token(),
            "HF_REPO": MODEL.hub_repo,
            "HF_XET_HIGH_PERFORMANCE": _env("HF_XET_HIGH_PERFORMANCE", "1"),
            "REASONING_STRENGTH": DATA.reasoning_strength,
        }
    )
]

# ---------------------------------------------------------------------------
# AFM dataset helpers (pure python — safe to unit-test locally)
# ---------------------------------------------------------------------------

COA_FUNCTIONS = ("think", "plan", "tool", "observation", "reflection", "suggested_answer", "double_check", "answer")

_SENTINEL = "SENTINELANSWER"


def extract_afm_fields(row: dict) -> dict:
    """Normalise one AFM-WebAgent-RL row to ``{question, prompt_text, targets}``.

    Raw schema (from HF viewer):
      - ``prompt``: ``[{content: <CoA system + question>} ]``
      - ``reward_model``: ``{ground_truth: {target: [...]}}``
      - ``extra_info``: ``{question, answer, need_tools_kwargs, ...}``
    """
    prompt_raw = row.get("prompt", [])
    if isinstance(prompt_raw, list) and prompt_raw:
        first = prompt_raw[0]
        prompt_text = first.get("content", "") if isinstance(first, dict) else str(first)
    else:
        prompt_text = str(prompt_raw)

    reward_model = row.get("reward_model") or {}
    gt = (reward_model.get("ground_truth") or {}).get("target", [])
    targets = [str(t).strip() for t in gt] if isinstance(gt, list) else [str(gt).strip()]

    extra = row.get("extra_info") or {}
    question = extra.get("question") or row.get("question") or prompt_text[-2000:]
    return {
        "question": str(question),
        "prompt_text": str(prompt_text),
        "targets": [t for t in targets if t],
        "data_source": row.get("data_source", ""),
    }


# Compact answering contract replacing the verbatim AFM Chain-of-Agents
# instruction dump. The full 8-function contract (think/plan/tool/...)
# conflicts with Glimmer's native self/user recipient scheme and produced
# empty, <think>-style, or never-ending rollouts in training. The question
# carries the task semantics; the overlay (system turn) already permits CoA
# functions inside the private to=self reasoning.
AFM_USER_CONTRACT = (
    "Answer the question below. Reason privately, then put your final "
    "answer on the user channel and nothing else."
)
MAX_QUESTION_CHARS = 3000


def build_grpo_messages(fields: dict, reasoning_strength: str = DATA.reasoning_strength) -> list[dict]:
    """Compose the Muse Glimmer chat messages for one AFM item.

    Sends the *question* plus a compact contract — not the raw CoA
    ``prompt_text`` dump. Single-sample inference confirmed the model opens
    ``to=self`` / ``to=user`` correctly on this shape.
    """
    question = fields["question"][:MAX_QUESTION_CHARS]
    return [
        {"role": "system", "content": DATA.system_overlay},
        {"role": "user", "content": f"{AFM_USER_CONTRACT}\n\nQuestion: {question}"},
    ]


def build_grpo_row(row: dict) -> dict | None:
    """Map AFM row → TRL GRPO row (``prompt`` / ``ground_truth`` / extras).

    Returns ``None`` when the row has no usable target, so callers can filter.
    ``chat_template_kwargs`` flows through TRL into ``apply_chat_template``.
    """
    fields = extract_afm_fields(row)
    if not fields["targets"]:
        return None
    return {
        "prompt": build_grpo_messages(fields),
        "ground_truth": fields["targets"],
        # Kept for the verifiable reward + judge prompt:
        "question": fields["question"],
        "data_source": fields["data_source"],
        "chat_template_kwargs": {"reasoning_strength": DATA.reasoning_strength},
    }


# ---------------------------------------------------------------------------
# Muse Glimmer channel helpers (pure python)
# ---------------------------------------------------------------------------

# Resolved at model-load time from the tokenizer; defaults match the ATEM format
# so reward code paths work in `--dry-run` without a GPU.
ANSWER_PREFIX_DEFAULT = "<|start|>assistant to=user<|message|>"
RECIPIENT_USER = "user"
USER_CHANNEL = "to=user"


def resolve_answer_channel(tokenizer) -> tuple[str, str]:
    """Recover the reply-channel markers from the tokenizer (no hardcoding).

    Renders a probe conversation containing a sentinel and splits around it.
    Returns ``(answer_prefix, user_channel)``.
    """
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "probe"}, {"role": "assistant", "content": _SENTINEL}],
        tokenize=False,
    )
    head, _ = probe.split(_SENTINEL)
    answer_prefix = head[head.rindex("<|start|>"):]
    recipient = answer_prefix.split("to=")[1].split("<|message|>")[0].strip()
    return answer_prefix, "to=" + recipient


# Fallback for AFM Chain-of-Agents rollouts that emit `answer:` /
# `suggested_answer:` instead of opening the Glimmer `to=user` channel.
_COA_REPLY_RE = re.compile(r"(suggested_answer|double_check|answer)\s*[:\-]", re.IGNORECASE)


def split_channels(text: str | None, user_channel: str = USER_CHANNEL) -> tuple[str, str]:
    """Split a (possibly special-token-stripped) completion into (reasoning, answer).

    Primary: the reply follows the last ``to=user`` marker. Handles both raw
    ATEM (``<|message|>`` etc. present) and TRL-decoded text (markers stripped).
    Fallback: AFM Chain-of-Agents rollouts that emit ``answer:`` /
    ``suggested_answer:`` without ever opening ``to=user`` — the reply is the
    text after the last such marker.
    """
    if not text:
        return "", ""
    idx = text.rfind(user_channel)
    if idx != -1:
        reasoning = text[:idx]
        answer = text[idx + len(user_channel):]
        for marker in ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>", "assistant", "to=self"):
            reasoning = reasoning.replace(marker, " ")
        answer = answer.replace("<|message|>", "")
        for marker in ("<|eom|>", "<|eot|>", "<|start|>"):
            answer = answer.split(marker)[0]
        return reasoning.strip(), answer.strip()
    coa_matches = list(_COA_REPLY_RE.finditer(text))
    if coa_matches:
        last = coa_matches[-1]
        reasoning = text[: last.start()]
        answer = text[last.end():]
        for marker in ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>", "assistant", "to=self"):
            reasoning = reasoning.replace(marker, " ")
        for marker in ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>"):
            answer = answer.split(marker)[0]
        return reasoning.strip(), answer.strip()
    return text.strip(), ""  # never opened a reply channel in any known format


def extract_reply_for_scoring(text: str | None) -> str:
    """Best-effort reply for the judge / verifiable rewards (scoring only).

    Uses the channelled reply when markers exist; last resort is the whole
    completion stripped of leftover ATEM markers. ``split_channels`` itself
    is unchanged so ``channel_format_reward`` still gives unchannelled prose
    its -1.0 — content signal flows without eroding channel pressure.
    """
    _, answer = split_channels(text)
    if answer:
        return answer
    fallback = text or ""
    for marker in ("<|message|>", "<|eom|>", "<|eot|>", "<|start|>", "assistant", "to=self", USER_CHANNEL):
        fallback = fallback.replace(marker, " ")
    return " ".join(fallback.split())


# ---------------------------------------------------------------------------
# Verifiable reward (pure python — exact match + token F1 over AFM targets)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = _normalize(prediction).split()
    tgt_tokens = _normalize(target).split()
    if not pred_tokens or not tgt_tokens:
        return 0.0
    common = set(pred_tokens) & set(tgt_tokens)
    if not common:
        return 0.0
    # multiset overlap for repeated tokens
    overlap = sum(min(pred_tokens.count(t), tgt_tokens.count(t)) for t in common)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    return 2 * precision * recall / (precision + recall)


def verifiable_score(reply: str, targets: list[str]) -> float:
    """Score ``reply`` against AFM ground-truth targets.

    - 1.0 for normalized exact match against any target,
    - else best token-F1 (partial credit for long web-agent answers),
    - 0.0 when reply/targets are empty.
    """
    if not reply or not targets:
        return 0.0
    norm_reply = _normalize(reply)
    for target in targets:
        if norm_reply == _normalize(target) and norm_reply:
            return 1.0
    return max(token_f1(reply, t) for t in targets)


# ---------------------------------------------------------------------------
# Muse Wonder judge via Meta's API (lazy `openai` import)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are an impartial judge for web-agent answers. "
    "Given the question, the ground-truth answer(s), and the model's reply, "
    "respond with exactly one line of JSON: "
    '{"score": <0.0-1.0>, "rationale": "<one sentence>"}. '
    "Score 1.0 for semantically correct, 0.0 for wrong, partial credit otherwise."
)


def _judge_messages(question: str, targets: list[str], reply: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "ground_truth": targets, "reply": reply},
                ensure_ascii=False,
            ),
        },
    ]


class MetaJudgeClient:
    """Muse Wonder judge hitting Meta's OpenAI-compatible API.

    Auth comes from ``JUDGE_API_KEY``; model id from ``JUDGE_MODEL``. The base
    URL defaults to ``JUDGE_BASE_URL`` so teams can point at whichever Meta
    gateway they use without code changes.
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        # Default None (not JUDGE) so env keys set after import are picked up.
        self.config = config or fresh_judge_config()

    @property
    def available(self) -> bool:
        return bool(self.config.api_key and self.config.model)

    def score(self, question: str, targets: list[str], reply: str) -> tuple[float, str]:
        """Return ``(score, rationale)``.

        Transient gateway errors (404 model propagation, 429, 5xx) are
        retried immediately — pure exception retry, no sleeping, no
        post-success pause. Auth and validation errors fail fast. Anything
        still failing after ``max_retries`` raises for the caller to fall back.
        """
        from openai import OpenAI  # lazy: only needed when judging

        try:  # openai>=1.0 exception types; degrade gracefully if absent
            from openai import APIStatusError as _APIStatusError
        except ImportError:  # pragma: no cover
            _APIStatusError = Exception  # type: ignore[assignment,misc]

        # HTTP statuses worth retrying at once (propagation / throttling / outage).
        _RETRYABLE = {404, 408, 429, 500, 502, 503, 529}

        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url, timeout=self.config.timeout_s)
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=_judge_messages(question, targets, reply),  # type: ignore[arg-type]
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                break  # success — fall through to parsing below
            except _APIStatusError as exc:
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status not in _RETRYABLE or attempt >= self.config.max_retries:
                    raise
                log.warning(
                    "Judge transient error (status %s, attempt %d/%d); retrying immediately.",
                    status, attempt + 1, self.config.max_retries + 1,
                )
        else:  # pragma: no cover — loop can't exhaust without break/raise
            raise last_exc or RuntimeError("judge retries exhausted")
        content = (response.choices[0].message.content or "").strip()
        try:
            payload = json.loads(content[content.index("{"): content.rindex("}") + 1])
            score = float(payload.get("score", 0.0))
            rationale = str(payload.get("rationale", ""))
        except (ValueError, TypeError, IndexError):
            # Non-JSON judge output: salvage a float if one is present.
            log.warning("Judge returned non-JSON content: %.200s", content)
            match = re.search(r"([01](?:\.\d+)?)", content)
            score = float(match.group(1)) if match else 0.0
            rationale = content[:200]
        return max(0.0, min(1.0, score)), rationale


# ---------------------------------------------------------------------------
# Langfuse testing harness (lazy `langfuse` import)
# ---------------------------------------------------------------------------

TRACE_NAME = "glimmer-webagent-eval"


class LangfuseHarness:
    """Logs judged completions to Langfuse and exposes scores to GRPO.

    The feedback loop is: ``evaluate_completion`` → judge → ``trace.score`` →
    the returned float *is* the GRPO reward. ``fetch_recent_score`` additionally
    lets offline analysis pull the stored Langfuse score for a trace id.
    When Langfuse env keys are absent, logging is skipped and the judge (or
    verifiable) score is returned directly so training never blocks on eval infra.
    """

    def __init__(
        self,
        config: LangfuseConfig | None = None,
        judge: MetaJudgeClient | None = None,
    ) -> None:
        # Defaults None so env keys set after import are picked up.
        self.config = config or fresh_langfuse_config()
        self.judge = judge or MetaJudgeClient()
        self._client = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _get_client(self):
        if self._client is None:
            from langfuse import Langfuse  # lazy: only needed when logging

            self._client = Langfuse(
                public_key=self.config.public_key or None,
                secret_key=self.config.secret_key or None,
                host=self.config.host or None,
            )
        return self._client

    def evaluate_completion(
        self,
        *,
        question: str,
        targets: list[str],
        reply: str,
        reasoning: str = "",
        trace_id: str | None = None,
    ) -> tuple[float, str]:
        """Judge one completion, log trace+score to Langfuse.

        Returns ``(reward, rationale)`` with reward in [0,1]. The rationale
        lets step logs show *why* the judge scored as it did.
        """
        if self.judge.available:
            try:
                score, rationale = self.judge.score(question, targets, reply)
            except Exception as exc:  # judge outage → verifiable fallback
                log.warning("Judge failed (%s); falling back to verifiable score.", exc)
                score, rationale = verifiable_score(reply, targets), "judge-error-fallback"
        else:
            score, rationale = verifiable_score(reply, targets), "no-judge-key-fallback"

        if self.enabled:
            try:
                client = self._get_client()
                trace = client.trace(
                    name=TRACE_NAME,
                    session_id=self.config.session_id,
                    id=trace_id,
                    input={"question": question, "targets": targets},
                    output={"reply": reply, "reasoning": reasoning[:2000]},
                    metadata={"judge_model": self.judge.config.model},
                )
                trace.score(name="judge_score", value=score, comment=rationale[:1000])
                client.flush()
            except Exception as exc:
                log.warning("Langfuse logging failed (%s); reward still returned.", exc)
        return score, rationale

    def fetch_recent_score(self, trace_id: str) -> float | None:
        """Fetch a stored Langfuse score (for offline reward reconstruction)."""
        if not self.enabled:
            return None
        try:
            client = self._get_client()
            trace = client.fetch_trace(trace_id)
            scores = getattr(trace, "scores", None) or []
            for s in reversed(scores):
                if getattr(s, "name", "") == "judge_score":
                    return float(s.value)
        except Exception as exc:
            log.warning("Langfuse fetch failed for %s: %s", trace_id, exc)
        return None


# Module-level harness used by reward functions (re-point in tests if needed).
HARNESS = LangfuseHarness()


# ---------------------------------------------------------------------------
# GRPO reward functions (signatures must match TRL: lists in, lists out)
# ---------------------------------------------------------------------------


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion and isinstance(completion[0], dict) else ""
    return str(completion or "")


def channel_format_reward(completions, **kwargs) -> list[float]:
    """Reward opening the ``to=user`` reply channel (after private reasoning)."""
    scores: list[float] = []
    for completion in completions:
        reasoning, answer = split_channels(_completion_text(completion))
        if answer:
            scores.append(1.5 if reasoning else 1.0)
        else:
            scores.append(-1.0)
    return scores


def answer_format_reward(completions, **kwargs) -> list[float]:
    """Shape the reply: non-empty answers beat empty ones; CoA answers beat dumps."""
    scores: list[float] = []
    for completion in completions:
        _, answer = split_channels(_completion_text(completion))
        if not answer:
            scores.append(-0.5)
        elif len(_normalize(answer).split()) <= 200:
            scores.append(0.5)
        else:  # suspiciously long — likely reasoning leaked into the reply
            scores.append(0.0)
    return scores


def verifiable_ground_truth_reward(prompts, completions, ground_truth, question=None, **kwargs) -> list[float]:
    """Exact/F1 match of the ``to=user`` reply against AFM ``ground_truth``.

    Scaled ×4 so correctness dominates formatting (same ordering as the DSMath
    starter: correct ≈ +4, wrong ≈ −0.5).
    """
    scores: list[float] = []
    questions = question if isinstance(question, list) else [question] * len(completions)
    for completion, targets, _q in zip(completions, ground_truth, questions):
        reply = extract_reply_for_scoring(_completion_text(completion))
        tgt_list = targets if isinstance(targets, list) else [targets]
        scores.append(4.0 if verifiable_score(reply, [str(t) for t in tgt_list]) >= 0.99 else -0.5)
    return scores


def llm_judge_reward(prompts, completions, ground_truth, question=None, **kwargs) -> list[float]:
    """Muse Wonder judge via Meta API, logged to Langfuse, returned as reward.

    This is the GRPO feedback loop: the Langfuse testing-harness output *is*
    the reward (scaled ×4 to match the verifiable term). Judge failures fall
    back to the verifiable score so one outage cannot zero a whole step.
    """
    scores: list[float] = []
    questions = question if isinstance(question, list) else [question] * len(completions)
    for i, (completion, targets, q) in enumerate(zip(completions, ground_truth, questions)):
        text = _completion_text(completion)
        reasoning, _ = split_channels(text)
        reply = extract_reply_for_scoring(text)
        tgt_list = [str(t) for t in (targets if isinstance(targets, list) else [targets])]
        try:
            s, s_why = HARNESS.evaluate_completion(
                question=str(q or ""),
                targets=tgt_list,
                reply=reply,
                reasoning=reasoning,
            )
        except Exception as exc:
            log.warning("Harness error (%s); verifiable fallback.", exc)
            s, s_why = verifiable_score(reply, tgt_list), "harness-error-fallback"
        scores.append(4.0 * s - 0.5 * (1.0 - s))  # 1.0 → +4.0, 0.0 → −0.5
        if i < 2:  # keep logs readable: sample the first items per step
            # Raw completion dump (~500 chars): confirms whether the rollout
            # used `to=user` or a CoA `answer:` block (diagnosed empty replies).
            raw = text.replace("\n", "\\n")
            log.info("raw completion %d chars: %.500s", len(text), raw)
            log.info(
                "judge sample q=%.80s reply=%.80s score=%.2f why=%.160s",
                str(q or ""), reply, s, s_why,
            )
    return scores


REWARD_FUNCS = [
    channel_format_reward,
    answer_format_reward,
    verifiable_ground_truth_reward,
    llm_judge_reward,
]

# ---------------------------------------------------------------------------
# Modal functions — all GPU/heavy work lives here
# ---------------------------------------------------------------------------


def _remote_load_model(train_cfg: TrainConfig = TRAIN, model_cfg: ModelConfig = MODEL):
    """Load Muse Glimmer (4-bit) + fix eos + attach LoRA. Runs ONLY on Modal."""
    import os as _os

    _os.environ["UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK"] = str(train_cfg.grpo_rows_per_chunk)
    # HF auth inside the container (Modal secrets inject HF_TOKEN/HF_TOKEN_NEW;
    # re-mirror here in case only one of the two names was provided).
    _hf = _os.environ.get("HF_TOKEN") or _os.environ.get("HF_TOKEN_NEW", "")
    if _hf and not _os.environ.get("HF_TOKEN"):
        _os.environ["HF_TOKEN"] = _hf

    from unsloth import FastModel

    batch_size = train_cfg.per_device_batch
    max_seq = train_cfg.max_seq_length

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_cfg.base_repo,
        max_seq_length=max_seq,
        device_map_planner_kwargs={"retained_rows": batch_size * max_seq},
        token=_hf or None,  # authenticated download for gated repos/rate limits
    )

    # Muse Glimmer has two end tokens; TRL reads eos from the tokenizer, so
    # point it at `<|eot|>` (turn end). Otherwise rollouts never stop and every
    # completion is truncated/masked. See original starter for full rationale.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.eos_token = "<|eot|>"
    model.generation_config.eos_token_id = sorted(
        {
            text_tok.eos_token_id,
            text_tok.convert_tokens_to_ids("<|end_of_text|>"),
        }
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=model_cfg.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    global USER_CHANNEL
    try:
        _, user_channel = resolve_answer_channel(tokenizer)
        USER_CHANNEL = user_channel
    except Exception as exc:
        log.warning("Could not resolve answer channel, using default: %s", exc)
    return model, tokenizer


def _remote_load_afm(data_cfg: DataConfig = DATA, max_rows: int | None = None):
    """Load + normalise the AFM WebAgent RL dataset on the remote."""
    import os as _os

    from datasets import load_dataset

    _hf = _os.environ.get("HF_TOKEN") or _os.environ.get("HF_TOKEN_NEW", "")
    raw = load_dataset(data_cfg.hf_id, split=data_cfg.split, token=_hf or None)
    rows = [build_grpo_row(r) for r in raw]
    rows = [r for r in rows if r is not None]
    if max_rows is not None:
        rows = rows[:max_rows]
    log.info("AFM rows usable for GRPO: %d", len(rows))
    # Re-hydrate as a HF Dataset so TRL keeps column semantics (incl. lists).
    from datasets import Dataset

    prompts = [r["prompt"] for r in rows]
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "ground_truth": [r["ground_truth"] for r in rows],
            "question": [r["question"] for r in rows],
            "chat_template_kwargs": [r["chat_template_kwargs"] for r in rows],
        }
    )


@app.function(
    image=image,
    gpu=INFRA.gpu,  # RTX-PRO-6000 per spec
    timeout=INFRA.timeout_s,
    volumes={"/vol": volume},
    secrets=MODAL_SECRETS,
)
def train_grpo(
    num_generations: int = TRAIN.num_generations,
    max_steps: int = TRAIN.max_steps,
    max_seq_length: int = TRAIN.max_seq_length,
) -> str:
    """Run GRPO on Modal (RTX-PRO-6000). Invoked via ``modal run starter.py --action train``.

    Reward stack: channel format → answer shape → verifiable AFM match →
    Muse Wonder judge (logged to Langfuse). Watch ``frac_reward_zero_std``:
    if ≈1 for many steps, raise ``num_generations`` (8 on RTX-PRO-6000) so groups
    disagree and advantages are non-zero.
    """
    import dataclasses

    cfg = dataclasses.replace(TRAIN, num_generations=num_generations, max_steps=max_steps, max_seq_length=max_seq_length)
    max_prompt = DATA.max_prompt_tokens
    max_completion = cfg.max_seq_length - max_prompt

    model, tokenizer = _remote_load_model(cfg)
    dataset = _remote_load_afm()

    from trl import GRPOConfig, GRPOTrainer

    args = GRPOConfig(
        temperature=1.0,
        learning_rate=cfg.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,
        logging_steps=1,
        log_completions=False,
        per_device_train_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=1,
        num_generations=cfg.num_generations,
        max_completion_length=max_completion,
        mask_truncated_completions=True,  # truncated = no reply channel → mask, don't punish
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,  # rolling single checkpoint: new save deletes the previous
        report_to="none",
        output_dir=cfg.output_dir,
        importance_sampling_level="sequence",  # GSPO: stable on long CoA rollouts
        loss_type="dr_grpo",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[channel_format_reward, answer_format_reward, verifiable_ground_truth_reward, llm_judge_reward],
        args=args,
        train_dataset=dataset,
    )
    trainer.train()

    import torch

    peak = torch.cuda.max_memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info("peak reserved VRAM %.2f GiB / %.2f GiB", peak, total)

    model.save_pretrained(cfg.lora_out)
    tokenizer.save_pretrained(cfg.lora_out)

    # Push the LoRA adapters to the Hub (same-name policy: repeated pushes
    # overwrite the same repo). A push failure must not lose the run, so it
    # is caught — adapters are already safe on the volume at this point.
    _hub_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_NEW", "")
    try:
        model.push_to_hub(MODEL.hub_repo, token=_hub_token or None, private=MODEL.hub_private)
        tokenizer.push_to_hub(MODEL.hub_repo, token=_hub_token or None, private=MODEL.hub_private)
        log.info("Pushed LoRA to hub repo %s", MODEL.hub_repo)
    except Exception as exc:
        log.warning("Hub push to %s failed (%s); adapters remain on the volume.", MODEL.hub_repo, exc)

    volume.commit()

    return cfg.lora_out


@app.function(
    image=image,
    gpu=INFRA.gpu,
    timeout=3600,
    volumes={"/vol": volume},
    secrets=MODAL_SECRETS,
)
def generate(prompts: list[str], max_new_tokens: int = 512, temperature: float = 0.7) -> list[str]:
    """Modal-hosted inference for Muse Glimmer (+LoRA when present).

    ``prompts`` are raw user strings; the CoA system overlay + chat template +
    ``reasoning_strength`` are applied remotely so callers stay thin.
    """
    import torch

    model, tokenizer = _base_only()
    # Finetuned-first: Hub LoRA, then volume copy, then base. (The volume was
    # deleted, so the Hub repo is currently the only adapter source.)
    # NOTE: attach onto the CLEAN base — _remote_load_model() carries a fresh
    # random LoRA that would double-stack adapters and silently null the
    # trained weights (peft "multiple adapters" + "missing adapter keys").
    import os as _os

    _hub_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HF_TOKEN_NEW", "")
    try:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, MODEL.hub_repo, token=_hub_token or None)
        log.info("Loaded LoRA from hub repo %s", MODEL.hub_repo)
    except Exception as hub_exc:
        log.warning("Hub LoRA load failed (%s); trying volume.", hub_exc)
        try:
            from peft import PeftModel as _PeftModel  # type: ignore

            adapter = f"{TRAIN.lora_out}/adapter_model.safetensors"
            if _os.path.exists(adapter):
                model = _PeftModel.from_pretrained(model, TRAIN.lora_out)
                log.info("Loaded LoRA from %s", TRAIN.lora_out)
        except Exception as exc:
            log.warning("LoRA load skipped: %s", exc)

    outputs: list[str] = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": DATA.system_overlay},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_strength=DATA.reasoning_strength
        )
        inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to("cuda")
        # temperature<=0 means greedy: do_sample=False (a zero temperature with
        # sampling raises `ValueError: temperature has to be strictly positive`).
        use_sampling = temperature > 0
        gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": use_sampling}
        if use_sampling:
            gen_kwargs["temperature"] = temperature
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        outputs.append(completion)
    return outputs


def _respond(model, tokenizer, item: dict) -> dict:
    """Shared generate→split→respond tail for the serving endpoints.

    Plain helper (NOT a Modal function) so ``serve``/``serve_base`` can call
    it directly in-container.
    """
    import torch

    prompt = str(item.get("prompt", ""))
    max_new_tokens = int(item.get("max_new_tokens", 512))
    temperature = float(item.get("temperature", 0.7))
    messages = [
        {"role": "system", "content": DATA.system_overlay},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, reasoning_strength=DATA.reasoning_strength
    )
    inputs = tokenizer(text=text, return_tensors="pt", add_special_tokens=False).to("cuda")
    # temperature<=0 means greedy (see generate(): zero + sampling raises).
    use_sampling = temperature > 0
    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": use_sampling}
    if use_sampling:
        gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    reasoning, reply = split_channels(completion)
    return {"reply": reply, "reasoning": reasoning}


@app.function(
    image=image,
    gpu=INFRA.gpu,
    timeout=21600,  # 6h: full MMLU-class slices take hours on 28B
    secrets=MODAL_SECRETS,
)
def lm_harness(
    tasks: list[str], source: str = "hub", limit: int | None = None, temperature: float = 0.3
) -> dict:
    """EleutherAI LM Evaluation Harness on Modal GPU (reasoning/multidomain).

    Loads the model through OUR proven Unsloth path (lm-eval's own
    ``AutoModelForCausalLM`` loader cannot instantiate the muse-glimmer
    arch — transformers only registers it for conditional generation) and
    hands the live objects to ``HFLM`` in-process, which explicitly
    supports pre-initialized models. ``source``: ``"hub"`` (finetuned
    LoRA), ``"base"``, or ``"both"`` (sequential reloads). ``limit`` caps
    samples per task (smoke: 10). ``temperature`` applies to GENERATIVE
    tasks only (aime/ifeval); loglikelihood tasks (gpqa) never sample.
    Failures surface as ``{"error": ...}`` instead of raising.
    """
    import torch

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    _hf = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_NEW", "")
    if _hf and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = _hf

    def _run(use_lora: bool, gen_kwargs: str | None) -> dict:
        try:
            if use_lora:
                from peft import PeftModel  # type: ignore

                # Clean base (NOT _remote_load_model: its fresh random LoRA
                # double-stacks adapters and nulls the trained weights).
                base, tokenizer = _base_only()
                model = PeftModel.from_pretrained(base, MODEL.hub_repo, token=_hf or None)
                log.info("lm_harness: loaded Hub LoRA %s", MODEL.hub_repo)
            else:
                model, tokenizer = _base_only()
                log.info("lm_harness: loaded clean base weights")
            text_tok = getattr(tokenizer, "tokenizer", tokenizer)
            hflm = HFLM(
                pretrained=model,
                tokenizer=text_tok,
                backend="causal",  # decoder-style use; no Auto-class mapping involved
                batch_size=8,  # fixed small: batch autodetect OOM-probed to the edge on 96GB
                max_length=2048,  # gpqa/aime/ifeval prompts are short; bounds memory
                trust_remote_code=True,
            )
            # Warmup: one short prefill + tiny generation so cold kernels /
            # Triton autotuning settle before timed tasks (scores unaffected).
            try:
                import torch as _torch

                _dev = next(model.parameters()).device
                _probe = text_tok("Warmup.", return_tensors="pt")
                _ids = _probe["input_ids"][:, :32].to(_dev)
                with _torch.inference_mode():
                    _ = model(input_ids=_ids)
                    _ = model.generate(input_ids=_ids, max_new_tokens=16, do_sample=False)
                log.info("lm_harness: warmup done")
            except Exception as exc:
                log.warning("lm_harness warmup skipped (%s); continuing.", exc)
        except Exception as exc:
            import traceback

            log.warning("lm_harness model load failed: %s", exc)
            return {"error": traceback.format_exc()[-3000:]}
        # One task at a time: a gated/missing dataset (e.g. GPQA needs Hub
        # access approval) must not kill the whole battery.
        merged: dict = {}
        for single in tasks:
            try:
                payload = simple_evaluate(model=hflm, tasks=[single], limit=limit, gen_kwargs=gen_kwargs)
                for task, vals in payload.get("results", {}).items():
                    merged[task] = {k: v for k, v in vals.items() if not k.startswith("alias")}
            except Exception as exc:
                import traceback

                log.warning("lm_harness task %s failed: %s", single, exc)
                merged[single] = {"error": traceback.format_exc()[-1500:]}
        return merged

    out: dict = {"sources": {}}
    todo = ["hub", "base"] if source == "both" else [source]
    gen_kwargs = f"temperature={temperature},do_sample=True" if temperature > 0 else None
    for src in todo:
        out["sources"][src] = _run(use_lora=(src == "hub"), gen_kwargs=gen_kwargs)
    out["tasks"] = tasks
    out["temperature"] = temperature
    return out


@app.function(
    image=image,
    gpu=INFRA.gpu,
    timeout=3600,
    secrets=MODAL_SECRETS,
)
@modal.fastapi_endpoint(method="POST")
def serve(item: dict) -> dict:
    """Public HTTPS endpoint for the finetuned model. Hub-only LoRA loading.

    Adapters come exclusively from ``MODEL.hub_repo``
    (``pranavupadhyaya52/muse_glimmer_agentFT``) — no volume fallback.
    POST ``{"prompt": "...", "max_new_tokens": 512, "temperature": 0.7}``
    → ``{"reply": "<to=user content>", "reasoning": "<to=self content>"}``.
    Run with ``modal serve starter.py`` (dev) or ``modal deploy starter.py``.
    """
    import os as _os

    model, tokenizer = _base_only()  # clean base: no random LoRA to double-stack
    _hub_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HF_TOKEN_NEW", "")
    try:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, MODEL.hub_repo, token=_hub_token or None)
        log.info("Loaded LoRA from hub repo %s", MODEL.hub_repo)
    except Exception as exc:
        log.warning("Hub LoRA load failed (%s); serving base model.", exc)
    return _respond(model, tokenizer, item)


def _base_only(train_cfg: TrainConfig = TRAIN, model_cfg: ModelConfig = MODEL):
    """Load base weights with the eos fix but NO LoRA adapters.

    ``_remote_load_model`` always attaches fresh (random) LoRA, which would
    contaminate a base-model measurement — hence this separate loader.
    """
    import os as _os

    _os.environ["UNSLOTH_GRPO_MAX_ROWS_PER_CHUNK"] = str(train_cfg.grpo_rows_per_chunk)

    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_cfg.base_repo,
        max_seq_length=train_cfg.max_seq_length,
        device_map_planner_kwargs={
            "retained_rows": train_cfg.per_device_batch * train_cfg.max_seq_length
        },
    )
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)
    text_tok.eos_token = "<|eot|>"
    model.generation_config.eos_token_id = sorted(
        {
            text_tok.eos_token_id,
            text_tok.convert_tokens_to_ids("<|end_of_text|>"),
        }
    )
    global USER_CHANNEL
    try:
        _, user_channel = resolve_answer_channel(tokenizer)
        USER_CHANNEL = user_channel
    except Exception as exc:
        log.warning("Could not resolve answer channel, using default: %s", exc)
    return model, tokenizer


@app.function(
    image=image,
    gpu=INFRA.gpu,
    timeout=3600,
    secrets=MODAL_SECRETS,
)
@modal.fastapi_endpoint(method="POST")
def serve_base(item: dict) -> dict:
    """Base-model sibling of ``serve``: identical code path, no adapters.

    Used for fair A/B evals (base vs finetuned) at matched temperature.
    Same request/response contract as ``serve``.
    """
    model, tokenizer = _base_only()
    return _respond(model, tokenizer, item)


@app.function(image=image, timeout=7200, secrets=MODAL_SECRETS)
def evaluate(limit: int = DATA.eval_size) -> dict:
    """Langfuse testing harness: AFM eval rows → Modal inference → Wonder judge → Langfuse.

    Runs on CPU (no GPU needed); it calls ``generate`` remotely, so Muse
    Glimmer inference still uses Modal's SDK while judging uses Meta's API.
    Returns aggregate metrics; per-item traces/scores live in Langfuse.
    """
    rows = _remote_load_afm(max_rows=limit)
    questions = list(rows["question"][:limit])
    targets = list(rows["ground_truth"][:limit])

    completions: list[str] = []
    batch = 8
    for i in range(0, len(questions), batch):
        completions.extend(generate.remote(questions[i: i + batch]))

    harness = LangfuseHarness()
    scores: list[float] = []
    for q, tgt, comp in zip(questions, targets, completions):
        reasoning, _ = split_channels(comp)
        reply = extract_reply_for_scoring(comp)
        tgt_list = [str(t) for t in (tgt if isinstance(tgt, list) else [tgt])]
        s, _ = harness.evaluate_completion(question=q, targets=tgt_list, reply=reply, reasoning=reasoning)
        scores.append(s)

    metrics = {
        "n": len(scores),
        "mean_judge": sum(scores) / max(len(scores), 1),
        "pass_rate": sum(1 for s in scores if s >= 0.99) / max(len(scores), 1),
        "session": LANGFUSE.session_id,
        "judge_model": JUDGE.model,
    }
    log.info("eval metrics: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Local entrypoint (thin orchestration; heavy lifting stays on Modal)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(action: str = "train", limit: int = DATA.eval_size):
    """``modal run starter.py --action [train|evaluate|infer]``."""
    if action == "train":
        out = train_grpo.remote()
        print(f"LoRA saved to {out}")
        pull_checkpoints_local()
    elif action == "evaluate":
        print(json.dumps(evaluate.remote(limit=limit), indent=2))
    elif action == "infer":
        outs = generate.remote(["What is 48 + 48/2? Reply with the number only."], max_new_tokens=256)
        print(outs[0][:2000])
    else:
        raise ValueError(f"unknown action {action!r}; expected train|evaluate|infer")


def pull_checkpoints_local() -> None:
    """Download the final LoRA + all step checkpoints from the Modal volume.

    Runs locally after training (invoked by ``main``). Uses the Modal CLI
    (`volume get` is recursive for directories). Pulls:
      - ``muse_glimmer_lora`` → ``./muse_glimmer_lora`` (final adapters)
      - ``outputs`` → ``./outputs`` (every-5-step TRL checkpoints)
    """
    import subprocess
    import sys

    for remote, local in (("muse_glimmer_lora", "muse_glimmer_lora"), ("outputs", "outputs")):
        cmd = [sys.executable, "-m", "modal", "volume", "get", "--force", INFRA.volume_name, remote, local]
        print(f"Pulling {INFRA.volume_name}/{remote} -> ./{local} (overwrite) ...")
        try:
            subprocess.run(cmd, check=True)
            print(f"Pulled {remote} OK")
        except subprocess.CalledProcessError as exc:
            print(f"Pull of {remote} failed ({exc}); retry manually with: {' '.join(cmd)}")


def dry_run() -> None:
    """Validate pure-python paths locally (no GPU / Modal / keys needed)."""
    sample_row = {
        "prompt": [{"content": "You can only use the following 8 functions: think, plan, tool... Q: sky color?"}],
        "reward_model": {"ground_truth": {"target": ["blue"]}},
        "extra_info": {"question": "What color is the sky?", "answer": ["blue"]},
        "data_source": "taskcraft",
    }
    fields = extract_afm_fields(sample_row)
    assert fields["targets"] == ["blue"], fields
    grpo_row = build_grpo_row(sample_row)
    assert grpo_row and grpo_row["ground_truth"] == ["blue"]
    # Prompt-stack simplification: user turn carries the question + compact
    # contract, never the raw CoA instruction dump.
    user_turn = grpo_row["prompt"][1]["content"]
    assert "What color is the sky?" in user_turn, user_turn
    assert "8 functions" not in user_turn, user_turn

    fake = f" to=self Let me think.<|eom|>{ANSWER_PREFIX_DEFAULT}blue<|eot|>"
    reasoning, reply = split_channels(fake)
    assert reply == "blue", (reasoning, reply)
    # CoA fallback: `answer:` block without any `to=user` marker.
    coa_fake = "think: the sky scatters blue light. answer: blue"
    coa_reasoning, coa_reply = split_channels(coa_fake)
    assert coa_reply == "blue", (coa_reasoning, coa_reply)
    assert channel_format_reward([coa_fake])[0] > 0
    assert verifiable_score("Blue.", ["blue"]) == 1.0
    assert 0.0 < verifiable_score("light blue sky", ["blue"]) <= 1.0

    assert channel_format_reward([fake]) == [1.5]
    # Last-resort scoring fallback: unchannelled prose keeps its -1.0 format
    # penalty but still yields its full text for judge/verifiable scoring.
    prose = "Pernod Ricard owns Oyster Bay and Barossa Valley Estate."
    assert split_channels(prose)[1] == ""
    assert channel_format_reward([prose]) == [-1.0]
    assert extract_reply_for_scoring(prose) == prose
    assert verifiable_ground_truth_reward([None], [fake], [["blue"]]) == [4.0]
    # Judge without keys → verifiable fallback path, still returns shaped rewards.
    assert llm_judge_reward([None], [fake], [["blue"]], question=["sky?"])[0] > 0

    for cfg_name, cfg in (("MODEL", MODEL), ("DATA", DATA), ("TRAIN", TRAIN)):
        log.info("%s: %s", cfg_name, cfg)
    log.info("JUDGE model=%s base=%s key_set=%s", JUDGE.model, JUDGE.base_url, bool(JUDGE.api_key))
    log.info("HF token_set=%s (HF_TOKEN_NEW/HF_TOKEN)", bool(hf_token()))
    log.info("LANGFUSE enabled=%s host=%s", LANGFUSE.enabled, LANGFUSE.host or "<unset>")
    log.info("dry-run OK — configs, AFM mapping, channels, and rewards all validate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Muse Glimmer GRPO starter (Modal + Langfuse + Meta judge)")
    parser.add_argument("--dry-run", action="store_true", help="validate configs/rewards locally, no GPU needed")
    parser.add_argument("--action", default="train", choices=["train", "evaluate", "infer"])
    parser.add_argument("--limit", type=int, default=DATA.eval_size)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        print("Local GPU training is unsupported; use: modal run starter.py --action", args.action)
