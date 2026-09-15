"""
ArbiterOS / pi-cost-down style runtime history prune for trae_agent.

Ported from pi-agent-strategy (core.ts). Mutates only tool-result carriers in
MessageManager.steps — never assistant tool calls, user prompts, or think text.

Entry point: perform_analysis_step_arbiteros(mgr, analysis_args)
Wired as MODE=arbiteros in traj_analyzer.py.
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass, field, replace

import tiktoken

if typing.TYPE_CHECKING:
    from .expert import MessageManager

CAPSULE_PREFIX = "[arbiteros-cost-down:"

ERROR_RE = re.compile(
    r"(?:error|failed|failure|exception|traceback|not found|no tests ran|timed?\s*out)",
    re.I,
)
USEFUL_LINE_RE = re.compile(
    r"(?:error|failed|exception|assert|expected|actual|warning|summary|passed|"
    r"tests?\s+(?:ran|failed)|exit\s+code)",
    re.I,
)
PATH_RE = re.compile(
    r"(?:^|\s)((?:\.?\.?\/)?[\w@.+-]+(?:\/[\w@.+-]+)+\.[A-Za-z0-9_-]+)"
)
FOCUS_TERM_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff-]{4,}")
FOCUS_STOPWORDS = frozenset(
    "this that with from have will please 可以 一个 这个 然后".split()
)
ARTIFACT_ARG_KEY_RE = re.compile(r"(?:path|file|directory|command|query|pattern)", re.I)
ARTIFACT_TOOLS = frozenset({"str_replace_editor", "read", "grep", "find", "ls", "edit", "write"})

_token_encoding = tiktoken.encoding_for_model("gpt-4o")


def count_token(s: str) -> int:
    return len(_token_encoding.encode(s))


@dataclass
class CostDownConfig:
    mode: str = "balanced"  # off | balanced | aggressive
    min_context_chars: int = 24_000
    min_carrier_chars: int = 1_200
    min_saved_chars: int = 600
    protect_recent_messages: int = 6
    keep_ratio: float = 0.7
    drop_ratio: float = 0.25
    target_ratio: float = 0.22


@dataclass
class CostDownStats:
    before_chars: int = 0
    after_chars: int = 0
    saved_chars: int = 0
    kept: int = 0
    compressed: int = 0
    folded: int = 0
    mode: str = "balanced"


@dataclass
class CallInfo:
    name: str
    artifact_keys: list[str] = field(default_factory=list)


DEFAULT_CONFIG = CostDownConfig()


def _strings_from(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for child in value:
            out.extend(_strings_from(child))
        return out
    if isinstance(value, dict):
        out = []
        for key, child in value.items():
            if ARTIFACT_ARG_KEY_RE.search(str(key)):
                out.extend(_strings_from(child))
        return out
    return []


def _normalize_artifact(value: str) -> str | None:
    clean = value.strip().strip("'\"")
    if not clean or len(clean) > 400 or clean.startswith("-") or "\n" in clean:
        return None
    if clean.startswith("./"):
        clean = clean[2:]
    return clean


def artifact_keys(name: str, args: object) -> list[str]:
    direct = [k for k in (_normalize_artifact(s) for s in _strings_from(args)) if k]
    command_paths: list[str] = []
    for value in direct:
        for match in PATH_RE.finditer(value):
            path = match.group(1)
            if path.startswith("./"):
                path = path[2:]
            command_paths.append(path)
    keys = direct if name in ARTIFACT_TOOLS else command_paths
    return sorted(set(keys))


def _parse_tool_args(raw: object) -> object:
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {"command": raw}
    return {}


def _call_info_from_assistant(msg: dict) -> dict[str, CallInfo]:
    calls: dict[str, CallInfo] = {}
    for tool_call in msg.get("tool_calls") or []:
        tc_id = tool_call.get("id")
        if not tc_id:
            continue
        fn = tool_call.get("function") or {}
        name = fn.get("name") or "tool"
        args = _parse_tool_args(fn.get("arguments", "{}"))
        calls[tc_id] = CallInfo(name=name, artifact_keys=artifact_keys(name, args))
    return calls


def _call_info_from_tool_msg(msg: dict, by_id: dict[str, CallInfo]) -> CallInfo | None:
    tc_id = str(msg.get("tool_call_id") or "")
    if tc_id and tc_id in by_id:
        return by_id[tc_id]

    caller = msg.get("agent_caller")
    if not caller or not isinstance(caller, (tuple, list)) or len(caller) < 1:
        return None
    name = caller[0] or "tool"
    args = caller[1] if len(caller) > 1 else {}
    if name == "invalid":
        return CallInfo(name=name, artifact_keys=[])
    return CallInfo(name=str(name), artifact_keys=artifact_keys(str(name), args or {}))


def _content_chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return 0


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _set_content_text(msg: dict, text: str) -> None:
    content = msg.get("content")
    if isinstance(content, str) or content is None:
        msg["content"] = text
        return
    if isinstance(content, list):
        applied = False
        new_blocks = []
        for block in content:
            if (
                not applied
                and isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].strip()
            ):
                new_blocks.append({**block, "text": text})
                applied = True
            else:
                new_blocks.append(block)
        if not applied:
            new_blocks.append({"type": "text", "text": text})
        msg["content"] = new_blocks
        return
    msg["content"] = text


def focus_terms_from_text(text: str) -> set[str]:
    terms = [
        t
        for t in FOCUS_TERM_RE.findall(text.lower())
        if t not in FOCUS_STOPWORDS
    ]
    return set(terms[-40:])


def semantic_compress(text: str, target_ratio: float, focus: set[str]) -> str:
    lines = text.split("\n")
    budget = max(500, int(len(text) * target_ratio))
    chosen: set[int] = set()

    def add_range(center: int, radius: int) -> None:
        for i in range(max(0, center - radius), min(len(lines) - 1, center + radius) + 1):
            chosen.add(i)

    for i in range(min(len(lines), 12)):
        chosen.add(i)
    for i in range(max(12, len(lines) - 8), len(lines)):
        chosen.add(i)

    for i, line in enumerate(lines):
        lower = line.lower()
        if USEFUL_LINE_RE.search(line) or any(term in lower for term in focus):
            add_range(i, 2)

    used = 0
    output: list[str] = []
    previous = -2
    for index in sorted(chosen):
        line = lines[index]
        if used + len(line) + 1 > budget and len(output) > 8:
            continue
        if index > previous + 1:
            output.append(f"... {index - previous - 1} lines omitted ...")
        output.append(line)
        used += len(line) + 1
        previous = index

    return f"{CAPSULE_PREFIX}semantic-capsule original_chars={len(text)}]\n" + "\n".join(output)


def transient_failure(text: str) -> str:
    lines = text.split("\n")
    useful = [line for line in lines if USEFUL_LINE_RE.search(line)][:12]
    body = "\n".join(useful) if useful else "\n".join(lines[:8])
    return (
        f"{CAPSULE_PREFIX}transient-failure original_chars={len(text)}]\n"
        f"{body}\n"
        "[This failed attempt is not evidence about the current workspace state.]"
    )


def folded_artifact(text: str, keys: list[str]) -> str:
    return (
        f"{CAPSULE_PREFIX}superseded-artifact original_chars={len(text)}]\n"
        f"A later observation supersedes this output for: {', '.join(keys)}. "
        "Consult the later canonical result."
    )


def score_carrier(
    index: int,
    total: int,
    info: CallInfo | None,
    text: str,
    latest: dict[str, int],
    focus: set[str],
) -> float:
    age = total - 1 - index
    score = max(0.08, 0.82 - age * 0.065)
    if age < 4:
        score = max(score, 0.86)
    if info and any(latest.get(key) == index for key in info.artifact_keys):
        score = max(score, 0.9)
    if ERROR_RE.search(text):
        score += 0.1 if age < 5 else -0.28
    lower = text.lower()
    if any(term in lower for term in focus):
        score += 0.18
    return max(0.0, min(1.0, score))


def config_from_analysis_args(analysis_args: dict | None = None) -> CostDownConfig:
    args = analysis_args or {}
    mode = str(args.get("costdown_mode", args.get("arbiteros_mode", "balanced"))).strip()
    if mode not in ("off", "balanced", "aggressive"):
        mode = "balanced"

    cfg = replace(DEFAULT_CONFIG, mode=mode)

    if "min_context_chars" in args:
        cfg.min_context_chars = int(args["min_context_chars"])
    if "min_carrier_chars" in args:
        cfg.min_carrier_chars = int(args["min_carrier_chars"])
    if "min_saved_chars" in args:
        cfg.min_saved_chars = int(args["min_saved_chars"])
    if "protect_recent_messages" in args:
        cfg.protect_recent_messages = int(args["protect_recent_messages"])
    if "keep_ratio" in args:
        cfg.keep_ratio = float(args["keep_ratio"])
    if "drop_ratio" in args:
        cfg.drop_ratio = float(args["drop_ratio"])
    if "target_ratio" in args:
        cfg.target_ratio = float(args["target_ratio"])

    if mode == "aggressive":
        if "min_context_chars" not in args:
            cfg.min_context_chars = 0
        if "target_ratio" not in args:
            cfg.target_ratio = 0.14
        if "protect_recent_messages" not in args:
            cfg.protect_recent_messages = 4

    return cfg


def _flatten_messages(mgr: MessageManager) -> tuple[list[dict], list[tuple[int, int] | None]]:
    """Return (flat messages, refs). refs[i] is (step_idx, msg_idx) for mutable tool msgs."""
    flat: list[dict] = [mgr.user_message]
    refs: list[tuple[int, int] | None] = [None]

    for step_i, step in enumerate(mgr.steps):
        for msg_i, msg in enumerate(step):
            flat.append(msg)
            if msg.get("role") == "tool":
                refs.append((step_i, msg_i))
            else:
                refs.append(None)
    return flat, refs


def optimize_message_manager(
    mgr: MessageManager,
    config: CostDownConfig,
) -> CostDownStats:
    flat, refs = _flatten_messages(mgr)
    before_chars = sum(_content_chars(m.get("content")) for m in flat)
    stats = CostDownStats(before_chars=before_chars, after_chars=before_chars, mode=config.mode)

    if config.mode == "off":
        return stats
    if config.mode == "balanced" and before_chars < config.min_context_chars:
        print(
            f"!!! arbiteros skip (context too small) {before_chars} < {config.min_context_chars}"
        )
        return stats

    by_id: dict[str, CallInfo] = {}
    for msg in flat:
        if msg.get("role") == "assistant":
            by_id.update(_call_info_from_assistant(msg))

    focus_src = _content_text(mgr.user_message.get("content"))
    # also pull last user-role reminders if any appear in steps
    user_texts = [
        _content_text(m.get("content"))
        for m in flat
        if m.get("role") == "user"
    ][-2:]
    focus = focus_terms_from_text(" ".join(user_texts) if user_texts else focus_src)

    latest: dict[str, int] = {}
    infos: list[CallInfo | None] = []
    for index, msg in enumerate(flat):
        if msg.get("role") != "tool":
            infos.append(None)
            continue
        info = _call_info_from_tool_msg(msg, by_id)
        infos.append(info)
        for key in info.artifact_keys if info else []:
            latest[key] = index

    protected_from = max(0, len(flat) - config.protect_recent_messages)
    erase_in = 0
    erase_out = 0
    erase_count = 0
    seen_tokens = 0

    for index, msg in enumerate(flat):
        if msg.get("role") != "tool" or index >= protected_from:
            continue
        if refs[index] is None:
            continue

        original = _content_text(msg.get("content"))
        if len(original) < config.min_carrier_chars or original.startswith(CAPSULE_PREFIX):
            stats.kept += 1
            continue

        if not msg.get("agent_arbiteros_seen"):
            seen_tokens += count_token(original)
            msg["agent_arbiteros_seen"] = True

        info = infos[index]
        superseded = bool(
            info
            and info.artifact_keys
            and all((latest.get(key, index) > index) for key in info.artifact_keys)
        )

        replacement: str | None = None
        if superseded:
            replacement = folded_artifact(original, info.artifact_keys if info else [])
            stats.folded += 1
        else:
            score = score_carrier(index, len(flat), info, original, latest, focus)
            if score < config.drop_ratio and ERROR_RE.search(original):
                replacement = transient_failure(original)
            elif score < config.keep_ratio:
                replacement = semantic_compress(original, config.target_ratio, focus)
            else:
                stats.kept += 1

        if not replacement or len(original) - len(replacement) < config.min_saved_chars:
            continue

        _set_content_text(msg, replacement)
        if "agent_arbiteros_orig" not in msg:
            msg["agent_arbiteros_orig"] = original

        erase_in += count_token(original)
        erase_out += count_token(replacement)
        erase_count += 1
        if not superseded:
            stats.compressed += 1

        print(
            f"!!! arbiteros {'fold' if superseded else 'compress'} "
            f"msg#{index} {len(original)} -> {len(replacement)} chars"
        )

    after_chars = sum(_content_chars(m.get("content")) for m in flat)
    stats.after_chars = after_chars
    stats.saved_chars = before_chars - after_chars

    mgr.metrics["seen_tokens"] += seen_tokens
    if erase_count:
        mgr.metrics["erase_tot_count"] += erase_count
        mgr.metrics["erase_in_tokens"] += erase_in
        mgr.metrics["erase_out_tokens"] += erase_out

    return stats


def perform_analysis_step_arbiteros(
    mgr: MessageManager,
    analysis_args: dict | None = None,
) -> CostDownStats:
    """Runtime hook compatible with traj_analyzer MODE dispatch."""
    config = config_from_analysis_args(analysis_args)
    print(
        f"-- arbiteros config: mode={config.mode} protect={config.protect_recent_messages} "
        f"min_ctx={config.min_context_chars} target={config.target_ratio}"
    )

    stats = optimize_message_manager(mgr, config)
    print(
        f"-- arbiteros stats: {stats.before_chars} -> {stats.after_chars} "
        f"(saved {stats.saved_chars}, folded={stats.folded}, "
        f"compressed={stats.compressed}, kept={stats.kept})"
    )
    return stats
