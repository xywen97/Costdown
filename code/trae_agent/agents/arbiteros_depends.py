"""
ArbiterOS depends_on + heuristic compress hybrid.

Identity / depends_on (from the simplified depends experiment):
  - Step ids: user, step_0, step_1, ...
  - System prompt gets depends_on rules (no REF on system)
  - System stamps [ARBITEROS_REF step_N] on each step's thinking
  - Model cites priors via [depends_on ...]

Prune policy (hybrid — default):
  - Keep set = {user, current step} ∪ current step's *direct* depends_on only
    (no protect_first / protect_recent / hop expansion). Steps no longer cited
    by the latest depends_on become compressible.
  - On-frontier steps: KEEP tool outputs intact (success-oriented)
  - Off-frontier steps: do NOT delete the whole step; heuristically
    compress/fold tool results only (cost-oriented), reusing arbiteros.py
  - Optional legacy whole-step DROP via off_frontier_action=drop

Prune policy (arbiteros_depends):
  - Build a live frontier from protect windows + depends_on (+ shallow hops)

Modes: arbiteros_depends | arbiteros_hybrid (shared compress/drop path;
hybrid uses the direct-depends keep rule above).
"""

from __future__ import annotations

import json
import os
import re
import typing
from dataclasses import dataclass

import tiktoken

from . import arbiteros as _heur

if typing.TYPE_CHECKING:
    from .expert import MessageManager

# ---------------------------------------------------------------------------
# System-prompt rule (only injection into main agent instructions)
# ---------------------------------------------------------------------------

DEPENDS_ON_SYSTEM_RULES = """
## Causal step dependency (required)

Work in ReAct steps. One step = your thinking + tool call(s) + tool result(s) for that turn.
Each step has a single short id: `user` (the initial user prompt) or `step_0`, `step_1`, ...
The system prompt has no id and must not be cited.

**Ids are assigned by the system.** History already starts with `[ARBITEROS_REF <id>]`.
Do **not** invent ids, and do **not** emit an `[ARBITEROS_REF ...]` line for your current turn.

**Every assistant turn you MUST put a non-empty `content` string** (even when you also issue
tool calls / also have private reasoning). Start `content` with exactly one depends line:

[depends_on user step_3 step_7]

(or the equivalent form `[depends_on <ARBITEROS_REF user> <ARBITEROS_REF step_3>]`)

### What to cite
- Cite **every** prior id whose content you actually use this turn: the issue text (`user`),
  a file you previously read, a failing test log, an edit result, etc.
- Cite the step that **produced** the evidence, not merely the immediately previous step.
- **Do not** habitually write only `[depends_on step_{N-1}]`. That is almost always wrong when
  you still rely on `user` or older tool outputs.
- If you use the bug description / acceptance criteria, include `user`.
- If you use code or logs from several earlier steps, list **all** of those step ids.
- Use `[depends_on ]` only on the first exploratory turn when nothing prior is needed.

Valid ids: `user` and earlier `step_*` only.
""".strip()


REF_LINE_RE = re.compile(
    r"^\[ARBITEROS_REF\s+(?P<id>user|step_\d+)\]\s*",
    re.I,
)
DEPENDS_ON_RE = re.compile(r"\[depends_on([^\]]*)\]", re.I)
REF_TOKEN_RE = re.compile(r"<ARBITEROS_REF\s+([^>\]]+)>", re.I)
BARE_ID_RE = re.compile(r"\b(user|step_\d+)\b", re.I)

DROP_PLACEHOLDER = "(System reminder: step omitted — not on dependency frontier)"
HYBRID_MODES = frozenset({"arbiteros_depends", "arbiteros_hybrid"})

_token_encoding = tiktoken.encoding_for_model("gpt-4o")


def count_token(s: str) -> int:
    return len(_token_encoding.encode(s or ""))


def _mode_name(analysis_args: dict | None = None) -> str:
    if analysis_args is not None:
        return str(analysis_args.get("mode", "")).strip()
    try:
        raw = os.environ.get("TRAJ_ANALYSIS", "").strip()
        if not raw:
            return ""
        return str(json.loads(raw).get("mode", "")).strip()
    except Exception:
        return ""


def is_enabled(analysis_args: dict | None = None) -> bool:
    return _mode_name(analysis_args) in HYBRID_MODES


@dataclass
class DependsConfig:
    mode: str = ""  # arbiteros_hybrid | arbiteros_depends | ...
    protect_first_steps: int = 2
    protect_recent_steps: int = 3
    min_step_chars: int = 400  # only for legacy drop
    frontier_hops: int = 2  # expand depends_on citations this many hops
    # compress (default, hybrid) | drop (legacy whole-step delete)
    off_frontier_action: str = "compress"
    # heuristic compress knobs (off-frontier tool results only)
    min_carrier_chars: int = 800
    min_saved_chars: int = 400
    keep_ratio: float = 0.55
    drop_ratio: float = 0.25
    target_ratio: float = 0.18
    off_frontier_target_ratio: float = 0.12  # more aggressive off frontier


@dataclass
class DependsStats:
    before_chars: int = 0
    after_chars: int = 0
    saved_chars: int = 0
    kept: int = 0
    dropped: int = 0
    compressed: int = 0
    folded: int = 0
    frontier_size: int = 0


def config_from_analysis_args(analysis_args: dict | None = None) -> DependsConfig:
    args = analysis_args or {}
    mode = _mode_name(args) or str(args.get("mode", "")).strip()
    cfg = DependsConfig(mode=mode)

    # hybrid mode defaults to compress; pure depends can still request drop
    if mode == "arbiteros_hybrid":
        cfg.off_frontier_action = "compress"
        cfg.protect_recent_steps = 3
        cfg.frontier_hops = 2
    elif mode == "arbiteros_depends":
        # keep backward-compatible drop unless explicitly overridden
        cfg.off_frontier_action = str(args.get("off_frontier_action", "drop")).strip()

    if "protect_first_steps" in args:
        cfg.protect_first_steps = int(args["protect_first_steps"])
    if "protect_recent_steps" in args:
        cfg.protect_recent_steps = int(args["protect_recent_steps"])
    if "min_step_chars" in args:
        cfg.min_step_chars = int(args["min_step_chars"])
    if "frontier_hops" in args:
        cfg.frontier_hops = int(args["frontier_hops"])
    if "off_frontier_action" in args:
        cfg.off_frontier_action = str(args["off_frontier_action"]).strip()
    if cfg.off_frontier_action not in ("compress", "drop"):
        cfg.off_frontier_action = "compress"

    if "min_carrier_chars" in args:
        cfg.min_carrier_chars = int(args["min_carrier_chars"])
    if "min_saved_chars" in args:
        cfg.min_saved_chars = int(args["min_saved_chars"])
    if "keep_ratio" in args:
        cfg.keep_ratio = float(args["keep_ratio"])
    if "drop_ratio" in args:
        cfg.drop_ratio = float(args["drop_ratio"])
    if "target_ratio" in args:
        cfg.target_ratio = float(args["target_ratio"])
    if "off_frontier_target_ratio" in args:
        cfg.off_frontier_target_ratio = float(args["off_frontier_target_ratio"])
    return cfg


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------


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


def step_id(step_idx: int) -> str:
    return f"step_{step_idx}"


def strip_leading_ref(text: str) -> tuple[str | None, str]:
    m = REF_LINE_RE.match(text or "")
    if not m:
        return None, text or ""
    return m.group("id"), text[m.end() :]


def ensure_leading_ref(text: str, ref_id: str) -> str:
    existing, rest = strip_leading_ref(text or "")
    if existing == ref_id:
        # normalize to single canonical first line
        return f"[ARBITEROS_REF {ref_id}]\n{rest.lstrip()}"
    return f"[ARBITEROS_REF {ref_id}]\n{rest.lstrip()}"


def parse_depends_on(text: str) -> list[str]:
    """Parse ids from [depends_on ...] supporting bare ids and <ARBITEROS_REF id>."""
    if not text:
        return []
    _, after_ref = strip_leading_ref(text)
    m = DEPENDS_ON_RE.search(text) or DEPENDS_ON_RE.search(after_ref)
    if not m:
        return []
    body = m.group(1) or ""
    ids: list[str] = []
    for tok in REF_TOKEN_RE.findall(body):
        tid = tok.strip()
        if re.fullmatch(r"user|step_\d+", tid, flags=re.I):
            ids.append(tid.lower() if tid.lower() == "user" else tid)
    for tok in BARE_ID_RE.findall(body):
        tid = tok.strip()
        if tid.lower() == "user":
            ids.append("user")
        else:
            ids.append(tid)
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        # normalize step_N casing
        if i.lower() == "user":
            i = "user"
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def collect_depends_on(assistant: dict) -> list[str]:
    """Parse depends_on from content, falling back to reasoning_content."""
    content = _content_text(assistant.get("content"))
    deps = parse_depends_on(content)
    if deps:
        return deps
    rc = assistant.get("reasoning_content") or ""
    if isinstance(rc, str) and rc.strip():
        return parse_depends_on(rc)
    return []


def step_text_chars(step: list[dict]) -> int:
    total = 0
    for msg in step:
        total += len(_content_text(msg.get("content")))
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                args = (tc.get("function") or {}).get("arguments") or ""
                total += len(str(args))
    return total


# ---------------------------------------------------------------------------
# Identity stamp (step-level)
# ---------------------------------------------------------------------------


def ensure_user_ref(mgr: MessageManager) -> None:
    text = _content_text(mgr.user_message.get("content"))
    stamped = ensure_leading_ref(text, "user")
    if stamped != text:
        _set_content_text(mgr.user_message, stamped)
    mgr.user_message["agent_step_id"] = "user"


def ensure_step_ref(mgr: MessageManager, step_idx: int, *, log_raw: bool = False) -> None:
    """Stamp [ARBITEROS_REF step_N] at the start of that step's assistant thinking."""
    if step_idx < 0 or step_idx >= len(mgr.steps):
        return
    step = mgr.steps[step_idx]
    rid = step_id(step_idx)
    for msg in step:
        msg["agent_step_id"] = rid
    assistant = next((m for m in step if m.get("role") == "assistant"), None)
    if not assistant:
        return

    raw_content = _content_text(assistant.get("content"))
    raw_reasoning = assistant.get("reasoning_content") or ""
    # Skip re-logging dropped placeholders / already-snapshotted steps.
    if assistant.get("agent_depends_dropped"):
        return
    if "agent_depends_raw_content" not in assistant:
        assistant["agent_depends_raw_content"] = raw_content
        assistant["agent_depends_raw_reasoning"] = raw_reasoning
        if log_raw or not assistant.get("agent_depends_raw_logged"):
            injected_empty = not raw_content.strip()
            print(
                f"-- arbiteros_depends pre-stamp {rid}: "
                f"content_empty={injected_empty} "
                f"content={raw_content!r} "
                f"reasoning_content={raw_reasoning!r}"
            )
            assistant["agent_depends_raw_logged"] = True

    text = raw_content
    # If model omitted thinking entirely, force a depends_on carrier; system stamps REF.
    if not text.strip():
        text = "[depends_on ]"
        assistant["agent_depends_injected_empty"] = True
    else:
        assistant.setdefault("agent_depends_injected_empty", False)
    # Drop a model-emitted leading REF if any — system owns the label.
    _maybe_ref, rest = strip_leading_ref(text)
    if _maybe_ref is not None:
        text = rest.lstrip()
    text = ensure_leading_ref(text, rid)
    _set_content_text(assistant, text)
    deps = collect_depends_on(assistant)
    # Prefer parsing the pre-stamp raw forms (bare ids) before REF rewrite.
    if not deps:
        deps = parse_depends_on(raw_content) or parse_depends_on(raw_reasoning)
    assistant["agent_depends_on"] = deps
    if log_raw or assistant.get("agent_depends_raw_logged"):
        # print parsed result once per step
        if not assistant.get("agent_depends_parsed_logged"):
            print(f"-- arbiteros_depends parsed {rid}: depends_on={deps}")
            assistant["agent_depends_parsed_logged"] = True


def stamp_all_steps(mgr: MessageManager) -> None:
    ensure_user_ref(mgr)
    for i in range(len(mgr.steps)):
        ensure_step_ref(mgr, i)


# ---------------------------------------------------------------------------
# Frontier + DROP (step unit)
# ---------------------------------------------------------------------------


def _assistant_depends(mgr: MessageManager, step_idx: int) -> list[str]:
    """Parse/cache depends_on for one step's assistant message."""
    if step_idx < 0 or step_idx >= len(mgr.steps):
        return []
    assistant = next(
        (m for m in mgr.steps[step_idx] if m.get("role") == "assistant"), None
    )
    if not assistant:
        return []
    deps = assistant.get("agent_depends_on")
    if deps is None:
        deps = collect_depends_on(assistant)
        assistant["agent_depends_on"] = deps
    return list(deps or [])


def build_keep_set(mgr: MessageManager, config: DependsConfig) -> set[str]:
    n = len(mgr.steps)
    keep: set[str] = {"user"}
    if n == 0:
        return keep

    # Hybrid: keep only the current step + its *direct* depends_on (and user).
    # Steps no longer cited by the latest depends_on become compressible.
    # Example: step_5 depends_on=[user, step_2, step_3, step_4]
    #   -> keep={user, step_2, step_3, step_4, step_5}  (not step_0/step_1)
    if config.mode == "arbiteros_hybrid":
        cur = n - 1
        keep.add(step_id(cur))
        for d in _assistant_depends(mgr, cur):
            keep.add(d)
        return keep

    first_n = min(config.protect_first_steps, n)
    recent_from = max(0, n - config.protect_recent_steps)

    for i in range(first_n):
        keep.add(step_id(i))
    for i in range(recent_from, n):
        keep.add(step_id(i))

    # Direct citations from protected seeds
    seed_steps = set(range(first_n)) | set(range(recent_from, n))
    edge_index: dict[str, list[str]] = {}
    for i in range(n):
        deps = _assistant_depends(mgr, i)
        edge_index[step_id(i)] = deps
        if i in seed_steps:
            for d in deps:
                keep.add(d)

    # Shallow transitive expansion along depends_on (helps "only previous step" chains)
    hops = max(0, int(config.frontier_hops))
    frontier = set(keep)
    for _ in range(hops):
        nxt: set[str] = set()
        for sid in frontier:
            for d in edge_index.get(sid, ()):
                if d not in keep:
                    nxt.add(d)
        if not nxt:
            break
        keep |= nxt
        frontier = nxt

    return keep


def drop_step(mgr: MessageManager, step_idx: int) -> tuple[int, int]:
    """Legacy: replace whole step with a short placeholder."""
    step = mgr.steps[step_idx]
    rid = step_id(step_idx)
    orig_parts = []
    for msg in step:
        orig_parts.append(_content_text(msg.get("content")))
    orig = "\n".join(p for p in orig_parts if p)
    placeholder = f"[ARBITEROS_REF {rid}]\n{DROP_PLACEHOLDER}"
    mgr.steps[step_idx] = [
        {
            "role": "assistant",
            "content": placeholder,
            "agent_step_id": rid,
            "agent_depends_on": [],
            "agent_erased": orig,
            "agent_depends_dropped": True,
        }
    ]
    return count_token(orig), count_token(placeholder)


def apply_step_drops(
    mgr: MessageManager,
    keep: set[str],
    config: DependsConfig,
    stats: DependsStats,
) -> None:
    erase_in = erase_out = erase_count = 0
    seen_tokens = 0

    for i, step in enumerate(mgr.steps):
        rid = step_id(i)
        if step and step[0].get("agent_depends_dropped"):
            stats.kept += 1
            continue

        chars = step_text_chars(step)
        if not step[0].get("agent_depends_seen"):
            seen_tokens += count_token(
                "\n".join(_content_text(m.get("content")) for m in step)
            )
            for m in step:
                m["agent_depends_seen"] = True

        if rid in keep:
            stats.kept += 1
            continue
        if chars < config.min_step_chars:
            stats.kept += 1
            continue

        tin, tout = drop_step(mgr, i)
        stats.dropped += 1
        erase_in += tin
        erase_out += tout
        erase_count += 1
        print(f"!!! arbiteros_depends DROP {rid} chars≈{chars}")

    mgr.metrics["seen_tokens"] += seen_tokens
    if erase_count:
        mgr.metrics["erase_tot_count"] += erase_count
        mgr.metrics["erase_in_tokens"] += erase_in
        mgr.metrics["erase_out_tokens"] += erase_out


def apply_off_frontier_heuristic_compress(
    mgr: MessageManager,
    keep: set[str],
    config: DependsConfig,
    stats: DependsStats,
) -> None:
    """
    Off-frontier: compress/fold tool results only (reuse arbiteros heuristics).
    On-frontier: leave tool outputs intact.
    Never deletes whole steps / assistant tool_calls / thinking structure.
    """
    # Flatten for artifact latest-index scoring (same as arbiteros)
    flat: list[dict] = [mgr.user_message]
    step_of: list[int | None] = [None]
    for step_i, step in enumerate(mgr.steps):
        for msg in step:
            flat.append(msg)
            step_of.append(step_i)

    by_id: dict[str, _heur.CallInfo] = {}
    for msg in flat:
        if msg.get("role") == "assistant":
            by_id.update(_heur._call_info_from_assistant(msg))

    focus_src = _content_text(mgr.user_message.get("content"))
    user_texts = [
        _content_text(m.get("content")) for m in flat if m.get("role") == "user"
    ][-2:]
    focus = _heur.focus_terms_from_text(
        " ".join(user_texts) if user_texts else focus_src
    )

    latest: dict[str, int] = {}
    infos: list[_heur.CallInfo | None] = []
    for index, msg in enumerate(flat):
        if msg.get("role") != "tool":
            infos.append(None)
            continue
        info = _heur._call_info_from_tool_msg(msg, by_id)
        infos.append(info)
        for key in info.artifact_keys if info else []:
            latest[key] = index

    erase_in = erase_out = erase_count = 0
    seen_tokens = 0

    for index, msg in enumerate(flat):
        if msg.get("role") != "tool":
            continue
        step_i = step_of[index]
        if step_i is None:
            continue
        rid = step_id(step_i)
        # On-frontier: keep full tool output
        if rid in keep:
            stats.kept += 1
            continue
        if msg.get("agent_depends_dropped"):
            continue

        original = _content_text(msg.get("content"))
        # Strip REF if somehow present on tool body (usually not)
        _rid, bare = strip_leading_ref(original)
        body = bare if _rid else original

        if len(body) < config.min_carrier_chars or body.startswith(_heur.CAPSULE_PREFIX):
            stats.kept += 1
            continue

        if not msg.get("agent_depends_seen"):
            seen_tokens += count_token(body)
            msg["agent_depends_seen"] = True

        info = infos[index]
        superseded = bool(
            info
            and info.artifact_keys
            and all((latest.get(key, index) > index) for key in info.artifact_keys)
        )

        replacement: str | None = None
        if superseded:
            replacement = _heur.folded_artifact(body, info.artifact_keys if info else [])
            stats.folded += 1
        else:
            score = _heur.score_carrier(index, len(flat), info, body, latest, focus)
            # Bias score down for off-frontier so compress triggers more often (cost)
            score = max(0.0, score - 0.15)
            if score < config.drop_ratio and _heur.ERROR_RE.search(body):
                replacement = _heur.transient_failure(body)
            else:
                replacement = _heur.semantic_compress(
                    body, config.off_frontier_target_ratio, focus
                )

        # If heuristic capsule does not actually save (common when focus hits every line),
        # fall back to a hard char budget — needed for cost reduction.
        if (
            replacement is None
            or len(body) - len(replacement) < config.min_saved_chars
        ):
            budget = max(240, int(len(body) * config.off_frontier_target_ratio))
            if len(body) - budget >= config.min_saved_chars:
                head_n = max(120, budget * 2 // 3)
                tail_n = max(80, budget - head_n)
                replacement = (
                    f"{_heur.CAPSULE_PREFIX}off-frontier-trim "
                    f"original_chars={len(body)}]\n"
                    f"{body[:head_n]}\n...[{len(body) - head_n - tail_n} chars omitted]...\n"
                    f"{body[-tail_n:]}"
                )
                if not superseded:
                    stats.compressed += 1
            else:
                stats.kept += 1
                continue
        else:
            # semantic / transient path saved enough
            if not superseded:
                stats.compressed += 1

        _set_content_text(msg, replacement)
        if "agent_depends_orig" not in msg:
            msg["agent_depends_orig"] = body
        erase_in += count_token(body)
        erase_out += count_token(replacement)
        erase_count += 1
        kind = (
            "fold"
            if superseded
            else (
                "trim"
                if "off-frontier-trim" in (replacement or "")[:80]
                else "compress"
            )
        )
        print(
            f"!!! arbiteros_hybrid {kind} {rid} "
            f"tool#{index} {len(body)} -> {len(replacement)} chars"
        )

    mgr.metrics["seen_tokens"] += seen_tokens
    if erase_count:
        mgr.metrics["erase_tot_count"] += erase_count
        mgr.metrics["erase_in_tokens"] += erase_in
        mgr.metrics["erase_out_tokens"] += erase_out


# ---------------------------------------------------------------------------
# format_messages hook
# ---------------------------------------------------------------------------


def decorate_outgoing_messages(mgr: MessageManager, messages: list[dict]) -> list[dict]:
    """
    - Append depends_on rules to system prompt (no ref on system)
    - Ensure user + history steps carry simple [ARBITEROS_REF ...] labels
    - Sync stamped assistant text into the outgoing message list
    """
    if not is_enabled():
        return messages
    if not messages:
        return messages

    stamp_all_steps(mgr)

    out = [dict(m) for m in messages]

    if out[0].get("role") == "system":
        sys_text = _content_text(out[0].get("content"))
        if "## Causal step dependency" not in sys_text:
            out[0]["content"] = sys_text.rstrip() + "\n\n" + DEPENDS_ON_SYSTEM_RULES

    if len(out) > 1 and out[1].get("role") == "user":
        out[1] = {
            **out[1],
            "content": _content_text(mgr.user_message.get("content")),
        }

    flat = [m for s in mgr.steps for m in s]
    base = 2
    for i, stored in enumerate(flat):
        idx = base + i
        if idx >= len(out):
            break
        if stored.get("role") != "assistant":
            continue
        stamped = _content_text(stored.get("content"))
        content = out[idx].get("content")
        if isinstance(content, list):
            new_blocks = []
            applied = False
            for block in content:
                if (
                    not applied
                    and isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ):
                    new_blocks.append({**block, "text": stamped})
                    applied = True
                else:
                    new_blocks.append(block)
            out[idx] = {**out[idx], "content": new_blocks}
        else:
            out[idx] = {**out[idx], "content": stamped}

    return out


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def optimize_message_manager_depends(
    mgr: MessageManager,
    config: DependsConfig,
) -> DependsStats:
    stamp_all_steps(mgr)

    before = len(_content_text(mgr.user_message.get("content"))) + sum(
        step_text_chars(s) for s in mgr.steps
    )
    stats = DependsStats(before_chars=before, after_chars=before)

    keep = build_keep_set(mgr, config)
    stats.frontier_size = len(keep)
    print(
        f"-- arbiteros_depends keep={sorted(keep)} "
        f"action={config.off_frontier_action} hops={config.frontier_hops}"
    )

    if config.off_frontier_action == "drop":
        apply_step_drops(mgr, keep, config, stats)
    else:
        apply_off_frontier_heuristic_compress(mgr, keep, config, stats)

    after = len(_content_text(mgr.user_message.get("content"))) + sum(
        step_text_chars(s) for s in mgr.steps
    )
    stats.after_chars = after
    stats.saved_chars = before - after
    return stats


def perform_analysis_step_arbiteros_depends(
    mgr: MessageManager,
    analysis_args: dict | None = None,
) -> DependsStats:
    config = config_from_analysis_args(analysis_args)
    print(
        f"-- arbiteros_depends config: action={config.off_frontier_action} "
        f"protect_first={config.protect_first_steps} "
        f"protect_recent={config.protect_recent_steps} "
        f"hops={config.frontier_hops} "
        f"off_target={config.off_frontier_target_ratio}"
    )
    if mgr.steps:
        ensure_step_ref(mgr, len(mgr.steps) - 1, log_raw=True)

    stats = optimize_message_manager_depends(mgr, config)
    print(
        f"-- arbiteros_depends stats: {stats.before_chars} -> {stats.after_chars} "
        f"(saved {stats.saved_chars}, dropped={stats.dropped}, "
        f"compressed={stats.compressed}, folded={stats.folded}, "
        f"kept={stats.kept}, frontier={stats.frontier_size})"
    )
    return stats


# Alias entry for hybrid mode name
perform_analysis_step_arbiteros_hybrid = perform_analysis_step_arbiteros_depends
