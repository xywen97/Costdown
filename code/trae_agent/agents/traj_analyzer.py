from __future__ import annotations

from utils.llm_polytool import get_llm_response, is_gpt5_family, record_llm_usage

import os
import typing
import tiktoken
import lz4.frame
import json
import random

analysis_args = json.loads(os.environ['TRAJ_ANALYSIS'].strip())
print('-- analysis args:', analysis_args)

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
_llm_lingua = None
def get_lingua():
    global _llm_lingua
    if not _llm_lingua:
        from llmlingua import PromptCompressor
        _llm_lingua = PromptCompressor(
            model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
            use_llmlingua2=True,
            device_map='cpu',
        )
    return _llm_lingua

def count_comp(b):
    return len(lz4.frame.compress(b))

_token_encoding = tiktoken.encoding_for_model('gpt-4o')
def count_token(s):
    #return len(s)
    return len(_token_encoding.encode(s))

if typing.TYPE_CHECKING:
    from .expert import MessageManager

SYS_PROMPT = """
You will analyze and compress a given step in a trajectory of an AI agent solving a software bug.

In the trajectory, each step is marked in <step id="..."></step>.
The agent will think in <think>, call external tools as marked in <call tool="..."></call>. Its result is marked in <result></result> within the <step> tag.

Your job is to compress the text within the given id to avoid harming efficiency, typically shortening it to 20%-50% of the original length.
Meanwhile, keep the compressed text useful such that you are able to continue the trajectory as close as the original path.

- You should ONLY remove redundant texts, which are either irrelevant to future steps or duplicated by other texts in the trajectory.
- Replace the text to remove to "..." and a short takeaway, e.g. "... (same as the content below)".
- You should keep the original structure unchanged, e.g., XML tags, Python indentation and line numbers.
- Again, keep useful details in the original content unchanged, e.g., XML tags, Python indentation and line numbers.

Typical examples:
- If the step opens a huge file but only one part is necessary for future steps, replace other parts to "... (unrelated function XXX, YYY)".
- If the step runs a verbose test script and everything goes fine, replace the verbose part to "... (expected output)".
- If the step uses str_replace_editor to modify a file and the content can be inferred by the content after it, replace the tool call argument to "... (see results below)".

You should only process the text within the <step> tag with the given id. STOP OUTPUT IMMEDIATELY AFTER </step>.
""".strip()

##### BEGIN ARGS

MODE = analysis_args['mode'].strip()

FIX_MODEL = analysis_args.get('fix_model', 'claude4-sonnet').strip()

#MODEL = analysis_args.get('model', 'gemini-2.5-flash').strip()
MODEL = analysis_args.get('model', 'gpt-5-mini-2025-08-07').strip()
LINGUA_RATIO = float(analysis_args.get('lingua_ratio', .25))

BYPASS_FILTER = is_gpt5_family(MODEL)

N_CTX_BEFORE = int(analysis_args.get('ctx_before', 1))
N_CTX_AFTER = int(analysis_args.get('ctx_after', 2))

SHOW_CTX = bool(int(analysis_args.get('show_ctx', 1)))
USE_LZ4 = bool(int(analysis_args.get('use_lz4', 0)))

THRESHOLD_TOKENS = int(analysis_args.get('threshold', 500))

##### END ARGS

def perform_analysis_step_llmlingua(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    new_content = get_lingua().compress_prompt(traj_content, rate=LINGUA_RATIO, force_tokens=['\n', '?'])

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += count_token(traj_content)
    mgr.metrics['erase_out_tokens'] += count_token(new_content['compressed_prompt'])

    mgr.perform_erase_step(idx, new_content['compressed_prompt'], traj_content)

def perform_analysis_step_random_drop(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    def drop_tokens(ts: list[int], n: float) -> tuple[str, int]:
        deletable_indices = []
        for ind, t in enumerate(ts):
            try:
                _token_encoding.decode_single_token_bytes(t).decode()
            except UnicodeDecodeError:
                pass
            else:
                deletable_indices.append(ind)

        deleted_indices = random.sample(deletable_indices, min(int(n), len(deletable_indices)))
        remaining_tokens = [t for ind, t in enumerate(ts) if ind not in deleted_indices]

        print('tot', len(ts), 'deletable', len(deletable_indices), 'deleted', len(deleted_indices), 'remaining', len(remaining_tokens))

        return _token_encoding.decode(remaining_tokens), len(remaining_tokens)

    old_tokens = _token_encoding.encode(traj_content)
    new_content, len_new_tokens = drop_tokens(old_tokens, len(old_tokens) * (1-LINGUA_RATIO))

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += len(old_tokens)
    mgr.metrics['erase_out_tokens'] += len_new_tokens

    mgr.perform_erase_step(idx, new_content, traj_content)


def perform_analysis_step_delete(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = mgr.extract_step_into_traj(idx)

    mgr.metrics['erase_tot_count'] += 1
    mgr.metrics['erase_in_tokens'] += count_token(traj_content)
    mgr.metrics['erase_out_tokens'] += 0

    mgr.perform_erase_step(idx, None, traj_content)


def perform_analysis_step_ours(mgr: MessageManager):
    idx = mgr.count_turn() - 1 - N_CTX_AFTER
    traj_content = [
        mgr.extract_step_into_traj(i, BYPASS_FILTER) # -1 is user prompt
        for i in range(idx - N_CTX_BEFORE, idx + N_CTX_AFTER + 1)
    ]

    if not SHOW_CTX:
        for i in range(len(traj_content)):
            if i != N_CTX_BEFORE:
                traj_content[i] = ''

    sys_prompt = SYS_PROMPT

    str_traj_content = '\n'.join(traj_content)

    if BYPASS_FILTER:
        sys_prompt = sys_prompt.replace('think', 'talk')
        sys_prompt = sys_prompt.replace('agent', 'engineer')

    user_prompt = f'{str_traj_content}\n\nNow, compress the step {idx}.'

    if mgr.USE_CACHING:
        sys_prompt = [{
            'type': 'text',
            'text': sys_prompt,
            'cache_control': {'type': 'ephemeral'},
        }]

    msgs = [
        {
            'role': 'system',
            'content': sys_prompt,
        },
        {
            'role': 'user',
            'content': user_prompt,
        },
        {
            'role': 'assistant',
            'content': f'Sure. Here is the compressed content of step {idx}: <step id="{idx}">',
        }
    ]

    kwargs = dict(temperature = 0.0, n = 1, stop='</step>')
    if 'qwen3-235b-a22b-instruct-2507' in MODEL: # overcome gateway timeout
        kwargs['stream'] = True

    expert_answer_list, finish_reason_list, usage = get_llm_response(MODEL, msgs, [], kwargs)
    if usage.get("completion_tokens") is None:
        print('wtf analysis result', usage, expert_answer_list, finish_reason_list, finish_reason_list)
        return

    record_llm_usage(mgr.metrics, usage, kind='analysis', model=MODEL)

    expert_answer = expert_answer_list[0]

    #print(expert_answer['content'])

    content, splitter, leftover_content = expert_answer['content'].partition('</step>')

    if not splitter:
        if 'stop' not in finish_reason_list:
            print('!!! invalid eraser', finish_reason_list, expert_answer)
            return

    if '<step' in content[:200]:
        content = content.partition('<step')[2]
        if '>' in content[:20]:
            content = content.partition('>')[2]

    old_tokens = count_token(traj_content[N_CTX_BEFORE])
    new_tokens = count_token(content)

    if old_tokens - new_tokens >= 400 or new_tokens < .8 * old_tokens:
        print('!!! erased', old_tokens, '->', new_tokens)

        mgr.metrics['erase_tot_count'] += 1
        mgr.metrics['erase_in_tokens'] += old_tokens
        mgr.metrics['erase_out_tokens'] += new_tokens

        mgr.perform_erase_step(idx, content, traj_content[N_CTX_BEFORE])
    else:
        print('  no erase', old_tokens, '->', new_tokens)

def should_perform_analysis(mgr: MessageManager):
    turn = mgr.count_turn()
    if turn < N_CTX_BEFORE + N_CTX_AFTER:
        return False

    traj_content = [
        mgr.extract_step_into_traj(i)
        for i in range(turn - N_CTX_BEFORE - N_CTX_AFTER - 1, turn)
    ]

    tokens = count_token(traj_content[N_CTX_BEFORE])
    mgr.metrics['seen_tokens'] += tokens

    if tokens < THRESHOLD_TOKENS:
        print('!!! no analysis (too short)', tokens)
        return False

    if not USE_LZ4:
        print('!!! DO analysis (length ok)', tokens)
        return True

    else:
        x1 = count_comp(''.join(traj_content[-(N_CTX_AFTER):]).encode('utf-8'))
        x2 = count_comp(''.join(traj_content[-(N_CTX_AFTER+1):]).encode('utf-8'))

        save_rate = 1 - max(0, x2-x1) / len(traj_content[N_CTX_BEFORE].encode('utf-8'))
        save_tokens = tokens * save_rate

        if save_tokens >= THRESHOLD_TOKENS:
            print('!!! DO analysis', tokens, save_rate, save_tokens)
            return True
        else:
            print('!!! no analysis', tokens, save_rate, save_tokens)
            return False

def maybe_perform_analysis_step(mgr: MessageManager):
    # judge if analysis is necessary
    if MODE == 'skip':
        print('skipped traj analysis')
        return

    # ArbiterOS operates on the full history each turn (with its own gates),
    # not on a single delayed step like the other modes.
    if MODE == 'arbiteros':
        print('===== begin traj analysis (arbiteros) =====')
        mgr.metrics['analysis_count'] += 1
        from .arbiteros import perform_analysis_step_arbiteros
        perform_analysis_step_arbiteros(mgr, analysis_args)
        return

    # Parallel experiment: step-level depends_on + frontier DROP.
    # Independent of MODE=arbiteros (heuristic cost-down in arbiteros.py).
    if MODE == 'arbiteros_depends':
        print('===== begin traj analysis (arbiteros_depends) =====')
        mgr.metrics['analysis_count'] += 1
        from .arbiteros_depends import perform_analysis_step_arbiteros_depends
        perform_analysis_step_arbiteros_depends(mgr, analysis_args)
        return

    # Hybrid: depends_on frontier + heuristic tool compress (no whole-step DROP).
    if MODE == 'arbiteros_hybrid':
        print('===== begin traj analysis (arbiteros_hybrid) =====')
        mgr.metrics['analysis_count'] += 1
        from .arbiteros_depends import perform_analysis_step_arbiteros_hybrid
        perform_analysis_step_arbiteros_hybrid(mgr, analysis_args)
        return

    if not should_perform_analysis(mgr):
        return

    # okay perform analysis now
    print('===== begin traj analysis =====')
    mgr.metrics['analysis_count'] += 1

    if MODE == 'lingua':
        perform_analysis_step_llmlingua(mgr)
    elif MODE == 'random':
        perform_analysis_step_random_drop(mgr)
    elif MODE == 'delete':
        perform_analysis_step_delete(mgr)
    elif MODE == 'ours':
        perform_analysis_step_ours(mgr)
    else:
        raise ZeroDivisionError(f'wtf mode: {analysis_args}')
