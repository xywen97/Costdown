import time
import json
import openai
import logging
import hashlib
from typing import Optional

logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpx2').setLevel(logging.ERROR)  # openai>=2 uses httpx2

### BEGIN CACHE

class HashKey:
    def __init__(self, info):
        self.cache_key = json.dumps(info, sort_keys=True)
        self.cache_hash = int(hashlib.sha256(self.cache_key.encode()).hexdigest()[:8], 16)

class NullCache:
    def __init__(self):
        pass

    def get(self, k: HashKey) -> Optional[object]:
        return None

    def put(self, k: HashKey, v: object):
        pass

llm_cache_chat = NullCache()

### END CACHE

def is_gpt5_family(model: str) -> bool:
    # Matches gpt-5-mini, gpt-5.6-terra, gpt-5.6-luna, etc.
    return model.startswith('gpt-5')

def is_gpt56_family(model: str) -> bool:
    # Newest GPT-5.6 line: must use Responses API for tools + reasoning.
    return model.startswith('gpt-5.6')


def _flatten_message_content(content) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get('text') is not None:
                    parts.append(str(part['text']))
                elif part.get('type') in ('text', 'output_text', 'input_text') and part.get('text') is not None:
                    parts.append(str(part['text']))
        return ''.join(parts)
    return str(content)


def _chat_tools_to_responses_tools(tools):
    """Chat Completions tool schema -> Responses API flat function tools."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get('type') == 'function' and 'function' in tool:
            fn = tool['function']
            out.append({
                'type': 'function',
                'name': fn['name'],
                'description': fn.get('description', ''),
                'parameters': fn.get('parameters') or {'type': 'object', 'properties': {}},
            })
        elif tool.get('type') == 'function' and 'name' in tool:
            out.append(tool)
        else:
            out.append(tool)
    return out


def _clean_responses_item(item: dict) -> dict:
    """Drop SDK-only null fields before replaying items into input."""
    skip = {'async_', 'status'}
    cleaned = {}
    for k, v in item.items():
        if k in skip or v is None:
            continue
        cleaned[k] = v
    if item.get('async_') is not None:
        cleaned['async'] = item['async_']
    return cleaned


def _chat_messages_to_responses_input(messages):
    """Convert our chat-style history into Responses `instructions` + `input`."""
    instructions_parts = []
    input_items = []

    for msg in messages:
        role = msg.get('role')
        if role == 'system':
            instructions_parts.append(_flatten_message_content(msg.get('content')))
            continue

        if role == 'user':
            input_items.append({
                'role': 'user',
                'content': _flatten_message_content(msg.get('content')),
            })
            continue

        if role == 'assistant':
            # Replay encrypted reasoning so tool loops keep model quality.
            for item in msg.get('agent_reasoning_items') or []:
                if isinstance(item, dict):
                    input_items.append(_clean_responses_item(item))

            text = _flatten_message_content(msg.get('content'))
            if text:
                input_items.append({'role': 'assistant', 'content': text})

            for tc in msg.get('tool_calls') or []:
                fn = tc.get('function') or {}
                input_items.append({
                    'type': 'function_call',
                    'call_id': tc.get('id') or tc.get('call_id'),
                    'name': fn.get('name'),
                    'arguments': fn.get('arguments') or '{}',
                })
            continue

        if role == 'tool':
            input_items.append({
                'type': 'function_call_output',
                'call_id': msg.get('tool_call_id'),
                'output': _flatten_message_content(msg.get('content')) or '',
            })
            continue

        # Already a Responses input item (rare).
        if msg.get('type'):
            input_items.append(msg)

    instructions = '\n\n'.join(p for p in instructions_parts if p).strip() or None
    return instructions, input_items


def _responses_output_to_chat_choice(output_items):
    """Map Responses output items -> chat Completions-like message + finish_reason."""
    content_parts = []
    tool_calls = []
    reasoning_items = []

    for item in output_items or []:
        if not isinstance(item, dict):
            continue
        typ = item.get('type')
        if typ == 'reasoning':
            reasoning_items.append(item)
        elif typ == 'function_call':
            tool_calls.append({
                'id': item.get('call_id') or item.get('id'),
                'type': 'function',
                'function': {
                    'name': item.get('name'),
                    'arguments': item.get('arguments') or '{}',
                },
            })
        elif typ == 'message':
            for part in item.get('content') or []:
                if isinstance(part, str):
                    content_parts.append(part)
                elif isinstance(part, dict):
                    if part.get('type') in ('output_text', 'text') or 'text' in part:
                        content_parts.append(part.get('text') or '')

    message = {
        'role': 'assistant',
        'content': '\n'.join(p for p in content_parts if p is not None),
        'refusal': None,
        'tool_calls': tool_calls or None,
        # Kept across turns; stripped from archival? kept as agent_* and preserved in format.
        'agent_reasoning_items': reasoning_items,
    }
    finish_reason = 'tool_calls' if tool_calls else 'stop'
    return message, finish_reason


def _as_number(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('$'):
            s = s[1:]
        try:
            return float(s)
        except ValueError:
            return None
    return None


_COST_KEYS = (
    'cost', 'total_cost', 'cost_usd', 'price', 'total_price',
    'usage_cost', 'billing', 'amount',
)


def _first_number(*vals):
    for v in vals:
        n = _as_number(v)
        if n is not None:
            return n
    return 0


def extract_usage_record(usage, response=None) -> dict:
    """Normalize provider usage into cache hit/miss/write, reasoning, and API cost.

    prompt_tokens is the provider's total input (typically already includes cached).
    cache_miss_tokens = prompt_tokens - cache_read_tokens (floored at 0).
    api_cost is only set when the upstream actually returns a price.
    """
    usage = dict(usage or {})
    response = response if isinstance(response, dict) else {}

    details_in = usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}
    details_out = usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}
    if not isinstance(details_in, dict):
        details_in = {}
    if not isinstance(details_out, dict):
        details_out = {}

    prompt = _first_number(usage.get('prompt_tokens'), usage.get('input_tokens'))
    completion = _first_number(usage.get('completion_tokens'), usage.get('output_tokens'))
    total = _first_number(usage.get('total_tokens'), prompt + completion)

    cache_read = _first_number(
        usage.get('cache_read_input_tokens'),
        usage.get('cache_read_tokens'),
        usage.get('cached_tokens'),
        details_in.get('cached_tokens'),
        details_in.get('cache_read_tokens'),
        details_in.get('cache_read_input_tokens'),
    )
    cache_write = _first_number(
        usage.get('cache_creation_input_tokens'),
        usage.get('cache_write_input_tokens'),
        usage.get('cache_write_tokens'),
        details_in.get('cache_write_tokens'),
        details_in.get('cache_creation_tokens'),
        details_in.get('cache_creation_input_tokens'),
    )
    for k in ('cache_creation_5_m_tokens', 'cache_creation_1_h_tokens'):
        cache_write += _first_number(usage.get(k), details_in.get(k))

    reasoning = _first_number(
        usage.get('reasoning_tokens'),
        details_out.get('reasoning_tokens'),
    )

    cache_miss = max(int(prompt) - int(cache_read), 0)

    api_cost = None
    for src in (usage, response, usage.get('cost_details') if isinstance(usage.get('cost_details'), dict) else None):
        if not isinstance(src, dict):
            continue
        for k in _COST_KEYS:
            n = _as_number(src.get(k))
            if n is not None:
                api_cost = n
                break
        if api_cost is not None:
            break

    extra = {}

    def _walk(obj, prefix=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ('raw_responses_usage', 'extra'):
                    continue
                key = f'{prefix}.{k}' if prefix else str(k)
                lk = str(k).lower()
                if any(s in lk for s in ('cache', 'cost', 'price', 'bill', 'reasoning')):
                    n = _as_number(v)
                    if n is not None:
                        extra[key] = n
                    elif isinstance(v, dict):
                        _walk(v, key)

    _walk(usage)
    for k in _COST_KEYS:
        n = _as_number(response.get(k))
        if n is not None:
            extra[f'response.{k}'] = n

    rec = {
        'prompt_tokens': int(prompt) if prompt == int(prompt) else prompt,
        'completion_tokens': int(completion) if completion == int(completion) else completion,
        'total_tokens': int(total) if total == int(total) else total,
        'cache_read_tokens': int(cache_read),
        'cache_write_tokens': int(cache_write),
        'cache_miss_tokens': int(cache_miss),
        'reasoning_tokens': int(reasoning),
        'api_cost': api_cost,
    }
    if extra:
        rec['extra'] = extra
    return rec


def record_llm_usage(metrics: dict, usage, *, kind: str, model: str | None = None, response=None):
    """Accumulate provider usage onto trajectory metrics; also append a per-call row."""
    rec = extract_usage_record(usage, response)
    rec['kind'] = kind
    if model:
        rec['model'] = model

    pfx = '' if kind == 'agent' else 'analysis_'
    metrics[f'{pfx}prompt_tokens'] = metrics.get(f'{pfx}prompt_tokens', 0) + (rec['prompt_tokens'] or 0)
    metrics[f'{pfx}completion_tokens'] = metrics.get(f'{pfx}completion_tokens', 0) + (rec['completion_tokens'] or 0)
    tot_key = 'cost_tokens' if kind == 'agent' else 'analysis_cost_tokens'
    metrics[tot_key] = metrics.get(tot_key, 0) + (rec['total_tokens'] or 0)

    metrics[f'{pfx}cache_read_tokens'] = metrics.get(f'{pfx}cache_read_tokens', 0) + rec['cache_read_tokens']
    metrics[f'{pfx}cache_write_tokens'] = metrics.get(f'{pfx}cache_write_tokens', 0) + rec['cache_write_tokens']
    metrics[f'{pfx}cache_miss_tokens'] = metrics.get(f'{pfx}cache_miss_tokens', 0) + rec['cache_miss_tokens']
    metrics[f'{pfx}reasoning_tokens'] = metrics.get(f'{pfx}reasoning_tokens', 0) + rec['reasoning_tokens']
    if rec.get('api_cost') is not None:
        metrics[f'{pfx}api_cost'] = (metrics.get(f'{pfx}api_cost') or 0) + rec['api_cost']

    prompt = rec['prompt_tokens'] or 0
    if prompt:
        metrics[f'{pfx}cache_hit_calls'] = metrics.get(f'{pfx}cache_hit_calls', 0) + int(rec['cache_read_tokens'] > 0)
        metrics[f'{pfx}cache_miss_calls'] = metrics.get(f'{pfx}cache_miss_calls', 0) + int(rec['cache_read_tokens'] <= 0)

    calls = metrics.setdefault('usage_calls', [])
    calls.append({k: v for k, v in rec.items() if v is not None})
    return rec


def _normalize_responses_usage(usage: dict, response: dict | None = None) -> dict:
    if not usage:
        usage = {}
    details_in = usage.get('input_tokens_details') or {}
    details_out = usage.get('output_tokens_details') or {}
    if not isinstance(details_in, dict):
        details_in = {}
    if not isinstance(details_out, dict):
        details_out = {}
    out = {
        'prompt_tokens': usage.get('input_tokens') or 0,
        'completion_tokens': usage.get('output_tokens') or 0,
        'total_tokens': usage.get('total_tokens') or 0,
        'cache_read_input_tokens': details_in.get('cached_tokens') or 0,
        'cache_creation_input_tokens': details_in.get('cache_write_tokens') or 0,
        'reasoning_tokens': details_out.get('reasoning_tokens') or 0,
        'prompt_tokens_details': details_in,
        'completion_tokens_details': details_out,
        'input_tokens_details': details_in,
        'output_tokens_details': details_out,
        'raw_responses_usage': usage,
    }
    src = response or {}
    for k in _COST_KEYS:
        if src.get(k) is not None:
            out[k] = src[k]
        elif usage.get(k) is not None:
            out[k] = usage[k]
    return out


def _apply_gpt5_chat_request_quirks(data: dict, model: str) -> None:
    """Quirks for Chat Completions path (gpt-5-mini and similar)."""
    if not is_gpt5_family(model) or is_gpt56_family(model):
        return
    if 'temperature' in data:
        del data['temperature']
    if 'stop' in data:
        del data['stop']
    if 'max_tokens' in data:
        data['max_completion_tokens'] = data.pop('max_tokens')
    # Legacy gpt-5-mini chat/completions still uses low; no tools+reasoning issue there
    # the same way as 5.6. Keep previous behavior.
    data['reasoning_effort'] = 'low'


def send_request_azure(endpoint, api_key):
    def s(model, messages, tools, kwargs):
        max_tokens = 8192  # range: [1, 8192]
        data = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            **kwargs,
        }

        if not tools: # Invalid 'tools': empty array. Expected an array with minimum length 1, but got an empty array instead.
            del data["tools"]

        _apply_gpt5_chat_request_quirks(data, model)

        hk = HashKey(data)

        res = llm_cache_chat.get(hk)
        if res:
            # print('cache hit')
            return res

        client = openai.AzureOpenAI(
            azure_endpoint=endpoint,
            api_version="2024-03-01-preview",
            api_key=api_key,
        )

        max_retries = 12
        retries = 0
        while retries < max_retries:
            try:
                completion = client.chat.completions.create(**data)
                if completion is None:
                    raise Exception("completion is None")

                if data.get('stream', False):
                    assert not data.get('tools', [])

                    resp_json = {
                        'choices': [{
                            'message': {
                                'role': 'assistant',
                                'content': '',
                                'refusal': None,
                                'annotations': None,
                                'audio': None,
                                'function_call': None,
                                'tool_calls': None,
                                'reasoning_content': '',
                            },
                            'finish_reason': None,
                            'index': 0,
                            'logprobs': None,
                        }],
                        'usage': {},
                    }

                    for ev in completion:
                        ev = ev.model_dump()
                        c = ev['choices']
                        if c:
                            assert len(c) == 1
                            c = c[0]
                            if c['finish_reason']:
                                resp_json['choices'][0]['finish_reason'] = c['finish_reason']
                            if c['delta'] and c['delta']['content']:
                                resp_json['choices'][0]['message']['content'] += c['delta']['content']
                        if ev.get('usage', None):
                            resp_json['usage'] = ev['usage']

                else:
                    resp_json = completion.model_dump()

                llm_cache_chat.put(hk, res)
                return resp_json
            except (openai.RateLimitError, openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError, openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as e:
                print(f"An error occurred: {type(e)} {e}")
                if retries < max_retries:
                    time.sleep(2 ** retries)
                retries += 1
            except Exception as e: # (openai.APIStatusError, openai.BadRequestError)
                print(f"A fatal error occurred: {type(e)} {e}")
                raise e

        print(f"Maximum retries ({max_retries}) exceeded.")
        return None

    return s


def send_request_openai_responses(base_url, api_key):
    """GPT-5.6 family: Responses API so tools can keep real reasoning_effort."""

    def s(model, messages, tools, kwargs):
        kwargs = dict(kwargs or {})
        # Chat Completions-only knobs.
        for k in ('n', 'stream', 'stop', 'temperature'):
            kwargs.pop(k, None)

        reasoning_effort = kwargs.pop('reasoning_effort', None)
        if reasoning_effort is None:
            # Keep agent quality; compressor (no tools) can stay cheaper.
            reasoning_effort = 'medium' if tools else 'low'

        max_output_tokens = kwargs.pop('max_output_tokens', None)
        if max_output_tokens is None:
            max_output_tokens = kwargs.pop('max_tokens', None) or kwargs.pop('max_completion_tokens', None) or 8192

        instructions, input_items = _chat_messages_to_responses_input(messages)
        data = {
            'model': model,
            'input': input_items,
            'max_output_tokens': max_output_tokens,
            'reasoning': {'effort': reasoning_effort},
        }
        if instructions:
            data['instructions'] = instructions
        if tools:
            data['tools'] = _chat_tools_to_responses_tools(tools)
        # Allow callers to override / extend (e.g. reasoning.mode later).
        data.update(kwargs)

        hk = HashKey(data)
        res = llm_cache_chat.get(hk)
        if res:
            return res

        if 'base_url' in (base_url or '') or api_key in (None, '', 'api_key'):
            raise RuntimeError(
                f"LLM upstream not configured for model={model!r}: "
                f"base_url={base_url!r}. Fill UPSTREAMS_PER_MODEL in llm_upstreams.py "
                f"or set fix_model/model to a configured entry (e.g. deepseek-v4-flash)."
            )

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=300.0,
        )

        max_retries = 12
        retries = 0
        while retries < max_retries:
            try:
                completion = client.responses.create(**data)
                if completion is None:
                    raise Exception("completion is None")
                raw = completion.model_dump()
                message, finish_reason = _responses_output_to_chat_choice(raw.get('output') or [])
                resp_json = {
                    'id': raw.get('id'),
                    'choices': [{
                        'message': message,
                        'finish_reason': finish_reason,
                        'index': 0,
                        'logprobs': None,
                    }],
                    'usage': _normalize_responses_usage(raw.get('usage') or {}, raw),
                    'raw_responses': {
                        'status': raw.get('status'),
                        'output_types': [x.get('type') for x in (raw.get('output') or []) if isinstance(x, dict)],
                    },
                }
                for k in _COST_KEYS:
                    if raw.get(k) is not None:
                        resp_json[k] = raw[k]
                llm_cache_chat.put(hk, resp_json)
                return resp_json
            except Exception as e:
                print(f"An error occurred [{model} @ {base_url} responses]: {type(e).__name__}: {e}")
                if retries < max_retries:
                    time.sleep(min(2 ** retries, 60))
                retries += 1

        print(f"Maximum retries ({max_retries}) exceeded for model={model} @ {base_url} (responses).")
        return None

    return s


def send_request_openai(base_url, api_key):
    def s(model, messages, tools, kwargs):
        # GPT-5.6+ requires Responses for tools + non-none reasoning.
        if is_gpt56_family(model):
            return send_request_openai_responses(base_url, api_key)(model, messages, tools, kwargs)

        max_tokens = 8192  # range: [1, 8192]
        data = {
            'model': model,
            'messages': messages,
            'max_tokens': max_tokens,
            'tools': tools,
            **kwargs,
        }

        if not tools: # Invalid 'tools': empty array. Expected an array with minimum length 1, but got an empty array instead.
            del data["tools"]

        _apply_gpt5_chat_request_quirks(data, model)

        hk = HashKey(data)

        res = llm_cache_chat.get(hk)
        if res:
            # print('cache hit')
            return res

        if 'base_url' in (base_url or '') or api_key in (None, '', 'api_key'):
            raise RuntimeError(
                f"LLM upstream not configured for model={model!r}: "
                f"base_url={base_url!r}. Fill UPSTREAMS_PER_MODEL in llm_upstreams.py "
                f"or set fix_model/model to a configured entry (e.g. deepseek-v4-flash)."
            )

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120.0,
        )

        max_retries = 12
        retries = 0
        while retries < max_retries:
            try:
                completion = client.chat.completions.create(**data)
                if completion is None:
                    raise Exception("completion is None")

                # if completion.choices[0].message.content == "":
                # raise Exception("completion.choices[0].message.content is empty")
                resp_json = completion.model_dump()

                llm_cache_chat.put(hk, res)
                return resp_json
            except Exception as e:
                print(f"An error occurred [{model} @ {base_url}]: {type(e).__name__}: {e}")
                if retries < max_retries:
                    time.sleep(min(2 ** retries, 60))
                retries += 1

        print(f"Maximum retries ({max_retries}) exceeded for model={model} @ {base_url}.")
        return None

    return s

try:
    from utils.llm_upstreams import UPSTREAMS_PER_MODEL
except ImportError as e:
    raise ImportError(
        'Missing utils/llm_upstreams.py — copy utils/llm_upstreams.example.py '
        'and fill in base_url / api_key.'
    ) from e

def get_llm_response(model: str, messages, tools, kwargs):
    if model not in UPSTREAMS_PER_MODEL:
        raise KeyError(
            f"Unknown model {model!r}. Configured: {sorted(UPSTREAMS_PER_MODEL)}"
        )
    upstream = UPSTREAMS_PER_MODEL[model]
    # time.sleep(10)
    decoded_answer = []
    finish_reason = []
    assistant_response = upstream(model, messages, tools, kwargs)
    if not assistant_response:
        raise RuntimeError(f'no response from api for model={model}')
    # print(assistant_response)
    for choice in assistant_response["choices"]:
        decoded_answer.append(choice["message"])
        finish_reason.append(choice["finish_reason"])
    usage = dict(assistant_response.get("usage") or {})
    for k in _COST_KEYS:
        if assistant_response.get(k) is not None and usage.get(k) is None:
            usage[k] = assistant_response[k]
    return decoded_answer, finish_reason, usage

if __name__ == "__main__":
    print(get_llm_response(
        "gpt-5.6-terra",
        [
                {"role": "system", "content": "You respond to what the user says."},
                {"role": "user", "content": "hello"},
        ],
        [],
        dict(temperature = 0.0, n = 1, stream=False),
    ))
