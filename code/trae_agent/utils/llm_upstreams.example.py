# Copy to llm_upstreams.py and fill in real base_url / api_key.
from utils.llm_polytool import send_request_openai

_ZHIZENGZENG = send_request_openai(
    'https://api.zhizengzeng.com/v1',
    'api_key',
)
UPSTREAMS_PER_MODEL = {
    'gemini-2.5-pro': _ZHIZENGZENG,
    'gemini-2.5-flash': send_request_openai('https://base_url', 'api_key'),
    'gpt-5-mini': _ZHIZENGZENG,
    'gpt-5.6-terra': _ZHIZENGZENG,
    'gpt-5.6-luna': _ZHIZENGZENG,
    'claude-sonnet-4-20250514': send_request_openai('https://base_url', 'api_key'),
    'claude35-haiku': send_request_openai('https://base_url', 'api_key'),
    'deepseek-v4-flash': send_request_openai('https://api.deepseek.com/v1', 'api_key'),
    'qwen3-235b-a22b-instruct-2507': send_request_openai('https://base_url', 'api_key'),
}
