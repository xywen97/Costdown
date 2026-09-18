import os
import sys
import unittest
from unittest import mock

import openai


TRAE_AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'trae_agent'))
if TRAE_AGENT_DIR not in sys.path:
    sys.path.insert(0, TRAE_AGENT_DIR)
os.environ.setdefault('TRAJ_ANALYSIS', '{"mode": "none"}')

from agents.expert import MessageManager
from utils.llm_polytool import (
    _chat_messages_to_responses_input,
    send_request_openai_responses,
)


def has_chat_breakpoint(message):
    return any(
        isinstance(block, dict) and block.get('cache_control') == {'type': 'ephemeral'}
        for block in message.get('content', [])
    )


def response_breakpoint_count(input_items):
    count = 0
    for item in input_items:
        for field in ('content', 'output'):
            value = item.get(field)
            if not isinstance(value, list):
                continue
            count += sum(
                isinstance(part, dict)
                and part.get('prompt_cache_breakpoint') == {'mode': 'explicit'}
                for part in value
            )
    return count


class PromptCacheBreakpointTests(unittest.TestCase):
    def make_manager(self):
        return MessageManager('/testbed', 'example issue', None, {}, True, 40, [])

    def test_format_messages_marks_user_and_every_tool_output(self):
        manager = self.make_manager()
        manager.steps = [
            [
                {'role': 'assistant', 'content': 'step zero', 'tool_calls': []},
                {'role': 'tool', 'content': 'result zero a', 'tool_call_id': 'call-0a'},
                {'role': 'tool', 'content': 'result zero b', 'tool_call_id': 'call-0b'},
            ],
            [
                {'role': 'assistant', 'content': 'step one', 'tool_calls': []},
            ],
        ]

        messages = manager.format_messages()

        self.assertTrue(has_chat_breakpoint(messages[1]))
        self.assertFalse(has_chat_breakpoint(messages[2]))
        self.assertTrue(has_chat_breakpoint(messages[3]))
        self.assertTrue(has_chat_breakpoint(messages[4]))
        self.assertFalse(has_chat_breakpoint(messages[5]))

        _, input_items = _chat_messages_to_responses_input(messages)
        self.assertEqual(response_breakpoint_count(input_items), 3)
        assistant_items = [item for item in input_items if item.get('role') == 'assistant']
        self.assertTrue(assistant_items)
        self.assertTrue(all(isinstance(item['content'], str) for item in assistant_items))

    def test_breakpoints_survive_reformat_and_early_step_rewrite(self):
        manager = self.make_manager()
        manager.steps = [
            [
                {'role': 'assistant', 'content': 'step zero'},
                {'role': 'tool', 'content': 'result zero', 'tool_call_id': 'call-0'},
            ],
            [
                {'role': 'assistant', 'content': 'step one'},
                {'role': 'tool', 'content': 'result one', 'tool_call_id': 'call-1'},
            ],
        ]

        first = manager.format_messages()
        manager.steps[1][1]['content'] = 'compressed result one'
        second = manager.format_messages()

        self.assertTrue(has_chat_breakpoint(first[3]))
        self.assertTrue(has_chat_breakpoint(first[5]))
        self.assertTrue(has_chat_breakpoint(second[3]))
        self.assertTrue(has_chat_breakpoint(second[5]))

    def test_arbiteros_decoration_does_not_remove_user_breakpoint(self):
        manager = self.make_manager()
        manager.steps = [[{'role': 'assistant', 'content': '[depends_on user]\nDone.'}]]

        with mock.patch('agents.arbiteros_depends.is_enabled', return_value=True):
            messages = manager.format_messages()

        self.assertTrue(has_chat_breakpoint(messages[1]))
        self.assertFalse(has_chat_breakpoint(messages[2]))

        _, input_items = _chat_messages_to_responses_input(messages)
        assistant = next(item for item in input_items if item.get('role') == 'assistant')
        self.assertIsInstance(assistant['content'], str)

    def test_responses_request_uses_implicit_mode_for_agent_breakpoints(self):
        manager = self.make_manager()
        manager.steps = [
            [
                {
                    'role': 'assistant',
                    'content': 'completed step',
                    'tool_calls': [
                        {
                            'id': 'call-1',
                            'type': 'function',
                            'function': {'name': 'noop', 'arguments': '{}'},
                        },
                        {
                            'id': 'call-2',
                            'type': 'function',
                            'function': {'name': 'noop', 'arguments': '{}'},
                        },
                    ],
                },
                {'role': 'tool', 'content': 'first result', 'tool_call_id': 'call-1'},
                {'role': 'tool', 'content': 'second result', 'tool_call_id': 'call-2'},
            ],
        ]
        captured = {}

        class FakeCompletion:
            def model_dump(self):
                return {
                    'id': 'response-id',
                    'status': 'completed',
                    'output': [],
                    'usage': {'input_tokens': 10, 'output_tokens': 1},
                }

        class FakeResponses:
            def create(self, **data):
                captured.update(data)
                return FakeCompletion()

        class FakeClient:
            def __init__(self, **kwargs):
                self.responses = FakeResponses()

        send = send_request_openai_responses('https://example.test', 'test-key')
        tools = [{
            'type': 'function',
            'function': {
                'name': 'noop',
                'description': '',
                'parameters': {'type': 'object', 'properties': {}},
            },
        }]

        with mock.patch('utils.llm_polytool.openai.OpenAI', FakeClient):
            send('gpt-5.6-terra', manager.format_messages(), tools, {})

        self.assertEqual(
            captured['prompt_cache_options'],
            {'mode': 'implicit', 'ttl': '30m'},
        )
        self.assertEqual(response_breakpoint_count(captured['input']), 3)

    def test_converter_ignores_accidental_assistant_breakpoint(self):
        _, input_items = _chat_messages_to_responses_input([
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'issue'},
            {
                'role': 'assistant',
                'content': [{
                    'type': 'text',
                    'text': 'compressed step',
                    'cache_control': {'type': 'ephemeral'},
                }],
            },
        ])

        assistant = next(item for item in input_items if item.get('role') == 'assistant')
        self.assertEqual(assistant['content'], 'compressed step')
        self.assertEqual(response_breakpoint_count(input_items), 0)

    def test_bad_request_is_not_retried(self):
        attempts = 0

        class FakeResponses:
            def create(self, **data):
                nonlocal attempts
                attempts += 1
                response = mock.Mock(status_code=400, headers={})
                response.request = mock.Mock()
                raise openai.BadRequestError(
                    'invalid input',
                    response=response,
                    body={'error': {'code': 'invalid_value'}},
                )

        class FakeClient:
            def __init__(self, **kwargs):
                self.responses = FakeResponses()

        send = send_request_openai_responses('https://example.test', 'test-key')
        with mock.patch('utils.llm_polytool.openai.OpenAI', FakeClient):
            with self.assertRaises(openai.BadRequestError):
                send(
                    'gpt-5.6-terra',
                    [{'role': 'user', 'content': 'hello'}],
                    [],
                    {},
                )

        self.assertEqual(attempts, 1)


if __name__ == '__main__':
    unittest.main()
