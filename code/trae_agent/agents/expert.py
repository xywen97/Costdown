from utils.llm_polytool import get_llm_response, record_llm_usage
from utils.agent_util import TIME_OUT_LABEL, remove_patches_to_tests
import time
import json
from .traj_analyzer import maybe_perform_analysis_step, FIX_MODEL
import shlex

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": """
Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file !!! If you know that the `path` already exists, please remove it first and then perform the `create` operation!
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
    """,
            "parameters": {
                "properties": {
                    "command": {
                        "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "type": "string",
                    },
                    "file_text": {
                        "description": "Required parameter of `create` command, with the content of the file to be created.",
                        "type": "string",
                    },
                    "insert_line": {
                        "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                        "type": "integer",
                    },
                    "new_str": {
                        "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "path": {
                        "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                        "type": "string",
                    },
                    "view_range": {
                        "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                        "items": {"type": "integer"},
                        "type": "array",
                    },
                },
                "required": ["command", "path"],
                "type": "object",
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": """
Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
    """,
            "parameters": {
                "properties": {
                    "command": {
                        "description": "The bash command to run. Required unless the tool is being restarted.",
                        "type": "string",
                    },
                },
                "type": "object",
            }
        }
    },
    {

        "type": "function",
        "function": {
            "name": "task_done",
            "description": """
            Report the completion of the task. Note that you cannot call this tool before any verfication is done. You can write reproduce / test script to verify your solution.
            """,
            "parameters": {
                "properties": {},
                "type": "object",
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed. For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective. Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your thoughts."
                    }
                },
                "required": ["thought"],
            },
        },
    },
]

def parse_tool_response(anwser, finish_reason, sandbox_session):
    result = []
    #print(f"finish_reason: {finish_reason}")
    for tool_call in (anwser.get("tool_calls", []) or []):
        tool_call_id = tool_call["id"]
        tool_name = tool_call["function"]["name"]
        try:
            tool_arguments = json.loads(tool_call["function"].get("arguments", "null"))
        except Exception:
            print('!!! cannot parse', tool_call)
            tool_message = {
                "role": "tool",
                "content": f"The argument given to {tool_name} is not a valid JSON. Please fix the problem and try again!",
                "tool_call_id": tool_call_id,
                "agent_caller": ('invalid', {'raw': tool_call['function']}),
                "is_error": True
            }
            result.append(tool_message)
            continue

        if tool_name == "think":
            tool_message = {
                "role": "tool",
                "content": "Continue.",
                "tool_call_id": tool_call_id,
                "agent_caller": (tool_name, tool_arguments),
            }
            result.append(tool_message)
            continue
        elif tool_name == "task_done":
            tool_message = {
                "role": "tool",
                "content": "Task done",
                "tool_call_id": tool_call_id,
                "agent_caller": (tool_name, tool_arguments),
            }
            #print("Tool Call Status: 1")
            result.append(tool_message)
            continue
        elif tool_name == "task_failed":
            tool_message = {
                "role": "tool",
                "content": "Task failed",
                "tool_call_id": tool_call_id,
                "agent_caller": (tool_name, tool_arguments),
            }
            #print("Tool Call Status: 1")
            result.append(tool_message)
            continue
        elif tool_name == "str_replace_editor":
            cmd = f"cd /home/swe-bench/tools/claude_tools/ && /home/swe-bench/conda_envs/py312/bin/python3 execute_str_replace_editor.py"
        elif tool_name == "bash":
            cmd = f"cd /home/swe-bench/tools/claude_tools/ && /home/swe-bench/conda_envs/py312/bin/python3 execute_bash.py"
        else:
            tool_message = {
                "role": "tool",
                "content": "The tool name you provided is not in the list!",
                "tool_call_id": tool_call_id,
                "agent_caller": (tool_name, tool_arguments),
                "is_error": True
            }
            result.append(tool_message)
            continue

        for key in tool_arguments:
            # print(key)
            # print(tool_arguments[key])
            if isinstance(tool_arguments[key], list):
                try:
                    tool_arguments[key] = str([int(factor) for factor in tool_arguments[key]])
                    cmd += f' --{key} {shlex.quote(tool_arguments[key])}'
                except Exception:
                    pass
            elif isinstance(tool_arguments[key], int):
                cmd += f' --{key} {tool_arguments[key]}'
            elif isinstance(tool_arguments[key], bool):
                cmd += f' --{key} {tool_arguments[key]}'
            else:
                cmd += f' --{key} {shlex.quote(tool_arguments[key])}'
        cmd += " > /home/swe-bench/tools/claude_tools/log.out 2>&1"
        # print(repr(cmd))
        sandbox_res =  sandbox_session.execute(cmd)
        if TIME_OUT_LABEL in sandbox_res:
            res_content = sandbox_res
            status = "Tool Call Status: -1"
        else:
            sandbox_res = sandbox_session.execute("cat /home/swe-bench/tools/claude_tools/log.out")
            status = ""
            status_line_index = -1
            sandbox_res_str_list = sandbox_res.split("\n")
            for index, line in enumerate(sandbox_res_str_list):
                if line.strip().startswith("Tool Call Status:"):
                    status = line
                    status_line_index = index
                    break
            if status_line_index != -1:
                sandbox_res_str_list.pop(status_line_index)
            res_content = "\n".join(sandbox_res_str_list)
        #print(status)
        tool_message = {
            "role": "tool",
            "content": res_content or "(no output)",
            "tool_call_id": tool_call_id,
            "agent_caller": (tool_name, tool_arguments),
        }
        if status == "Tool Call Status: -1":
            tool_message.update({"is_error": True})
        result.append(tool_message)

    return result

SYS_PROMPT = ("""
You are an expert AI software engineering agent. 
Your primary goal is to resolve a given GitHub issue by navigating the provided codebase, identifying the root cause of the bug, implementing a robust fix, and ensuring your changes are safe and well-tested.

Follow these steps methodically:

1.  Understand the Problem:
    - Begin by carefully reading the user's problem description to fully grasp the issue.
    - Identify the core components and expected behavior.

2.  Explore and Locate:
    - Use the available tools to explore the codebase.
    - Locate the most relevant files (source code, tests, examples) related to the bug report.

3.  Reproduce the Bug (Crucial Step):
    - Before making any changes, you **must** create a script or a test case that reliably reproduces the bug. This will be your baseline for verification.
    - Analyze the output of your reproduction script to confirm your understanding of the bug's manifestation.

4.  Debug and Diagnose:
    - Inspect the relevant code sections you identified.
    - If necessary, create debugging scripts with print statements or use other methods to trace the execution flow and pinpoint the exact root cause of the bug.

5.  Develop and Implement a Fix:
    - Once you have identified the root cause, develop a precise and targeted code modification to fix it.
    - Use the provided file editing tools to apply your patch. Aim for minimal, clean changes.

6.  Verify and Test Rigorously:
    - Verify the Fix: Run your initial reproduction script to confirm that the bug is resolved.
    - Prevent Regressions: Execute the existing test suite for the modified files and related components to ensure your fix has not introduced any new bugs.
    - Write New Tests: Create new, specific test cases (e.g., using `pytest`) that cover the original bug scenario. This is essential to prevent the bug from recurring in the future. Add these tests to the codebase.
    - Consider Edge Cases: Think about and test potential edge cases related to your changes.

7.  Summarize Your Work:
    - Conclude your trajectory with a clear and concise summary. Explain the nature of the bug, the logic of your fix, and the steps you took to verify its correctness and safety.

**Guiding Principle:** Act like a senior software engineer. Prioritize correctness, safety, and high-quality, test-driven development.

If you are sure the issue has been solved, you should call the `task_done` to finish the task.
""").strip()

INIT_USER_PROMPT = """
[Project root path]:
{project_path}

[Problem statement]: We're currently solving the following issue within our repository. Here's the issue text:
{issue}
""".strip()

class MessageManager:
    USE_CACHING = True

    STEP_BASE = 0
    RESULT_BASE = 100

    def __init__(self, project_path, issue, sandbox, metrics, turn_reminder, max_turn, tools):
        self.steps = []
        self.sandbox = sandbox
        self.metrics = metrics
        self.max_turn = max_turn
        self.turn_reminder = turn_reminder
        self.tools = tools

        user_prompt = INIT_USER_PROMPT.format(project_path=project_path, issue=issue)
        self.user_message = {"role": "user", "content": user_prompt}

    def push_step(self, assistant_msg, followup_msgs):
        assert assistant_msg['role']=='assistant'

        #ckpt = self.sandbox.make_checkpoint()
        #assistant_msg['agent_ckpt'] = ckpt

        cur_step = [assistant_msg, *followup_msgs]

        if self.steps and len(self.steps[-1])==0 and self.steps[-1][0]['role']=='assistant':
            # combine assistant messages with prefix
            old_step = self.steps.pop()
            assistant_msg['content'] = (old_step[0].get('content') or '') + (assistant_msg.get('content') or '')

        self.steps.append(cur_step)

    @staticmethod
    def _remove_internal_fields(msgs):
        # Keep agent_reasoning_items: GPT-5.6 Responses API needs encrypted
        # reasoning replayed on the next turn when tools are used.
        keep = {'agent_reasoning_items'}
        return [
            {k: v for k, v in msg.items() if (not k.startswith('agent_')) or k in keep}
            for msg in msgs
        ]

    def perform_erase_step(self, idx: int, content: str | None, orig: str):
        self.steps[idx] = [
            {
                'role': 'assistant',
                'content': (
                    f'(System reminder: compressed for better efficiency) {content.strip()}'
                    if content else
                    '(System reminder: long content deleted for better efficiency)'
                ),
                'agent_erased': orig,
            }
        ]

    def extract_step_into_traj(self, idx, bypass_filter=False):
        if idx==-1:
            return self.user_message['content']

        else:
            out = [f'<step id="{idx}">']
            for msg in self.steps[idx]:
                tag_params = ''
                content = msg.get('content') or ''
                if msg['role'] == 'user':
                    tag = 'user'
                elif msg['role'] == 'assistant':
                    tag = 'talk' if bypass_filter else 'think'
                    if msg.get('agent_erased', ''):
                        out.append(msg['agent_erased'])
                        continue
                elif msg['role'] == 'tool':
                    tag = 'result'
                    if msg['agent_caller'][0]=='think':
                        tag = 'talk' if bypass_filter else 'think'
                        content = (msg['agent_caller'][1] or {}).get('thought', '') or ''
                        content = f'\n{content.strip()}\n'
                else:
                    raise ValueError(f'wtf msg role {msg["role"]}')

                if content.strip():
                    out.append(f'<{tag}{tag_params}>{content}</{tag}>')
                if msg['role'] == 'assistant' and msg.get('tool_calls', None):
                    for tool_call in msg['tool_calls']:
                        tool_name = tool_call["function"]["name"]
                        tool_arguments = (tool_call["function"].get("arguments") or "").strip()
                        out.append(f'<call tool="{tool_name}">{tool_arguments}</call>')

            out += ['</step>']
            return '\n'.join(out)

    def get_input_info_for_archival(self):
        return {
            'system_prompt': {
                'role': 'system',
                'content': SYS_PROMPT,
            },
            'user_prompt': self.user_message,
            'tools': self.tools,
        }

    def format_messages(self):
        msgs = [m for s in self.steps for m in self._remove_internal_fields(s)]

        ret = [
            {
                'role': 'system',
                'content': SYS_PROMPT,
            },
            self.user_message,
            *msgs,
        ]
        turn_left = self.max_turn - self.count_turn()

        if ret[-1]['role']=='assistant': # used when ctx_after==0 or assistant did not call any tool
            if self.turn_reminder:
                ret.append({
                    'role': 'user',
                    'content': f'Continue in standard tool call format. ENVIRONMENT REMINDER: You have {turn_left} turns left to complete the task.'
                })
            else:
                ret.append({
                    'role': 'user',
                    'content': f'Continue in standard tool call format.'
                })
        else:
            # begin a new turn: add cache control and env reminder
            if self.USE_CACHING and msgs:
                ret[-1] = {
                    **msgs[-1],
                    'content': [{
                        'type': 'text',
                        'text': msgs[-1]['content'],
                        'cache_control': {'type': 'ephemeral'},
                    }],
                }

            if self.turn_reminder:
                ret.append({
                    'role': 'user',
                    'content': f'ENVIRONMENT REMINDER: You have {turn_left} turns left to complete the task.'
                })

        # Parallel ArbiterOS depends_on mode: append system rules + step REF labels.
        # No-op unless TRAJ_ANALYSIS mode == arbiteros_depends.
        try:
            from .arbiteros_depends import decorate_outgoing_messages, is_enabled
            if is_enabled():
                ret = decorate_outgoing_messages(self, ret)
        except Exception as exc:
            print(f'!!! arbiteros_depends decorate skipped: {exc}')

        return ret

    def count_turn(self):
        return len(self.steps)

NEED_RESET = True
class Expert():
    def __init__(self,sandbox):
        #self.model_list = ["aws_sdk_claude37_sonnet"]
        self.model_list = [FIX_MODEL]
        self.sandbox = sandbox
        self.sandbox_session = self.sandbox.get_session()
        self.project_path = ''
        self.issue =''
        self.mgr = None

    def run(self, project_path, issue_item, trajectory):
        final_patch = ''
        final_result = None

        metrics = trajectory['metrics']
        #init_ckpt = self.sandbox.make_checkpoint()

        for model in self.model_list:
            #self.sandbox.restore_checkpoint(init_ckpt)
            #self.sandbox_session.execute(f"git reset --hard HEAD && rm {project_path}/reproduce.py")

            mgr = MessageManager(project_path, issue_item["problem_statement"], self.sandbox, metrics, issue_item["turn_reminder"], issue_item["max_turn"], TOOLS)
            self.mgr = mgr

            trajectory['input'] = mgr.get_input_info_for_archival()

            patch = ''
            result = None

            finish_task = False
            cost_tokens = 0

            while True:
                turn = mgr.count_turn()
                if turn >= mgr.max_turn:
                    trajectory['result']['gen'] = 'turn_capped'

                    # try the wip patch if that helps
                    original_patch = self.sandbox.get_diff_result(project_path)
                    patch = remove_patches_to_tests(original_patch)

                    if patch.strip():
                        finish_task = True

                    break
                print(f"============================[Expert] turn {turn+1}=============================", time.ctime())

                msgs_in = mgr.format_messages()

                expert_answer_list, finish_reason_list, usage = get_llm_response(model, msgs_in, TOOLS, dict(temperature = 0.0, n = 1))
                metrics['tot_step'] += 1

                # print(json.dumps(expert_answer_list, indent=4))
                expert_answer = expert_answer_list[0]
                system_res_messages = []

                if usage["completion_tokens"] is not None:
                    # print(expert_answer_list)
                    # print(finish_reason_list)
                    # print(usage)
                    finish_reason = finish_reason_list[0]
                    record_llm_usage(metrics, usage, kind='agent', model=model)

                    expert_answer_content = expert_answer["content"]
                    if expert_answer_content is None:
                        expert_answer_content = ""
                    expert_answer["content"] = expert_answer_content

                    #print(json.dumps(expert_answer, indent=4))
                    tool_call_list = parse_tool_response(expert_answer, finish_reason, self.sandbox_session)

                    #print(tool_call_list)

                    if len(tool_call_list) == 1 and tool_call_list[0]['agent_caller'][0] == 'task_done':
                        original_patch = self.sandbox.get_diff_result(project_path)
                        patch = remove_patches_to_tests(original_patch)
                        if patch.strip() == '':
                            system_res_messages += tool_call_list
                            system_res_messages.append({
                                "role": "user",
                                "content": "ERROR! Your Patch is empty. Please provide a patch that fixes the problem."
                            })
                            result = None
                        else:
                            finish_task = True
                            trajectory['result']['gen'] = 'task_done'
                            mgr.push_step(expert_answer, system_res_messages)
                            break
                    elif len(tool_call_list) == 1 and tool_call_list[0]['agent_caller'][0] == 'task_failed':
                        print("GENERATE PATCH ERROR!")
                        finish_task = True
                        trajectory['result']['gen'] = 'task_failed'
                        mgr.push_step(expert_answer, system_res_messages)
                        break
                    elif not tool_call_list:
                        pass
                    else:
                        system_res_messages += tool_call_list
                        if TIME_OUT_LABEL in str(system_res_messages):
                            print('COMMAND TIMEOUT. restart sandbox session.')
                            self.sandbox_session = self.sandbox.get_session()

                mgr.push_step(expert_answer, system_res_messages)

                maybe_perform_analysis_step(mgr)

            trajectory['messages'] = [m for s in mgr.steps for m in s]

            if finish_task:
                final_result = result
                final_patch = patch
                break

        self.sandbox_session.close()

        return final_result, final_patch, ''