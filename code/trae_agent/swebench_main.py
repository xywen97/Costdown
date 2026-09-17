from utils.swebench_validate import validators, MSB_FLASH_FILES # import first because it set envs for logging
from agents.expert import Expert
from agents.traj_analyzer import analysis_args
from utils.sandbox import Sandbox, install_cleanup_handlers
from utils.agent_util import save_trajectory, save_patches
import argparse
import json
import multiprocessing
import os
import time
import sys
import traceback
import random
from pathlib import Path
from datetime import datetime

def _cleanup_sandbox(sandbox):
    if sandbox is None:
        return None
    try:
        sandbox.stop_container()
        sandbox.destroy_all_checkpoints()
    except Exception:
        traceback.print_exc()
    return None

def run_autodebug_task_claude_tool(args, item, lock, single_inst_mode):
    print(f"processing: ", item["instance_id"])
    output_path = args.output_path
    output_file_path = os.path.join(output_path, f'task_{item["instance_id"]}.log')

    with open(output_file_path, 'w', buffering=1) as log_file:
        if not single_inst_mode:
            sys.stdout = log_file
            sys.stderr = log_file

        trajectory = {
            'result': {
                'gen': '',
                'val': '',
            },
            'metrics': {
                'analysis_args': analysis_args,

                'tot_step': 0,

                'cost_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cache_read_tokens': 0,
                'cache_write_tokens': 0,
                'cache_miss_tokens': 0,
                'reasoning_tokens': 0,
                'api_cost': 0,
                'cache_hit_calls': 0,
                'cache_miss_calls': 0,

                'analysis_cost_tokens': 0,
                'analysis_prompt_tokens': 0,
                'analysis_completion_tokens': 0,
                'analysis_cache_read_tokens': 0,
                'analysis_cache_write_tokens': 0,
                'analysis_cache_miss_tokens': 0,
                'analysis_reasoning_tokens': 0,
                'analysis_api_cost': 0,
                'analysis_cache_hit_calls': 0,
                'analysis_cache_miss_calls': 0,

                'usage_calls': [],

                'analysis_count': 0,

                'erase_tot_count': 0,

                'seen_tokens': 0,
                'erase_in_tokens': 0,
                'erase_out_tokens': 0,
            },
            'input': None,
            'messages': [],
        }

        try:
            #print(f"*********************{index}/500*********************")
            p2p_retry = 1
            current_try = 0
            valid_patch = ''
            sandbox = None
            try:
                while(current_try < p2p_retry):
                    print("current_try:", current_try)
                    now = datetime.now()
                    datetime_str = now.strftime('%Y%m%d%H%M%S')
                    print("time: ", datetime_str)
                    current_try += 1

                    expert = None
                    sandbox = _cleanup_sandbox(sandbox)

                    try:
                        namespace, image_name, tag = item['docker_info']
                        image_name = image_name.lower()

                        sandbox = Sandbox(namespace, image_name, tag, item)
                        sandbox.start_container()
                        project_path = sandbox.get_project_path()
                        expert = Expert(sandbox)

                        print("Expert Start!!!")
                        _, final_patch, _ = expert.run(project_path, item, trajectory)

                        if final_patch.strip() == '':
                            if current_try < p2p_retry:
                                continue

                        valid_patch = final_patch
                        break

                    except Exception as e:
                        print(f"Error occurred: {e}")
                        trajectory['result']['gen'] = f'err-{type(e)}'
                        if expert and expert.mgr:
                            trajectory['messages'] = [m for s in expert.mgr.steps for m in s]

                        traceback.print_exc()
            finally:
                sandbox = _cleanup_sandbox(sandbox)

            save_patches(instance_id=item["instance_id"], patches_path=args.patches_path, patches=valid_patch)

            if valid_patch.strip():
                print('========== validating patch', time.ctime())
                try:
                    val = validators[item['validator']]({item['instance_id']: valid_patch}, lock)
                    if val[item['instance_id']]:
                        trajectory['result']['val'] = 'pass'
                        print(f"========== PASS: This instance has been successfully resolved!!")
                    else:
                        trajectory['result']['val'] = 'fail'
                        print('========== FAIL: Validation failed !!')
                except Exception:
                    if single_inst_mode:
                        raise
                    else:
                        print(f'========== ERROR: Validation error !! (id = {item["instance_id"]})')
                        traceback.print_exc()
                        trajectory['result']['val'] = 'error'
            else:
                trajectory['result']['val'] = 'nopatch'
                print(f"========== NOPATCH: Not resolved by agent !!")

            save_trajectory(instance_id=item["instance_id"], traj_dir=args.log_path, trajectory=trajectory)

        finally:
            if not single_inst_mode:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

            print(f"== finished (val = {trajectory['result']['val']} / gen = {trajectory['result']['gen']}): {item['instance_id']} @ {time.ctime()}")


def worker(task_queue, lock, single_inst_mode):
    install_cleanup_handlers()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            args, item = task
            run_autodebug_task_claude_tool(args, item, lock, single_inst_mode)
    finally:
        Sandbox.sweep_orphans()

def main(args, lock):
    num_processes = 4  # 4C/15G machine: safe=2, max~3; was 0 (sequential) / 9 (paper default)

    SPECIFIC_ID = os.environ.get('INSTANCE_ID')
    if SPECIFIC_ID:
        print('!! use specific id', SPECIFIC_ID)

    jsonList = []
    need_run_set = set()

    if args.benchmark.startswith('swebench-verified-'):
        with open(Path(__file__).resolve().parent.parent / 'data' / 'swebench-verified.json', 'r') as file:
            jsonList = json.load(file)

        for d in jsonList:
            d['validator'] = 'swebench'
            d['docker_info'] = ("swebench", 'sweb.eval.x86_64.' + d["instance_id"].replace('__', '_1776_'), 'latest')
            d['turn_reminder'] = True
            d['max_turn'] = 50

        subset_fn = {
            'swebench-verified-appr100': '../subjects/approach_100.json',
            'swebench-verified-eval200': '../subjects/eval_200.json',
        }[args.benchmark]

        with open(subset_fn, 'r') as file:
            for k in json.load(file):
                need_run_set.add(k)

    elif args.benchmark == 'multiswebench-flash':
        with (MSB_FLASH_FILES / 'multi_swe_bench_flash.jsonl').open() as file:
            for l in file.read().splitlines():
                d = json.loads(l)
                issue = d['resolved_issues'][0]
                jsonList.append({
                    **d,
                    'base_commit': None,
                    'custom_cwd': f'/home/{d["repo"]}',
                    'problem_statement': f'{issue["title"]}\n\n{issue["body"]}',
                    'validator': 'multiswebench',
                    'docker_info': ('mswebench', f'{d["org"]}_m_{d["repo"]}', f'pr-{d["number"]}'),
                    'turn_reminder': False,
                    'max_turn': 100,
                })
                need_run_set.add(d['instance_id'])

    else:
        raise RuntimeError(f'unknown benchmark: {args.benchmark}')

    if SPECIFIC_ID:
        need_run_set = set(SPECIFIC_ID.split(','))

    tasks = []
    for t in jsonList:
        if t["instance_id"] in need_run_set.copy():
            done = False
            bugid = t["instance_id"]
            for serial in range(9, 0, -1):
                log_p = Path(args.log_path) / f'{t["instance_id"]}_{serial}.json'
                if log_p.is_file():
                    raw = log_p.read_text().strip()
                    if not raw:
                        print(f'!!empty log, will rerun: {log_p}')
                        break
                    try:
                        d = json.loads(raw)
                    except json.JSONDecodeError as e:
                        print(f'!!corrupt log ({e}), will rerun: {log_p}')
                        break
                    if 'result' in d and d['result']['val'] and not d['result']['gen'].startswith('err') and not d['result']['val'].startswith('err'):
                        print('!!done', bugid)
                        done = True
                    break

            if not done:
                tasks.append((args, t))
            need_run_set.remove(bugid)

    print('tot tasks:', len(tasks))

    if need_run_set:
        print('!!! invalid tasks:', need_run_set)
        time.sleep(10)

    will_del = []
    for _, task in tasks:
        bugid = task['instance_id']
        for p in Path(args.output_path).glob(f'task_{bugid}*.log'):
            print('delete', p)
            will_del.append(p)
        for p in Path(args.patches_path).glob(f'{bugid}_*.patch'):
            print('delete', p)
            will_del.append(p)
        for p in Path(args.log_path).glob(f'{bugid}_*.json'):
            print('delete', p)
            will_del.append(p)

    #if will_del:
    #    print(f'will delete {len(will_del)} files in 10 seconds')
    #    time.sleep(10)
    #    for p in will_del:
    #        p.unlink()

    random.seed(666)
    random.shuffle(tasks)

    if len(tasks) <= 1: # use single process and disable file logging if only one job
        num_processes = 0

    install_cleanup_handlers()
    Sandbox.sweep_orphans()

    task_queue = multiprocessing.Queue()

    pool = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(task_queue, lock, False))
        p.start()
        pool.append(p)

    try:
        for task in tasks:
            task_queue.put(task)

        for _ in range(num_processes):
            task_queue.put(None)

        if num_processes==0:
            task_queue.put(None)
            worker(task_queue, lock, True)

        for p in pool:
            p.join()
    except KeyboardInterrupt:
        print('interrupted, stopping workers and sweeping containers')
        for p in pool:
            p.terminate()
        for p in pool:
            p.join(timeout=15)
            if p.is_alive():
                p.kill()
        Sandbox.sweep_orphans()
        raise
    else:
        Sandbox.sweep_orphans()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to handle paths.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name to use")
    parser.add_argument("--log_path", required=True, help="Path to the log directory")
    parser.add_argument("--patches_path", required=True, help="Path to the patches directory")
    parser.add_argument("--output_path", required=True, help="Path to the output directory")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.patches_path):
        os.makedirs(args.patches_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # process_instances(args)
    from multiprocessing import Lock
    l = Lock()
    main(args, l)