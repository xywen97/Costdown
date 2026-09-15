import atexit
import docker
import pexpect
import time
import subprocess
import os
import random
import re
import signal

from docker.models.containers import Container

SANDBOX_LABEL = 'agentdiet.sandbox'
INSTANCE_LABEL = 'agentdiet.instance'
PID_LABEL = 'agentdiet.pid'
CONTAINER_NAME_PREFIX = 'sweb.sbx.'

_CLEANUP_HANDLERS_INSTALLED = False


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Sandbox:
    def __init__(self, namespace, name, tag, instance):
        self.namespace = namespace
        self.name = name
        self.tag = tag
        self.client = docker.from_env()
        self.container: Container | None = None
        self._container_id = None
        self.shell = None
        self.commit_id = instance["base_commit"]
        self.instance_id = instance["instance_id"]
        self.shell_ready_ts = 0
        self.registered_checkpoints = []

        self.custom_cwd = instance.get('custom_cwd', None)
        print('custom_cwd', self.custom_cwd)
        atexit.register(self._atexit_cleanup)

    def get_project_path(self):
        project_path = self.container.exec_run("pwd").output.decode(errors='replace').strip()
        return project_path

    def apply_patch(self, patch, project_path):
        random_integer = str(random.randint(1, 10000))
        with open(f"/tmp/{random_integer}.diff", "w", encoding="utf-8") as file:
            file.write(patch)
        copy_db_cmd = f"docker cp /tmp/{random_integer}.diff {self.container.name}:{project_path}/patch.diff"
        subprocess.run(copy_db_cmd, check=True, shell=True)
        apply_command = f"git apply --ignore-space-change --ignore-whitespace {project_path}/patch.diff"
        output = self.container.exec_run(cmd = apply_command, workdir = project_path).output.decode(errors='replace').strip()
        print("git_apply: ", output)
        return output
        


    def get_file_content(self, file_path, start_line = None, end_line = None):
        file_name = os.path.basename(file_path)
        copy_db_cmd = f"docker cp {self.container.name}:{file_path} /tmp/{file_name}"
        subprocess.run(copy_db_cmd, check=True, shell=True)
        if not os.path.exists(f"/tmp/{file_name}"):
            print(f"Error Occurred: {copy_db_cmd} Failed!")
            return None
        
        snippet_lines = []
        with open(f"/tmp/{file_name}", 'r', encoding='utf-8') as f:
            code = f.read()
            lines = code.split('\n')
            if start_line < 0:
                start_line = 1
            if end_line >= len(lines):
                end_line = len(lines)

            snippet_lines = [f"【{i + start_line + 1}】{line}" for i, line in enumerate(lines[start_line:end_line + 1])]

        subprocess.run(f"rm /tmp/{file_name}", check=True, shell=True)
        return '\n'.join(snippet_lines)


    def _container_labels(self):
        return {
            SANDBOX_LABEL: '1',
            INSTANCE_LABEL: str(self.instance_id or ''),
            PID_LABEL: str(os.getpid()),
        }

    def _container_name(self):
        inst = re.sub(r'[^a-zA-Z0-9_.-]', '-', str(self.instance_id or 'unknown'))[:80]
        return f'{CONTAINER_NAME_PREFIX}{inst}.{os.getpid()}.{random.randint(1000, 999999)}'

    def _remember_container(self, container):
        self.container = container
        self._container_id = container.id

    def _run_container(self, image, **extra):
        last_err = None
        for _ in range(3):
            kwargs = dict(
                detach=True,
                tty=True,
                stdin_open=True,
                privileged=True,
                name=self._container_name(),
                labels=self._container_labels(),
            )
            kwargs.update(extra)
            try:
                container = self.client.containers.run(image, **kwargs)
                self._remember_container(container)
                return container
            except docker.errors.APIError as e:
                last_err = e
                if 'Conflict' not in str(e) and 'already in use' not in str(e):
                    raise
        raise last_err

    def start_container_build(self):
        image = f"{self.namespace}/{self.name}:{self.tag}"
        self._run_container(image)
        print(f"Container {self.container.short_id} started with image {image}")

    def start_container(self):
        self.destroy_all_checkpoints() # cleanup leftover checkpoints

        image = f"{self.namespace}/{self.name}:{self.tag}"
        try:
            self.client.images.get(image)
        except docker.errors.ImageNotFound:
            print(f"Pulling image {image} ...")
            self.client.images.pull(f"{self.namespace}/{self.name}", tag=self.tag)
            print(f"Pulled image {image}")

        try:
            #host_path = '/tmp'
            #container_path = '/tmp'
            self._run_container(
                image,
                **({'working_dir': self.custom_cwd} if self.custom_cwd else {}),
                #volumes={host_path: {'bind': container_path, 'mode': 'rw'}},
            )
            print(f"Container {self.container.short_id} started with image {image}")
            _ = self.container.exec_run(cmd="mkdir -p /home/swe-bench/conda_envs/")
            current_file_path = os.path.abspath(__file__)
            current_directory = os.path.dirname(current_file_path)
            project_directory = os.path.dirname(current_directory)
            cmd = f"chmod -R 777 {project_directory}/tools && docker cp {project_directory}/tools {self.container.name}:/home/swe-bench"
            subprocess.run(cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # install_res = self.container.exec_run(cmd="conda create -p /home/swe-bench/conda_envs/py312/ python=3.12")
            # print('install_res: ', install_res)
            copy_python_cmd = f"docker cp ~/miniconda3/envs/py312 {self.container.name}:/home/swe-bench/conda_envs/py312/"
            subprocess.run(copy_python_cmd, check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if self.commit_id:
                checkout_res = self.container.exec_run(f"git checkout {self.commit_id}")
                print('checkout: ',checkout_res)

            self.spawn_bg_processes()
        except BaseException:
            print('start_container failed, cleaning up')
            self.stop_container(remove_image=False)
            raise

    def spawn_bg_processes(self):
        self.start_shell()

    def get_diff_result(self, project_path: str, base_commit=None):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                if not base_commit:
                    res = self.container.exec_run(f"/home/swe-bench/conda_envs/py312/bin/python3 /home/swe-bench/tools/get_diff.py -p {project_path}").output.decode(errors='replace')
                else:
                    res = self.container.exec_run(f"/home/swe-bench/conda_envs/py312/bin/python3 /home/swe-bench/tools/get_diff.py -p {project_path} -c {base_commit}").output.decode(errors='replace')
                    print(base_commit)
                    print("diff res: ", res)
                return res
            except Exception as e:
                print(f"Attempt {retries + 1}: An error occurred while executing the command - {e}")
                time.sleep(5)
                retries += 1
        print(f"Failed to execute the command after {max_retries} attempts.")
        return ''
    def start_shell(self):
        if self.container:
            if self.shell and self.shell.isalive():
                try:
                    self.shell.close(force=True)
                except Exception: # pexpect.exceptions.ExceptionPexpect: Could not terminate the child.
                    pass
            command = f'docker exec -it {self.container.id} /bin/bash'
            self.shell = pexpect.spawn(command, maxread=200000)
            self.shell.expect([r'\$ ', r'# '], timeout=10)
        else:
            raise Exception("Container not started. Call start_container() first.")
    def get_session(self):
        self.start_shell()
        class Session:
            def __init__(self, sandbox):
                self.sandbox = sandbox
            def execute(self, command, timeout=180):
                delay = self.sandbox.shell_ready_ts - time.time()
                if delay>0:
                    time.sleep(delay)

                try:
                    if command[-1] != '&':
                        self.sandbox.shell.sendline(command + " && sleep 0.5")
                    else:
                        self.sandbox.shell.sendline(command)
                    before = ''
                    try_limit = 5
                    current_try = 0
                    self.sandbox.shell.before = b''
                    self.sandbox.shell.after = b''
                    self.sandbox.shell.buffer = b''
                    time.sleep(.5)
                    self.sandbox.shell.expect([r'swe-bench@.*:.*\$ ', r'root@.*:.*# '], timeout)
                    output = self.sandbox.shell.before.decode('utf-8', errors='replace') + self.sandbox.shell.after.decode('utf-8', errors='replace') + self.sandbox.shell.buffer.decode('utf-8', errors='replace')

                    #output = output.rpartition('')
                    output_lines = output.split('\r\n')
                    if len(output_lines) > 1:
                        output_lines = output_lines[1:-1]
                    # result_message = '### Observation: ' + '\n'.join(output_lines)
                    result_message = '\n'.join(output_lines).replace("\x1b[?2004l\r", "")
                    # truncation_length = 5000
                    # if len(result_message) > truncation_length:
                    #     return result_message[:truncation_length] + "\n...[Truncation]"
                    return result_message
                except pexpect.TIMEOUT:
                    partial_output = ''
                    if isinstance(self.sandbox.shell.before, bytes):
                        partial_output += self.sandbox.shell.before.decode('utf-8', errors='replace')
                    if isinstance(self.sandbox.shell.after, bytes):
                        partial_output += self.sandbox.shell.after.decode('utf-8', errors='replace')
                    if isinstance(self.sandbox.shell.buffer, bytes):
                        partial_output += self.sandbox.shell.buffer.decode('utf-8', errors='replace')
                    partial_output_lines = partial_output.split('\n')
                    if len(partial_output_lines) > 1:
                        partial_output_lines = partial_output_lines[1:-1]
                        partial_output = '\n'.join(partial_output_lines)
                    return f"Command timed out after {timeout} seconds. Partial output:\n + {partial_output}"
            def close(self):
                if self.sandbox.shell:
                    try:
                        self.sandbox.shell.sendline('exit')
                        self.sandbox.shell.expect(pexpect.EOF)
                    except Exception:
                        pass
                    self.sandbox._close_shell()
        return Session(self)

    def _atexit_cleanup(self):
        try:
            self.stop_container(remove_image=False)
        except Exception as e:
            print(f'atexit sandbox cleanup skipped: {e}')

    def _close_shell(self):
        shell = self.shell
        self.shell = None
        if not shell:
            return
        try:
            if shell.isalive():
                shell.close(force=True)
        except Exception as e:
            print(f'shell close skipped: {e}')
            pid = getattr(shell, 'pid', None)
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def _remove_container_id(self, container_id):
        if not container_id:
            return
        short = container_id[:12]
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=True)
            print(f"Container {short} stopped and removed")
            return
        except docker.errors.NotFound:
            print(f"Container {short} already gone")
            return
        except Exception as e:
            print(f"docker-py remove failed ({short}): {e}, trying docker rm -f")
        subprocess.run(
            ['docker', 'rm', '-f', container_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Container {short} force-removed via docker rm -f")

    def _maybe_remove_image(self):
        image = f"{self.namespace}/{self.name}:{self.tag}"
        try:
            users = self.client.containers.list(all=True, filters={'ancestor': image})
            if users:
                print(f"Image remove skipped ({image}): still used by {len(users)} container(s)")
                return
            self.client.images.remove(image, force=False)
            print(f"Image {image} removed")
        except docker.errors.ImageNotFound:
            pass
        except Exception as e:
            print(f"Image remove skipped ({image}): {e}")

    def stop_container(self, remove_image: bool = True):
        # pexpect close used to raise and skip docker rm, which leaked
        # idle eval containers across the 3-worker runs.
        self._close_shell()
        container_id = self._container_id
        if not container_id and self.container is not None:
            container_id = getattr(self.container, 'id', None)
        self.container = None
        self._container_id = None
        self._remove_container_id(container_id)

        if remove_image:
            self._maybe_remove_image()

    def copy_to_host(self, docker_path, host_path):
        copy_cmd = f"docker cp {self.container.name}:{docker_path} {host_path}"
        subprocess.run(copy_cmd, check=True, shell=True)

    def make_checkpoint(self):
        image = f"ckpt-{self.name}-{self.tag}"
        ckpt_tag = f'{int(time.time())}-{int(random.random()*1000000)}'
        self.container.commit(image, ckpt_tag, pause=False)

        self.registered_checkpoints.append(ckpt_tag)
        return ckpt_tag

    def restore_checkpoint(self, ckpt_tag):
        self.stop_container(remove_image=False)
        image = f"ckpt-{self.name}-{self.tag}:{ckpt_tag}"
        self._run_container(image)
        self.spawn_bg_processes()

    def destroy_all_checkpoints(self):
        print(f'!! deleting {len(self.registered_checkpoints)} checkpoints')
        for ckpt_tag in self.registered_checkpoints:
            image = f"ckpt-{self.name}-{self.tag}:{ckpt_tag}"
            try:
                self.client.images.remove(image, force=True)
            except Exception as e:
                print(f'checkpoint remove skipped ({image}): {e}')

        self.registered_checkpoints.clear()

    @staticmethod
    def sweep_orphans():
        """Remove sandbox containers whose creating process is already dead."""
        try:
            client = docker.from_env()
        except Exception as e:
            print(f'Orphan sweep skipped: {e}')
            return

        seen = set()
        containers = []
        for flt in ({'label': f'{SANDBOX_LABEL}=1'}, {'name': CONTAINER_NAME_PREFIX}):
            try:
                for container in client.containers.list(all=True, filters=flt):
                    if container.id not in seen:
                        seen.add(container.id)
                        containers.append(container)
            except Exception as e:
                print(f'Orphan sweep list failed ({flt}): {e}')

        for container in containers:
            pid_s = (container.labels or {}).get(PID_LABEL, '')
            try:
                pid = int(pid_s)
            except (TypeError, ValueError):
                pid = None
            if pid is not None and _pid_is_alive(pid):
                continue
            print(f'Sweeping orphan sandbox container {container.name} ({container.short_id}) pid={pid_s!r}')
            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
            except Exception as e:
                print(f'Orphan sweep failed for {container.short_id}: {e}')
                subprocess.run(
                    ['docker', 'rm', '-f', container.id],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


def install_cleanup_handlers():
    global _CLEANUP_HANDLERS_INSTALLED
    if _CLEANUP_HANDLERS_INSTALLED:
        return
    _CLEANUP_HANDLERS_INSTALLED = True
    atexit.register(Sandbox.sweep_orphans)

    def _on_term(signum, frame):
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        pass

if __name__ == "__main__":
    sandbox = Sandbox("mswebench", "ponylang_m_ponyc", "pr-2007", {'base_commit': None, 'instance_id': None})
    sandbox.start_container_build()
    session = sandbox.get_session()
    print('-----')
    output = session.execute("ls")
    print(output)
    print('-----')
    output = session.execute("sleep 70")
    print(output)
    print('-----')
    session = sandbox.get_session()
    output = session.execute("ls")
    print(output)
    # output = session.execute("cd astropy")
    # print(output)
    # output = session.execute("ls")
    # print(output)
    # output = session.execute("conda env list")
    # print(output)
    # output = session.execute("cd miniconda3")
    # output = session.execute("ls")
    # print(output)
    # output = session.execute("pytest --no-header -rA --tb=no -p no:cacheprovider")
    # print(output)
    session.close()
    # session2 = sandbox.get_session()
    # output = session2.execute("pwd")
    # print(output)
    # session2.close()
    sandbox.stop_container()
