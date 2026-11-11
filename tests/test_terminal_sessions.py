"""Real disposable subprocesses only; no ROS, Gazebo or desktop launches."""
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/'scripts/terminal_launcher/session_processes.py'
LAUNCHER = HELPER.with_name('launcher.sh')
spec = importlib.util.spec_from_file_location('terminal_sessions', HELPER)
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='session_test_')
        self.folder = Path(self.temp.name)
        self.session = self.folder/'session'
        self.session.mkdir()
        manager.atomic_json(self.session/'session.json', dict(schema=manager.SCHEMA, uid=os.getuid()))
        self.processes = []

    def tearDown(self):
        try:
            self.stop()
        finally:
            for process in self.processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)
            self.temp.cleanup()

    def wait_for(self, condition):
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.025)
        self.fail('Condition was not reached in time')

    def start(self, slot, command, env=None):
        p = subprocess.Popen([sys.executable, '-B', str(HELPER), 'run', '--grace-seconds', '.3',
            str(self.session), slot, '--', *command], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        self.processes.append(p)
        self.wait_for(lambda: (self.session/f'{slot}.json').exists())
        return p

    def stop(self):
        return subprocess.run([sys.executable, '-B', str(HELPER), 'stop', '--grace-seconds', '.3',
            str(self.session)], text=True, capture_output=True, timeout=12, check=True)

    def test_stop_t6_first_and_preserve_unrelated_process(self):
        events = self.folder/'events'
        code = '''import signal,sys,time
from pathlib import Path
slot,ready,events=sys.argv[1:]
def stop(sig,frame):
    with open(events,'a') as stream: stream.write(slot+'\\n')
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
Path(ready).touch()
while True: time.sleep(.1)
'''
        for slot in ('T1', 'T6'):
            ready = self.folder/slot
            self.start(slot, [sys.executable, '-B', '-c', code, slot, str(ready), str(events)])
            self.wait_for(ready.exists)
        unrelated = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
        self.processes.append(unrelated)
        result = self.stop()
        self.assertIn('[STOPPED]', result.stdout)
        self.assertEqual(events.read_text().splitlines(), ['T6', 'T1'])
        self.assertIsNone(unrelated.poll())

    def test_detached_child_is_owned_and_forcibly_stopped(self):
        ready = self.folder/'detached.json'
        child_code = '''import json,os,signal,time
from pathlib import Path
signal.signal(signal.SIGINT,signal.SIG_IGN)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
Path(%r).write_text(json.dumps(dict(pid=os.getpid(),start=fields[19])))
while True: time.sleep(.1)
''' % str(ready)
        parent_code = 'import subprocess,sys; subprocess.Popen([sys.executable,"-B","-c",'+repr(child_code)+'],start_new_session=True,env={})'
        supervisor = self.start('T2', [sys.executable, '-B', '-c', parent_code])
        self.wait_for(ready.exists)
        child = json.loads(ready.read_text())
        self.assertIsNone(supervisor.poll())
        self.stop()
        self.assertFalse(manager.live(child))

    def test_pending_t6_and_late_start_cannot_launch_after_stop(self):
        config = self.folder/'config'
        config.mkdir()
        ran = self.folder/'ran'
        (config/'T6.sh').write_text('touch "'+str(ran)+'"\n')
        env = dict(os.environ, DN012_PROJECT=str(ROOT), TERMINAL_LAUNCHER_CONFIG=str(config))
        pending = self.start('T6', ['bash', str(LAUNCHER), 'run-command', 'T6'], env)
        self.assertIsNone(pending.poll())
        self.stop()
        self.assertFalse(ran.exists())
        late = subprocess.run([sys.executable, '-B', str(HELPER), 'run', str(self.session), 'T1', '--',
            sys.executable, '-B', '-c', 'from pathlib import Path; Path('+repr(str(ran))+').touch()'],
            capture_output=True, text=True, check=True)
        self.assertIn('[SESSION_STOPPED]', late.stdout)
        self.assertFalse(ran.exists())

    def test_stale_pid_identity_is_not_signalled(self):
        unrelated = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'])
        self.processes.append(unrelated)
        info = manager.process_info(unrelated.pid)
        stale = dict(pid=unrelated.pid, start=str(int(info['start'])+1))
        manager.atomic_json(self.session/'T1.json', dict(supervisor=stale, children=[], state='running'))
        self.stop()
        self.assertIsNone(unrelated.poll())

    def test_command_receives_terminal_input(self):
        command = [sys.executable, '-B', '-c', 'print("input="+input(),flush=True)']
        process = self.start('T6', command)
        output, _ = process.communicate('START\n', timeout=5)
        self.assertEqual(process.returncode, 0, output)
        self.assertIn('input=START', output)

    def test_tab_is_still_owned_after_initial_command_finishes(self):
        command = shlex.join([sys.executable, '-B', str(HELPER), 'run', '--terminal-pid'])
        command += ' $$ '+shlex.join([str(self.session), 'T3', '--', sys.executable, '-B', '-c', 'pass'])
        ready = self.folder/'later_command'
        later = 'from pathlib import Path; import time; Path('+repr(str(ready))+').touch(); time.sleep(30)'
        command += '; exec '+shlex.join([sys.executable, '-B', '-c', later])
        terminal = subprocess.Popen(['bash', '-c', command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.processes.append(terminal)
        self.wait_for(ready.exists)
        self.assertIsNone(terminal.poll())
        self.stop()
        terminal.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
