#!/usr/bin/env python3
"""Own each terminal's process tree and stop only this launch session (Linux)."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SCHEMA = 'hospital_terminal_session_v1'
SLOTS = tuple(f'T{i}' for i in range(1, 7))
TERM_GRACE = 3.0
KILL_GRACE = 2.0


def atomic_json(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def create_session():
    path = Path(tempfile.mkdtemp(prefix='hospital_terminals_'))
    atomic_json(path/'session.json', dict(schema=SCHEMA, uid=os.getuid()))
    return path


def validate_session(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError('Invalid session owner or path')
    path = path.resolve(strict=True)
    meta = json.loads((path/'session.json').read_text())
    if meta != dict(schema=SCHEMA, uid=os.getuid()):
        raise ValueError('Invalid terminal session')
    return path


@contextlib.contextmanager
def locked(path, nonblocking=False):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def process_info(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start=fields[19], parent=int(fields[1]), state=fields[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def identity(info):
    return dict(pid=info['pid'], start=info['start'])


def live(record):
    info = process_info(record['pid'])
    return info and info['start'] == record['start'] and info['state'] not in ('Z', 'X')


def descendants(roots):
    table = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            info = process_info(int(path.name))
            if info:
                table[info['pid']] = info
    owned = {r['pid'] for r in roots if r['pid'] in table and table[r['pid']]['start'] == r['start']}
    while True:
        found = {pid for pid, info in table.items() if info['parent'] in owned}
        if found <= owned:
            break
        owned |= found
    return [identity(table[pid]) for pid in sorted(owned) if table[pid]['state'] not in ('Z', 'X')]


def send(record, sig):
    # Bind the signal to this exact process, even if a PID is recycled.
    try:
        fd = os.pidfd_open(record['pid'])
        try:
            if live(record):
                signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)
    except ProcessLookupError:
        pass


def run(session, slot, command, grace=None, terminal_pid=None):
    session = validate_session(session)
    if slot not in SLOTS or not command:
        raise ValueError('A T1-T6 slot and command are required')
    grace = (40.0 if slot == 'T6' else 15.0) if grace is None else grace
    # Detached grandchildren are reparented here, so closing a launcher does
    # not lose ownership of Gazebo/ROS children that create their own session.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot track detached child processes')
    stopping = [False]
    def interrupted(sig, frame):
        stopping[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    me = identity(process_info(os.getpid()))
    terminal = None
    if terminal_pid is not None:
        info = process_info(terminal_pid)
        if info is None or terminal_pid == os.getpid():
            raise ValueError('Invalid terminal process')
        terminal = identity(info)
        if me not in descendants([terminal]):
            raise ValueError('Terminal is not an ancestor of this command')
    record_path = session/f'{slot}.json'
    with locked(session/f'{slot}.lock', nonblocking=True):
        with locked(session/'launch.lock'):
            if (session/'STOPPING').exists():
                print('[SESSION_STOPPED] No command started.', flush=True)
                return 0
            record = dict(supervisor=me, children=[], terminal=terminal, state='running')
            atomic_json(record_path, record)
            try:
                child = subprocess.Popen(command, start_new_session=True)
            except BaseException:
                record['state'] = 'finished'
                atomic_json(record_path, record)
                raise
            info = process_info(child.pid)
            record['children'] = [identity(info)] if info else []
            atomic_json(record_path, record)
        started_stopping = None
        sent = set()
        result = None
        while True:
            # Reap both the direct command and any adopted grandchildren.
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                if pid == child.pid:
                    result = os.waitstatus_to_exitcode(status)
                    child.returncode = result
            children = [r for r in descendants([me]) if r != me]
            if children != record['children']:
                record['children'] = children
                atomic_json(record_path, record)
            if result is not None and not children:
                record['state'] = 'finished'
                atomic_json(record_path, record)
                return 130 if stopping[0] else (result if result >= 0 else 128-result)
            if stopping[0]:
                if started_stopping is None:
                    started_stopping = time.monotonic()
                elapsed = time.monotonic() - started_stopping
                sig = signal.SIGINT if elapsed < grace else signal.SIGTERM if elapsed < grace+TERM_GRACE else signal.SIGKILL
                for r in children:
                    key = (r['pid'], r['start'], sig)
                    if key not in sent:
                        send(r, sig)
                        sent.add(key)
            time.sleep(.1)


def records(session, slots):
    result = []
    for slot in slots:
        path = session/f'{slot}.json'
        if path.is_file():
            result.append(json.loads(path.read_text()))
    return result


def stop_phase(session, slots, grace):
    began = time.monotonic()
    sent = set()
    while True:
        pending = []
        elapsed = time.monotonic() - began
        for record in records(session, slots):
            # Saved children also allow cleanup if a supervisor exited early.
            terminal = record.get('terminal')
            roots = [record['supervisor'], *record['children']]
            if terminal:
                roots.append(terminal)
            owned = descendants(roots)
            pending += owned
            # The original tab shell remains owned after a command finishes,
            # including any later manual resume command in that same tab.
            if terminal and live(terminal) and owned == [terminal]:
                send(terminal, signal.SIGHUP)
                continue
            if elapsed < grace and live(record['supervisor']):
                targets = [record['supervisor']]
                sig = signal.SIGINT
            else:
                targets = [r for r in owned if r != terminal] if elapsed < grace else owned
                sig = signal.SIGINT if elapsed < grace else signal.SIGTERM if elapsed < grace+TERM_GRACE else signal.SIGKILL
            for r in targets:
                key = (r['pid'], r['start'], sig)
                if key not in sent:
                    send(r, sig)
                    sent.add(key)
        if not pending:
            return
        if elapsed > grace+TERM_GRACE+KILL_GRACE:
            raise RuntimeError('Processes still running: '+str([r['pid'] for r in pending]))
        time.sleep(.1)


def stop(session, grace=None):
    session = validate_session(session)
    with locked(session/'stop.lock', nonblocking=True):
        with locked(session/'launch.lock'):
            (session/'STOPPING').touch()
        errors = []
        for slots, seconds, message in [(['T6'], 40.0, 'Stopping T6...'), (list(SLOTS[:-1]), 15.0, 'Stopping T1-T5...')]:
            print(message, flush=True)
            try:
                stop_phase(session, slots, seconds if grace is None else grace)
            except RuntimeError as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))
        print('[STOPPED] All tracked session processes have exited. Logs kept.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('create')
    p = sub.add_parser('run')
    p.add_argument('--grace-seconds', type=float)
    p.add_argument('--terminal-pid', type=int)
    p.add_argument('session', type=Path)
    p.add_argument('slot', choices=SLOTS)
    p.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('stop')
    p.add_argument('--grace-seconds', type=float)
    p.add_argument('session', type=Path)
    args = parser.parse_args()
    if args.action == 'create':
        print(create_session())
        return 0
    if args.grace_seconds is not None and not 0 < args.grace_seconds <= 60:
        parser.error('Grace period must be in (0, 60]')
    if args.action == 'stop':
        return stop(args.session, args.grace_seconds)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    return run(args.session, args.slot, command, args.grace_seconds, args.terminal_pid)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print('[SESSION_ERROR]', str(exc), flush=True)
        raise SystemExit(2)
