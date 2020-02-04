import os
import subprocess
import sys
import time

from fabric.api import local, quiet, settings

from celery.bin import celery, multi  # pylint: disable=import-error,no-name-in-module
from django.utils.termcolors import make_style
from djcelery.app import app
from djcelery.management.commands.celery import Command as DJCeleryCommand

_BASE = celery.CeleryCommand(app=app)

_CWD = os.getcwd()

_DEFAULTS = {
    'range': '1',
    'pidfile': os.path.join(_CWD, 'run', 'celery%n.pid'),
    'logfile': os.path.join(_CWD, 'log', 'celery.log'),
    'loglevel': 'DEBUG',
    'pool': 'threads',
    'beat': '[enabled]',
}

_FLAGS = {
    'range': tuple(),
    'pidfile': ('--pidfile=',),
    'logfile': ('-f', '--logfile='),
    'loglevel': ('-l', '--loglevel='),
    'pool': ('-P', '--pool='),
    'beat': ('--beat',),
}

multi.USAGE += """\

-=-----------------------------------------------------------------------------
ottertune defaults:
-=-----------------------------------------------------------------------------

    * <range>         {range}
    *     --pidfile   {pidfile}
    * -f, --logfile   {logfile}
    * -l, --loglevel  {loglevel}
    * -P, --pool      {pool}
    *     --beat      {beat}
-=-----------------------------------------------------------------------------
""".format(**_DEFAULTS)

_PROC_CMD = "ps auxww | grep '[c]elery worker' | awk '{print $2}'"
_KILL_CMD = _PROC_CMD + " | xargs kill -9"
_CMD_TIMEOUT = 30  # seconds


class Command(DJCeleryCommand):

    def __init__(self, stdout=None, stderr=None, no_color=False):
        super().__init__(stdout=stdout, stderr=stderr, no_color=no_color)
        self.style.COMMENT = make_style(fg='blue', opts=('bold',))
        self.style.META = make_style(fg='cyan', opts=('bold',))

    @staticmethod
    def _fix_path(argv, cur_idx):
        def _fix(_path):
            _path = os.path.realpath(_path)
            if not os.path.exists(os.path.dirname(_path)):
                with quiet():
                    local('mkdir -p {}'.format(os.path.dirname(path)))
            return _path

        path = None
        if cur_idx < len(argv):
            arg = argv[cur_idx]
            if arg.startswith('--'):
                flag, path = arg.split('=')
                path = _fix(path)
                argv[cur_idx] = '{}={}'.format(flag, path)
            elif arg.startswith('-'):
                if cur_idx + 1 < len(argv):
                    path = _fix(argv[cur_idx + 1])
                    argv[cur_idx + 1] = path
        return path

    def handle_default_options(self, argv):
        argv = super().handle_default_options(argv)

        if len(argv) > 3 and argv[2] == 'multi':
            command = argv[3]
            start_idx = 5 if command in ('get', 'expand') else 4
            check_opts = []
            values = []

            if command != 'help':
                check_opts.append('pidfile')
            if command in ('start', 'restart', 'show', 'get'):
                check_opts += ['logfile', 'loglevel', 'pool', 'beat']

            i = start_idx
            while i < len(argv):
                arg = argv[i]
                if arg and arg[0] == '-':
                    for opt_name in check_opts:
                        found = None
                        flags = _FLAGS[opt_name]
                        for flag in flags:
                            if arg.startswith(flag):
                                if opt_name in ('pidfile', 'logfile'):
                                    self._fix_path(argv, i)
                                found = opt_name
                                break
                        if found:
                            check_opts.remove(found)
                    if len(arg) > 1 and arg[1] == '-':
                        i += 1
                    else:
                        i += 2
                else:
                    values.append(arg)
                    i += 1

            if not values:
                argv = argv[:start_idx] + [_DEFAULTS['range']] + argv[start_idx:]

            for opt_name in check_opts:
                flag = _FLAGS[opt_name][-1]
                default = str(_DEFAULTS[opt_name])
                if default.startswith('[') and default.endswith(']'):
                    default_opt = [flag]
                elif flag.startswith('--'):
                    default_opt = [flag + default]
                else:
                    default_opt = [flag, default]
                argv += default_opt

        return argv

    def _ls(self, pattern):  # pylint: disable=no-self-use
        out = ''
        if pattern:
            with settings(warn_only=True), quiet():
                res = local('ls -1 ' + pattern, capture=True)
            out = res.stdout.strip()
        return out

    def _procs(self):  # pylint: disable=no-self-use
        with settings(warn_only=True), quiet():
            res = local(_PROC_CMD, capture=True)
        out = res.stdout.strip()
        return out

    def run_from_argv(self, argv):
        argv = self.handle_default_options(argv)
        kill_cmd = _KILL_CMD

        base_command = argv[2] if len(argv) > 2 else None
        command = argv[3] if len(argv) > 3 else None

        pidfile, pattern = None, None
        if base_command == 'multi':
            for arg in argv:
                if '--pidfile=' in arg:
                    _, pidfile = arg.split('=')
                    pattern = pidfile
                    for exp in ('%h', '%n', '%d'):
                        pattern = pattern.replace(exp, '*')
                    break
            if pattern:
                kill_cmd += '; rm -f {}'.format(pattern)

        timeout_cmd = "sleep {}; {}".format(_CMD_TIMEOUT, kill_cmd)
        p = subprocess.Popen(timeout_cmd, shell=True)
        exit_code = 0

        try:
            _BASE.execute_from_commandline(
                ['{0[0]} {0[1]}'.format(argv)] + argv[2:],
            )
        except SystemExit as e:
            exit_code = e.code

        ret_code = p.poll()
        if ret_code is None:
            p.terminate()  # Terminate timeout command

        if base_command == 'multi':
            if command in ('stopwait', 'kill'):
                with settings(warn_only=True), quiet():
                    local(kill_cmd)

            time.sleep(5)
            pidfiles = self._ls(pattern)
            pids = self._procs()
            if pidfiles:
                self.stdout.write(self.style.COMMENT("> pidfile(s): {}".format(pidfiles)))
            if pids:
                self.stdout.write(self.style.COMMENT("> pid(s): {}".format(pids)))

        self.stdout.write(self.style.META("\n[exit {}]".format(exit_code)))
        sys.exit(exit_code)

    def handle(self, *args, **options):
        pass
