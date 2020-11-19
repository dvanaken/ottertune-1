import json
import os
from collections import OrderedDict

from django.core.management.base import BaseCommand

from website.models import Result, SessionKnobManager
from website.utils import JSONUtil

cnf_start = """
# Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

#
# The MySQL  Server configuration file.
#
# For explanations see
# http://dev.mysql.com/doc/mysql/en/server-system-variables.html

[mysqld]
pid-file        = /var/run/mysqld/mysqld.pid
socket          = /var/run/mysqld/mysqld.sock
datadir         = /var/lib/mysql
secure-file-priv= NULL

# Custom config should go here
!includedir /etc/mysql/conf.d/

innodb_monitor_enable = all
skip-log-bin

# configurations recommended by ottertune:
[mysqld]
"""

class Command(BaseCommand):
    help = 'Dump configurations for results.'

    def add_arguments(self, parser):
        parser.add_argument(
            'result_ids',
            metavar='RESULT_IDS',
            type=int,
            nargs='+',
            help='Result IDs.')
        parser.add_argument(
            '-d',
            '--directory',
            metavar='DIRECTORY',
            default='configs',
            help='Output directory. Default: ./configs')
    def handle(self, *args, **options):
        result_ids = options['result_ids']
        outdir = options['directory']
        os.makedirs(outdir, exist_ok=True)
        session_tunable_knobs = {}
        for res_id in result_ids:
            result = Result.objects.get(id=res_id)
            knob_data = JSONUtil.loads(result.knob_data.knobs)
            session = result.session
            tunable_knobs = session_tunable_knobs.get(session.id, None)
            if not tunable_knobs:
                tunable_knobs = sorted(SessionKnobManager.get_knob_min_max_tunability(session, tunable_only=True).keys())
                session_tunable_knobs[session.id] = tunable_knobs
            config = OrderedDict([(k.replace('global.', ''), knob_data[k]) for k in tunable_knobs])
            cnf = cnf_start + ''.join('{} = {}\n'.format(k, v) for k, v in config.items())
            with open(os.path.join(outdir, '{}.json'.format(res_id)), 'w') as f:
                json.dump(config, f, indent=4)
            with open(os.path.join(outdir, '{}.cnf'.format(res_id)), 'w') as f:
                f.write(cnf)

        self.stdout.write(self.style.SUCCESS("\nSuccessfully saved configs.\n"))
