import json
import MySQLdb
import os
import time
from collections import OrderedDict

# 1  apt-get update
# 2  apt-get install -y python3-dev mysql-client libmysqlclient-dev python-mysqldb gcc g++
# 4  pip3 install mysqlclient==1.3.12
# 9  apt purge -y --autoremove gcc g++

class MySQLCollector:

    Q1 = "select @@global.version"
    Q2 = "show variables"
    Q3 = "show global status"
    Q4 = "select name, count from information_schema.innodb_metrics where subsystem='transaction'"

    def __init__(self, passwd, host='127.0.0.1', user='root', port=3306, db='', result_dir='', **kwargs):
        port = int(port)
        self._conn = MySQLdb.connect(passwd=passwd, host=host, user=user, port=port, db=db, **kwargs)
        self.passwd = passwd
        self.host = host
        self.user = user
        self.port = port
        self.db = db or ''
        self.result_dir = result_dir or ''
        self.conn_meta = dict(kwargs)
        cur = self._conn.cursor()
        cur.execute(self.Q1)
        self.full_version = cur.fetchone()[0]
        self.version = '.'.join(self.full_version.split('.')[:2])

    @property
    def conn(self):
        if self._conn is None:
            self._conn = MySQLdb.connect(
                passwd=self.passwd,
                host=self.host,
                user=self.user,
                port=self.port,
                db=self.db,
                **self.conn_meta)
        return self._conn

    def get_version(self):
        cur = self.conn.cursor()
        cur.execute(self.Q1)
        return cur.fetchone()

    @staticmethod
    def _wrap(d):
        return OrderedDict([
            ('global', {'global': d}),
            ('local', None),
        ])

    def get_knobs(self):
        cur = self.conn.cursor()
        cur.execute(self.Q2)
        knobs = OrderedDict([(k.lower(), v) for k, v in cur])
        return knobs

    def get_metrics(self):
        cur = self.conn.cursor()
        cur.execute(self.Q3)
        m1 = [(k.lower(), v) for k, v in cur]
        cur.execute(self.Q4)
        m2 = [(k.lower(), v) for k, v in cur]
        metrics = OrderedDict(sorted(m1 + m2))
        return metrics

    def collect_knobs(self, result_dir='', filename='knobs.json'):
        result_dir = result_dir or self.result_dir
        savepath = os.path.join(result_dir, filename)
        knobs = self.get_knobs()
        knobs = self._wrap(knobs)
        with open(savepath, 'w') as f:
            json.dump(knobs, f, indent=4)

    def collect_metrics(self, result_dir='', filename='metrics.json'):
        result_dir = result_dir or self.result_dir
        savepath = os.path.join(result_dir, filename)
        metrics = self.get_metrics()
        metrics = self._wrap(metrics)
        with open(savepath, 'w') as f:
            json.dump(metrics, f, indent=4)

    def close(self):
        try:
            self._conn.close()
        except:
            pass
        self._conn = None
