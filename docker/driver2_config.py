import os
from collections import OrderedDict

#==========================================================
#  HOST LOGIN
#==========================================================

# Location of the database host relative to this driver
# Valid values: local, remote, docker or remote_docker
HOST_CONN = 'remote_docker'

# The name of the Docker container for the target database
# (only required if HOST_CONN = docker)
CONTAINER_NAME = 'postgres'  # e.g., 'postgres_container'

# Host SSH login credentials (only required if HOST_CONN=remote)
LOGIN_NAME = os.environ['LOGIN_NAME']
LOGIN_HOST = os.environ['LOGIN_HOST']
LOGIN_PASSWORD = None
LOGIN_PORT = None  # Set when using a port other than the SSH default


#==========================================================
#  DATABASE OPTIONS
#==========================================================

# Postgres, Oracle or Mysql
DB_TYPE = 'postgres'

# Database version
DB_VERSION = '9.6'

# Name of the database
DB_NAME = os.environ['DB_NAME']

# Database username
DB_USER = 'postgres'

# Password for DB_USER
DB_PASSWORD = 'postgres'

# Database admin username (for tasks like restarting the database)
ADMIN_USER = DB_USER

# Database host address
DB_HOST = os.environ.get('DB_HOST', LOGIN_HOST)

# Database port
DB_PORT = '5432'

# If set to True, DB_CONF file is mounted to database container file
# Only available when HOST_CONN is docker or remote_docker
DB_CONF_MOUNT = True

# Path to the configuration file on the database server
# If DB_CONF_MOUNT is True, the path is on the host server, not docker
DB_CONF = os.environ['DB_CONF']

# Path to the directory for storing database dump files
DB_DUMP_DIR = os.environ['DB_DUMP_DIR']

# Base config settings to always include when installing new configurations
if DB_TYPE == 'mysql':
    BASE_DB_CONF = {
        'innodb_monitor_enable': 'all',
        # Do not generate binlog, otherwise the disk space will grow continuely during the tuning
        # Be careful about it when tuning a production database, it changes your binlog behavior.
        'skip-log-bin': None,
    }
elif DB_TYPE == 'postgres':
    BASE_DB_CONF = {
        'track_counts': 'on',
        'track_functions': 'all',
        'track_io_timing': 'on',
        'autovacuum': 'off',
    }
else:
    BASE_DB_CONF = None

if DB_TYPE == 'postgres':
    RESTORE_DB_CONF = {
        'track_counts': 'off',
        'track_functions': 'none',
        'track_io_timing': 'off',
        'work_mem': '64MB',
        'shared_buffers': '4GB',
        'maintenance_work_mem': '2GB',
        'full_page_writes': 'off',
        'wal_buffers': -1,
    }
else:
    RESTORE_DB_CONF = None

# Name of the device on the database server to monitor the disk usage, or None to disable
DATABASE_DISK = None

# Set this to a different database version to override the current version
OVERRIDE_DB_VERSION = os.environ.get('OVERRIDE_DB_VERSION', None)

# POSTGRES-SPECIFIC OPTIONS >>>
PG_DATADIR = os.environ['PG_DATADIR'] 

# ORACLE-SPECIFIC OPTIONS >>>
ORACLE_AWR_ENABLED = False
ORACLE_FLASH_BACK = True
RESTORE_POINT = 'tpcc_point'
RECOVERY_FILE_DEST = '/opt/oracle/oradata/ORCL'
RECOVERY_FILE_DEST_SIZE = '15G'


#==========================================================
#  DRIVER OPTIONS
#==========================================================

# Path to this driver
DRIVER_HOME = os.path.dirname(os.path.realpath(__file__))

# Path to the directory for storing results
RESULT_DIR = os.path.join(DRIVER_HOME, 'results')

# Set this to add user defined metrics
ENABLE_UDM = True

# Path to the User Defined Metrics (UDM), only required when ENABLE_UDM is True
UDM_DIR = os.path.join(DRIVER_HOME, 'userDefinedMetrics')

# Path to temp directory
TEMP_DIR = '/tmp/driver'

# Path to the directory for storing database dump files
if DB_DUMP_DIR is None:
    if HOST_CONN == 'local':
        DB_DUMP_DIR = os.path.join(DRIVER_HOME, 'dumpfiles')
        if not os.path.exists(DB_DUMP_DIR):
            os.mkdir(DB_DUMP_DIR)
    else:
        DB_DUMP_DIR = os.path.expanduser('~/')

# Reload the database after running this many iterations
RELOAD_INTERVAL = int(os.environ.get('RELOAD_INTERVAL', 10))

# The maximum allowable disk usage percentage. Reload the database
# whenever the current disk usage exceeds this value.
MAX_DISK_USAGE = 90

# Execute this many warmup iterations before uploading the next result
# to the website
WARMUP_ITERATIONS = 0

# Let the database initialize for this many seconds after it restarts
RESTART_SLEEP_SEC = int(os.environ.get('RESTART_SLEEP_SEC', 50))

#==========================================================
#  OLTPBENCHMARK OPTIONS
#==========================================================

# Path to OLTPBench directory
OLTPBENCH_HOME = os.path.expanduser('/app/oltpbench')

# Path to the OLTPBench configuration file
OLTPBENCH_CONFIG = os.environ['OLTPBENCH_CONFIG']

# Name of the benchmark to run
OLTPBENCH_BENCH = os.environ['OLTPBENCH_BENCH']


#==========================================================
#  CONTROLLER OPTIONS
#==========================================================

# Controller observation time, OLTPBench will be disabled for
# monitoring if the time is specified
CONTROLLER_OBSERVE_SEC = None

# Path to the controller directory
CONTROLLER_HOME = DRIVER_HOME + '/../controller'

# Path to the controller configuration file
CONTROLLER_CONFIG = os.path.join(CONTROLLER_HOME, 'config/postgres_config.json')


#==========================================================
#  LOGGING OPTIONS
#==========================================================

LOG_LEVEL = 'DEBUG'

# Path to log directory
LOG_DIR = os.path.join(DRIVER_HOME, 'log')

# Log files
DRIVER_LOG = os.path.join(LOG_DIR, 'driver.log')
OLTPBENCH_LOG = os.path.join(LOG_DIR, 'oltpbench.log')
CONTROLLER_LOG = os.path.join(LOG_DIR, 'controller.log')

ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', None) 


#==========================================================
#  WEBSITE OPTIONS
#==========================================================

# OtterTune website URL
WEBSITE_URL = os.environ.get('WEBSITE_URL', 'http://127.0.0.1:8000')

# Code for uploading new results to the website
UPLOAD_CODE = os.environ['UPLOAD_CODE']

# Name of the workload for this tuning session
# If unset or None, OLTPBENCH_BENCH is used instead
WORKLOAD_NAME = os.environ.get('WORKLOAD_NAME', None)

if DB_TYPE == 'postgres':
    KNOB_RANGES = OrderedDict([
        ('global.effective_cache_size', {
            'minval': 262144,
            'maxval': 30000000000,
            'tunable': True,
        }),
        ('global.maintenance_work_mem', {
            'minval': 2097152,
            'maxval': 549755813,
            'tunable': True,
        }),
        ('global.max_wal_size', {
            'minval': 671088640,
            'maxval': 17179869184,
            'tunable': True,
        }),
        ('global.max_worker_processes', {
            'minval': 0,
            'maxval': 16,
            'tunable': True,
        }),
        ('global.shared_buffers', {
            'minval': 4194304,
            'maxval': 27487790694,
            'tunable': True,
        }),
        ('global.temp_buffers', {
            'minval': 26214400,
            'maxval': 549755813,
            'tunable': True,
        }),
        ('global.wal_buffers', {
            'minval': 65536,
            'maxval': 2147475456,
            'tunable': True,
        }),
        ('global.work_mem', {
            'minval': 2097152,
            'maxval': 549755813,
            'tunable': True,
        }),
        # ('global.backend_flush_after', {
        #     'minval': 0,
        #     'maxval': 2097152,
        #     'tunable': True,
        # }),
        ('global.bgwriter_delay', {
            'minval': 10,
            'maxval': 10000,
            'tunable': True,
        }),
        # ('global.bgwriter_flush_after', {
        #     'minval': 0,
        #     'maxval': 2097152,
        #     'tunable': True,
        # }),
        ('global.bgwriter_lru_maxpages', {
            'minval': 0,
            'maxval': 1000,
            'tunable': True,
        }),
        ('global.bgwriter_lru_multiplier', {
            'minval': 0.0,
            'maxval': 10.0,
            'tunable': True,
        }),
        ('global.checkpoint_completion_target', {
            'minval': 0.0,
            'maxval': 1.0,
            'tunable': True,
        }),
        # ('global.checkpoint_flush_after', {
        #     'minval': 0,
        #     'maxval': 2097152,
        #     'tunable': True,
        # }),
        ('global.checkpoint_timeout', {
            'minval': 300000,
            'maxval': 3600000,
            'tunable': True,
        }),
        ('global.commit_delay', {
            'minval': 0,
            'maxval': 10000,
            'tunable': True,
        }),
        ('global.commit_siblings', {
            'minval': 0,
            'maxval': 100,
            'tunable': True,
        }),
        ('global.default_statistics_target', {
            'minval': 1,
            'maxval': 10000,
            'tunable': True,
        }),
        ('global.effective_io_concurrency', {
            'minval': 0,
            'maxval': 1000,
            'tunable': True,
        }),
        ('global.from_collapse_limit', {
            'minval': 1,
            'maxval': 100,
            'tunable': True,
        }),
        ('global.join_collapse_limit', {
            'minval': 1,
            'maxval': 100,
            'tunable': True,
        }),
        ('global.max_parallel_workers_per_gather', {
            'minval': 0,
            'maxval': 8,
            'tunable': True,
        }),
        ('global.random_page_cost', {
            'minval': 1.0,
            'maxval': 4.0,
            'tunable': True,
        }),
        ('global.wal_sync_method', {
            'minval': None,
            'maxval': None,
            'tunable': True,
        }),
        ('global.wal_writer_delay', {
            'minval': 1,
            'maxval': 10000,
            'tunable': True,
        }),
        # ('global.wal_writer_flush_after', {
        #     'minval': 0,
        #     'maxval': 2097152,
        #     'tunable': True,
        # }),
        #('', {
        #    'minval': ,
        #    'maxval': ,
        #    'tunable': True,
        #}),
    ])
else:
    KNOB_RANGES = None
