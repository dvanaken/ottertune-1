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
CONTAINER_NAME = os.environ.get('CONTAINER_NAME', 'container')  # e.g., 'postgres_container'

# Host SSH login credentials (only required if HOST_CONN=remote)
LOGIN_NAME = os.environ['LOGIN_NAME']
LOGIN_HOST = os.environ['LOGIN_HOST']
LOGIN_PASSWORD = None
LOGIN_PORT = None  # Set when using a port other than the SSH default


#==========================================================
#  DATABASE OPTIONS
#==========================================================

# Postgres, Oracle or Mysql
DB_TYPE = os.environ['DB_TYPE']

if DB_TYPE == 'mysql':
    _DEFAULT_DB_USER = 'root'
    _DEFAULT_DB_PORT = '3306'
elif DB_TYPE == 'postgres':
    _DEFAULT_DB_USER = 'postgres'
    _DEFAULT_DB_PORT = '5432'
else:  # oracle
    _DEFAULT_DB_USER = 'system'
    _DEFAULT_DB_PORT = '1521'

# Database version
DB_VERSION = os.environ['DB_VERSION']

# Name of the database
DB_NAME = os.environ['DB_NAME']

# Database username
DB_USER = os.environ.get('DB_USER', _DEFAULT_DB_USER)

# Password for DB_USER
DB_PASSWORD = os.environ['DB_PASSWORD']

# Database admin username (for tasks like restarting the database)
ADMIN_USER = DB_USER

# Database host address
DB_HOST = os.environ.get('DB_HOST', LOGIN_HOST)

# Database port
DB_PORT = os.environ.get('DB_PORT', _DEFAULT_DB_PORT)

# If set to True, DB_CONF file is mounted to database container file
# Only available when HOST_CONN is docker or remote_docker
DB_CONF_MOUNT = True

# Path to the configuration file on the database server
# If DB_CONF_MOUNT is True, the path is on the host server, not docker
DB_CONF = os.environ['DB_CONF']

# If set to True, DB_DUMP_DIR is mounted to database container file
# Only available when HOST_CONN is docker or remote_docker
DB_DUMP_MOUNT = True

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
elif DB_TYPE == 'mysql':
    RESTORE_DB_CONF = {
        'innodb_buffer_pool_size': '20G',
        'innodb_log_buffer_size': '256M',
        'innodb_log_file_size': '15G',
        'innodb_write_io_threads': 16,
        'innodb_flush_log_at_trx_commit': 0,
    }
else:
    RESTORE_DB_CONF = None

# Name of the device on the database server to monitor the disk usage, or None to disable
DATABASE_DISK = None

# Set this to a different database version to override the current version
OVERRIDE_DB_VERSION = os.environ.get('OVERRIDE_DB_VERSION', None)

# POSTGRES-SPECIFIC OPTIONS >>>
PG_DATADIR = os.environ.get('PG_DATADIR', None)
if DB_TYPE == 'postgres':
    assert PG_DATADIR

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

# Path to controller's latest result
CONTROLLER_DIR = os.path.join(DRIVER_HOME, 'latest_result')

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
WARMUP_ITERATIONS = 1

# Let the database initialize for this many seconds after it restarts
RESTART_SLEEP_SEC = int(os.environ.get('RESTART_SLEEP_SEC', 50))

# Seconds to wait for stop before killing the container
CONTAINER_RESTART_SEC = int(os.environ.get('CONTAINER_RESTART_SEC', 60))

#==========================================================
#  OLTPBENCHMARK OPTIONS
#==========================================================

# Path to OLTPBench directory
OLTPBENCH_HOME = os.path.expanduser('/app/oltpbench')

# Path to the OLTPBench configuration file
OLTPBENCH_CONFIG = os.environ['OLTPBENCH_CONFIG']

# Name of the benchmark to run
OLTPBENCH_BENCH = os.environ['OLTPBENCH_BENCH']

# Path to OLTPBench result directory
OLTPBENCH_RESULTS = os.environ.get(os.path.join(OLTPBENCH_HOME, 'results'))


### #==========================================================
### #  CONTROLLER OPTIONS
### #==========================================================
### 
### # Controller observation time, OLTPBench will be disabled for
### # monitoring if the time is specified
### CONTROLLER_OBSERVE_SEC = None
### 
### # Path to the controller directory
### CONTROLLER_HOME = DRIVER_HOME + '/../controller'
### 
### # Path to the controller configuration file
### CONTROLLER_CONFIG = os.path.join(CONTROLLER_HOME, 'config/{}_config.json'.format(DB_TYPE))


#==========================================================
#  LOGGING OPTIONS
#==========================================================

LOG_LEVEL = 'DEBUG'

# Path to log directory
LOG_DIR = os.path.join(DRIVER_HOME, 'log')

# Log files
DRIVER_LOG = os.path.join(LOG_DIR, 'driver.log')
OLTPBENCH_LOG = os.path.join(LOG_DIR, 'oltpbench.log')
TIMES_LOG = os.path.join(LOG_DIR, 'times.log')
### CONTROLLER_LOG = os.path.join(LOG_DIR, 'controller.log')

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

# Path to JSON dict of tunable knobs and their valid ranges
# (for non-numeric types set minval/maxval to None).
#
# Example:
#
# {
#     "global.random_page_cost": {
#         "minval": 1.0,
#         "maxval": 4.0,
#         "tunable": true
#     },
#     "global.wal_sync_method": {
#         "minval": null,
#         "maxval": null,
#         "tunable": true
#     },
#     "global.wal_writer_delay": {
#         "minval": 1,
#         "maxval": 10000,
#         "tunable": true
#     }
# }
KNOB_RANGES_FILE = os.environ.get('KNOB_RANGES_FILE', None)


#==========================================================
#  RUN KNOB CONFIGS OPTIONS
#==========================================================

# Directory containing the DBMS knob configs to run
KNOB_CONFIGDIR = os.environ.get('KNOB_CONFIGDIR', '/app/driver/knob_configs')

# Comma-separated string of config names (filename only without extension)
if 'KNOB_CONFIGS' in os.environ and os.environ['KNOB_CONFIGS']:
    KNOB_CONFIGS = os.environ['KNOB_CONFIGS'].split(',')
else:
    KNOB_CONFIGS = []
