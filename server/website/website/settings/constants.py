#
# OtterTune - constants.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from website.types import DBMSType

# These parameters are not specified for any session, so they can only be set here

# If this flag is set, we check if celery is running, and restart celery if it is not.
CHECK_CELERY = True

# address categorical knobs (enum, boolean)
ENABLE_DUMMY_ENCODER = False

# Whether to include the pruned metrics from the workload characterization subtask in
# the output (y) when ranking the knobs for a given workload in the knob identification
# subtask.

# When computing the ranked knobs in the knob identification subtask, the output (y) is
# the set of target objectives used to tune the given workload. If this flag is enabled
# then the pruned metrics from the workload characterization subtask are also included
# in the output. (See website/tasks/periodic_tasks.py)
KNOB_IDENT_USE_PRUNED_METRICS = True

# The background tasks only process workloads containing this minimum amount of results
MIN_WORKLOAD_RESULTS_COUNT = 5

# The views used for metrics pruning
VIEWS_FOR_PRUNING = {
    DBMSType.ORACLE: ['dba_hist_osstat', 'dba_hist_sysstat', 'dba_hist_system_event',
                      'dba_workload_replays', 'dba_hist_sys_time_model'],
}

# The views used for DDPG
# WARNING: modifying this parameter will cause all existing DDPG sessions broken
VIEWS_FOR_DDPG = {
    DBMSType.ORACLE: ['dba_hist_sys_time_model'],
}

ACTIVE_METRICS_ONLY = True

PRUNED_METRICS_MIN_CLUSTERS = 4

PRUNED_METRICS_MAX_CLUSTERS = 20

BG_TASKS_PROCESS_COMBINED_DATA = True

CONTEXT_METRICS = {
    DBMSType.MYSQL: {
        'context5': ["latency_99", "global.innodb_data_fsyncs", "global.innodb_buffer_pool_bytes_data", "global.innodb_log_write_requests", "global.innodb_log_waits", "global.innodb_pages_created"],

        'context8': ["latency_99", "global.innodb_buffer_pool_wait_free", "global.innodb_buffer_pool_reads", "global.innodb_buffer_pool_read_requests", "global.innodb_row_lock_time_avg", "global.innodb_data_read", "global.innodb_data_written", "global.innodb_log_waits"],

        'context16': ["latency_99", "global.innodb_buffer_pool_write_requests", "global.threads_cached", "global.innodb_data_read", "global.innodb_dblwr_pages_written", "global.queries", "global.innodb_buffer_pool_reads", "global.innodb_buffer_pool_read_ahead", "global.innodb_data_written", "latency_95", "global.innodb_buffer_pool_pages_free", "global.innodb_dblwr_writes", "global.innodb_buffer_pool_pages_total", "global.innodb_buffer_pool_bytes_dirty", "global.innodb_log_writes", "global.uptime"],
    },
}

ALL_METRICS = ['global.bytes_received', 'global.bytes_sent', 'global.com_commit', 'global.com_delete', 'global.com_insert', 'global.com_rollback', 'global.com_select', 'global.com_update', 'global.created_tmp_files', 'global.created_tmp_tables', 'global.handler_commit', 'global.handler_delete', 'global.handler_external_lock', 'global.handler_read_key', 'global.handler_read_next', 'global.handler_read_rnd_next', 'global.handler_rollback', 'global.handler_update', 'global.handler_write', 'global.innodb_buffer_pool_bytes_data', 'global.innodb_buffer_pool_bytes_dirty', 'global.innodb_buffer_pool_pages_data', 'global.innodb_buffer_pool_pages_dirty', 'global.innodb_buffer_pool_pages_flushed', 'global.innodb_buffer_pool_pages_free', 'global.innodb_buffer_pool_pages_misc', 'global.innodb_buffer_pool_pages_total', 'global.innodb_buffer_pool_read_ahead', 'global.innodb_buffer_pool_read_ahead_evicted', 'global.innodb_buffer_pool_read_requests', 'global.innodb_buffer_pool_reads', 'global.innodb_buffer_pool_wait_free', 'global.innodb_buffer_pool_write_requests', 'global.innodb_data_fsyncs', 'global.innodb_data_pending_fsyncs', 'global.innodb_data_read', 'global.innodb_data_reads', 'global.innodb_data_writes', 'global.innodb_data_written', 'global.innodb_dblwr_pages_written', 'global.innodb_dblwr_writes', 'global.innodb_log_waits', 'global.innodb_log_write_requests', 'global.innodb_log_writes', 'global.innodb_os_log_fsyncs', 'global.innodb_os_log_written', 'global.innodb_pages_created', 'global.innodb_pages_read', 'global.innodb_pages_written', 'global.innodb_row_lock_time', 'global.innodb_row_lock_time_avg', 'global.innodb_row_lock_time_max', 'global.innodb_row_lock_waits', 'global.innodb_rows_deleted', 'global.innodb_rows_inserted', 'global.innodb_rows_read', 'global.innodb_rows_updated', 'global.open_tables', 'global.opened_table_definitions', 'global.opened_tables', 'global.queries', 'global.questions', 'global.select_range', 'global.select_scan', 'global.table_open_cache_hits', 'global.table_open_cache_misses', 'global.threads_cached', 'global.uptime', 'global.uptime_since_flush_status', 'innodb_metrics.trx_active_transactions', 'innodb_metrics.trx_commits_insert_update', 'innodb_metrics.trx_nl_ro_commits', 'innodb_metrics.trx_ro_commits', 'innodb_metrics.trx_rollbacks', 'innodb_metrics.trx_rollbacks_savepoint', 'innodb_metrics.trx_rseg_current_size', 'innodb_metrics.trx_rseg_history_len', 'innodb_metrics.trx_rw_commits', 'innodb_metrics.trx_undo_slots_cached', 'innodb_metrics.trx_undo_slots_used', 'latency_99', 'throughput']

TARGET_METRICS = ['throughput', 'latency_99']

RANDOM_METRICS1 = TARGET_METRICS + ['global.innodb_rows_read', 'global.open_tables', 'innodb_metrics.trx_ro_commits', 'global.innodb_rows_read', 'innodb_metrics.trx_rollbacks'] 

RANDOM_METRICS2 = TARGET_METRICS + ['global.innodb_rows_inserted', 'global.innodb_rows_read', 'innodb_metrics.trx_undo_slots_cached', 'global.innodb_buffer_pool_read_ahead_evicted', 'global.created_tmp_files']

RANDOM_METRICS3 = TARGET_METRICS + ['innodb_metrics.trx_rollbacks_savepoint', 'global.innodb_buffer_pool_wait_free', 'global.opened_tables', 'global.innodb_buffer_pool_pages_data', 'global.uptime']

WIKIPEDIA_METRICS = ["global.innodb_log_waits", "global.innodb_log_write_requests", "global.bytes_received", "global.innodb_pages_written", "global.innodb_buffer_pool_reads", "global.innodb_rows_read", "global.innodb_row_lock_time_avg", "throughput", "udm.latency_95", "global.innodb_buffer_pool_bytes_dirty", "global.innodb_data_writes", "global.com_update"] 

OVERRIDE_PRUNED_METRICS = {
    DBMSType.MYSQL: ["throughput"] + CONTEXT_METRICS[DBMSType.MYSQL]['context5'], 
    #DBMSType.MYSQL: WIKIPEDIA_METRICS,
    #DBMSType.MYSQL: ALL_METRICS,
    #DBMSType.MYSQL: TARGET_METRICS, 
    #DBMSType.MYSQL: RANDOM_METRICS1, 
    #DBMSType.MYSQL: RANDOM_METRICS2, 
    #DBMSType.MYSQL: RANDOM_METRICS3, 
}
