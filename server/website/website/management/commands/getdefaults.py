import json
from collections import OrderedDict

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from website.models import BackupData, Result

DEFAULTS = OrderedDict([
    ("global.innodb_adaptive_flushing", 1),
    ("global.innodb_adaptive_flushing_lwm", 10),
    ("global.innodb_adaptive_hash_index", 1),
    ("global.innodb_adaptive_max_sleep_delay", 150000),
    ("global.innodb_autoextend_increment", 64),
    ("global.innodb_buffer_pool_dump_at_shutdown", 1),
    ("global.innodb_buffer_pool_instances", 1),
    ("global.innodb_buffer_pool_load_at_startup", 1),
    ("global.innodb_buffer_pool_size", 134217728),
    ("global.innodb_change_buffer_max_size", 25),
    ("global.innodb_concurrency_tickets", 5000),
    ("global.innodb_disable_sort_file_cache", 0),
    ("global.innodb_file_per_table", 1),
    ("global.innodb_flush_log_at_timeout", 1),
    ("global.innodb_flush_method", 0),
    ("global.innodb_flushing_avg_loops", 30),
    ("global.innodb_io_capacity", 200),
    ("global.innodb_lock_wait_timeout", 50),
    ("global.innodb_log_buffer_size", 16777216),
    ("global.innodb_log_file_size", 50331648),
    ("global.innodb_log_files_in_group", 2),
    ("global.innodb_lru_scan_depth", 1024),
    ("global.innodb_max_dirty_pages_pct", 90.0),
    ("global.innodb_max_dirty_pages_pct_lwm", 10),
    ("global.innodb_max_purge_lag", 0),
    ("global.innodb_old_blocks_pct", 37),
    ("global.innodb_old_blocks_time", 1000),
    ("global.innodb_purge_batch_size", 300),
    ("global.innodb_purge_threads", 4),
    ("global.innodb_read_ahead_threshold", 56),
    ("global.innodb_read_io_threads", 4),
    ("global.innodb_rollback_segments", 128),
    ("global.innodb_spin_wait_delay", 6),
    ("global.innodb_sync_array_size", 1),
    ("global.innodb_sync_spin_loops", 30),
    ("global.innodb_thread_concurrency", 0),
    ("global.innodb_thread_sleep_delay", 10000),
    ("global.innodb_write_io_threads", 4),
    ("global.join_buffer_size", 262144),
    ("global.lock_wait_timeout", 31536000),
    ("global.max_write_lock_count", 18446744073709551615),
    ("global.query_alloc_block_size", 8192),
    ("global.query_prealloc_size", 8192),
    ("global.read_rnd_buffer_size", 262144),
    ("global.skip_name_resolve", 1),
    ("global.sort_buffer_size", 262144),
    ("global.table_open_cache", 4000),
    ("global.table_open_cache_instances", 16),
    ("global.thread_cache_size", 9),
    ("global.tmp_table_size", 16777216),
    ("global.transaction_prealloc_size", 4096),
])

DEFAULTS2 = dict(DEFAULTS)


class Command(BaseCommand):
    help = 'Get default results.'

    def add_arguments(self, parser):
        parser.add_argument(
            'workloads',
            metavar='WORKLOADS',
            type=int,
            nargs='*',
            help='Check only these workloads.')

    def handle(self, *args, **options):
        workloads = options['workloads']
        res_obs_times = {}
        workload_defaults = {}
        default_results = []
        dups = 0
        if workloads:
            results = Result.objects.filter(workload__name__in=workloads).order_by('id')
        else:
            results = Result.objects.all().order_by('id')
        num_results = len(results)
        self.stdout.write(
            "Searching for default configs in {} results\n".format(num_results))
        for i, res in enumerate(results):
            res_knobs = json.loads(res.knob_data.data)
            res_knobs = {k: v for k, v in res_knobs.items() if k in DEFAULTS2}
            if res_knobs == DEFAULTS2:
                st = str(res.observation_start_time)
                sess = res.session
                res_name = '{}.{}#{}'.format(sess.project.name, sess.name, res.id)
                if st in res_obs_times:
                    dups += 1
                    #rname = res_obs_times[st]
                    #self.stdout.write("DUP({}): {} == {}".format(st, rname, res_name))
                else:
                    res_obs_times[st] = res_name
                    wkld_name = res.workload.name
                    if wkld_name not in workload_defaults:
                        workload_defaults[wkld_name] = []
                    workload_defaults[wkld_name].append(res.id)
                    default_results.append(res)
                    #self.stdout.write('DEFAULT: {}'.format(res_name))

        self.stdout.write('\n# DUPS: {}'.format(dups))
        self.stdout.write('')
        total = 0
        for wkld_name, res_ids in workload_defaults.items():
            num_rids = len(res_ids)
            total += num_rids
            k = '{} ({}):'.format(wkld_name.upper(), num_rids)
            self.stdout.write("{: <15} {}  ({})".format(
                wkld_name.upper(),
                ', '.join(str(r) for r in sorted(res_ids)), num_rids))
        self.stdout.write('\nTotal: {}\n'.format(total))

        bdatas = BackupData.objects.filter(result__in=default_results)
        self.stdout.write('{}/{} results have backup data\n'.format(
            bdatas.count(), total))

