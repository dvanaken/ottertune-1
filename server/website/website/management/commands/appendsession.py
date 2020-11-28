from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from website.models import BackupData, KnobData, MetricData, ModelData, Result, Session, Workload
from website.types import AlgorithmType, WorkloadStatusType
from website.utils import MediaUtil


class Command(BaseCommand):
    help = 'Create a new user.'

    def add_arguments(self, parser):
        parser.add_argument(
            'from_upload_code',
            metavar='FROM_UPLOAD_CODE',
            help='Specifies the upload code of the existing session copy from.')
        parser.add_argument(
            'to_upload_code',
            metavar='TO_UPLOAD_CODE',
            help='Specifies the upload code of the existing session copy to.')
        parser.add_argument(
            'results',
            metavar='RESULTS',
            type=int,
            nargs='*',
            help='Copy only these result IDs.')
        parser.add_argument(
            '-w',
            '--workload',
            metavar='WORKLOAD',
            default=None,
            help='Specifies the workload name for the new session results. '
                 'Default: same workload as the existing session results.')
        parser.add_argument(
            '-x',
            '--workload-status',
            metavar='STATUS',
            choices=WorkloadStatusType.TYPE_NAMES.values(),
            default=WorkloadStatusType.name(WorkloadStatusType.MODIFIED),
            help='Specifies the workload status. Default: MODIFIED.'.format(
                WorkloadStatusType.name(WorkloadStatusType.MODIFIED)))
        parser.add_argument(
            '-q',
            '--quiet',
            action='store_true',
            help="Quiet execution")

    def handle(self, *args, **options):
        from_upload_code = options['from_upload_code']
        to_upload_code = options['to_upload_code']
        copy_results = options['results']
        to_workload_name = options['workload']
        workload_status = WorkloadStatusType.type(options['workload_status'])
        quiet = options['quiet']

        with transaction.atomic():
            from_session = Session.objects.get(upload_code=from_upload_code)
            to_session = Session.objects.get(upload_code=to_upload_code)
            initial_num_results = Result.objects.filter(session=to_session).count()

            if copy_results:
                from_results = Result.objects.filter(
                    session=from_session, id__in=copy_results)

                if len(from_results) != len(copy_results):
                    missing = set(copy_results) - set([r.id for r in from_results])
                    raise CommandError(
                        "Result IDs for project '{}' session '{}' do not exist: {}".format(
                            from_session.project.name, from_session.name,
                            ', '.join(str(m) for m in missing)))
            else:
                from_results = Result.objects.filter(session=from_session)

            unique_workloads = [r.workload for r in from_results.distinct('workload')]
            model_datas = {m.result.id: m for m in ModelData.objects.filter(result__in=from_results)}
            #backup_datas = {b.result.id: b for b in BackupData.objects.filter(result__in=from_results)}
            from_results = from_results.order_by('id')

            if to_workload_name:
                res_workload, created = Workload.objects.get_or_create(
                    dbms=to_session.dbms, hardware=to_session.hardware,
                    name=to_workload_name, project=to_session.project,
                    defaults={'status': workload_status})
                if res_workload.status != workload_status:
                    res_workload.status = workload_status 
                    res_workload.save()
            else:
                res_workload = None

            workload_map = {}
            for workload in unique_workloads:
                wid = workload.id
                if res_workload:
                    workload_map[wid] = res_workload
                    v_workload = res_workload
                else:
                    v_workload, created = Workload.objects.get_or_create(
                        dbms=to_session.dbms, hardware=to_session.hardware,
                        name=workload.name, project=to_session.project,
                        defaults={'status': workload_status})
                    if v_workload.status != workload_status:
                        v_workload.status = workload_status 
                        v_workload.save()

                workload_map[wid] = v_workload

            for res in from_results:
                knob_data = res.knob_data
                knob_data = KnobData.objects.create_knob_data(
                    session=to_session, knobs=knob_data.knobs,
                    data=knob_data.data, dbms=to_session.dbms)

                metric_data = res.metric_data
                metric_data = MetricData.objects.create_metric_data(
                    session=to_session, metrics=metric_data.metrics,
                    data=metric_data.data, dbms=to_session.dbms)

                res_id_orig = res.id 
                res.id = None
                res.session = to_session
                res.workload = workload_map[res.workload.id]
                res.knob_data = knob_data
                res.metric_data = metric_data
                res.save()

                mdata = model_datas.pop(res_id_orig, None)
                if mdata:
                    mdata.id = None
                    mdata.result = res
                    mdata.save()

                #bdata = backup_datas.pop(res_id_orig, None)
                #if bdata:
                #    pass

                #backup_data = BackupData.objects.create(
                #    result=result, raw_knobs=files['knobs'],
                #    raw_initial_metrics=files['metrics_before'],
                #    raw_final_metrics=files['metrics_after'],
                #    raw_summary=files['summary'],
                #    knob_log=JSONUtil.dumps(knob_diffs, pprint=True),
                #    metric_log=JSONUtil.dumps(metric_diffs, pprint=True),
                #    other=JSONUtil.dumps(other_data))

        final_num_results = Result.objects.filter(session=to_session).count()
        if not quiet:
            self.stdout.write(self.style.SUCCESS((
                "Successfully appended session '{}.{}' to session '{}.{}'"
                "\n(initial_results: {}, final_results: {})").format(
                    from_session.project.name, from_session.name,
                    to_session.project.name, to_session.name,
                    initial_num_results, final_num_results)))
