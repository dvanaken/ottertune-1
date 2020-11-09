from django.core.management.base import BaseCommand
from django.db import transaction

from website.models import KnobData, MetricData, Project, Result, Session, SessionKnob, Workload
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

    def handle(self, *args, **options):
        from_upload_code = options['from_upload_code']
        to_upload_code = options['to_upload_code']

        from_session = Session.objects.get(upload_code=from_upload_code)
        to_session = Session.objects.get(upload_code=to_upload_code)
        from_results = Result.objects.filter(session=from_session).order_by('creation_time')
        to_workload = Result.objects.filter(session=to_session).first().workload

        with transaction.atomic():
            for res in from_results:
                knob_data = res.knob_data
                knob_data = KnobData.objects.create_knob_data(
                    session=to_session, knobs=knob_data.knobs,
                    data=knob_data.data, dbms=knob_data.dbms)

                metric_data = res.metric_data
                metric_data = MetricData.objects.create_metric_data(
                    session=to_session, metrics=metric_data.metrics,
                    data=metric_data.data, dbms=metric_data.dbms)

                res.pk = None
                res.session = to_session
                res.workload = to_workload
                res.knob_data = knob_data
                res.metric_data = metric_data
                res.save()

        self.stdout.write(self.style.SUCCESS(
            "Successfully appended session '{}.{}' to session '{}.{}'".format(
                from_session.project.name, from_session.name,
                to_session.project.name, to_session.name)))
