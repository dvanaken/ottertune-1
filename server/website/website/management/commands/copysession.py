from django.core.management.base import BaseCommand
from django.db import transaction

from website.models import KnobData, MetricData, Project, Result, Session, SessionKnob, Workload
from website.types import AlgorithmType, WorkloadStatusType
from website.utils import MediaUtil


class Command(BaseCommand):
    help = 'Create a new user.'

    def add_arguments(self, parser):
        parser.add_argument(
            'upload_code',
            metavar='UPLOAD_CODE',
            help='Specifies the upload code of the existing session to be copied.')
        parser.add_argument(
            'new_sessionname',
            metavar='NEW_SESSIONNAME',
            help='Specifies what to name the newly copied session.')
        parser.add_argument(
            '--projectname',
            metavar='PROJECTNAME',
            default=None,
            help='Specifies which existing project the new session will belong to.')
        parser.add_argument(
            '--algorithm',
            metavar='ALGORITHM',
            default=None,
            help='Specifies which algorithm the new session will use.')

    def handle(self, *args, **options):
        upload_code = options['upload_code']
        new_sessionname = options['new_sessionname']
        projectname = options['projectname']
        algorithm = options['algorithm']

        session = Session.objects.get(upload_code=upload_code)
        session_knobs = SessionKnob.objects.filter(session=session)
        results = Result.objects.filter(session=session).order_by('creation_time')

        session.pk = None
        session.name = new_sessionname
        session.upload_code = MediaUtil.upload_code_generator()
        if algorithm:
            algorithm = algorithm.lower()
            if algorithm == 'gpr':
                session.algorithm = AlgorithmType.GPR
            elif algorithm == 'dnn':
                session.algorithm = AlgorithmType.DNN
            elif algorithm == 'ddpg':
                session.algorithm = AlgorithmType.DDPG

        if projectname and projectname != session.project.name:
            project = Project.objects.get(name=projectname)
            session.project = project

        with transaction.atomic():
            session.save()

            for knob in session_knobs:
                knob.pk = None
                knob.session = session
                knob.save()

            for res in results:
                knob_data = res.knob_data
                knob_data = KnobData.objects.create_knob_data(
                    session=session, knobs=knob_data.knobs,
                    data=knob_data.data, dbms=knob_data.dbms)

                metric_data = res.metric_data
                metric_data = MetricData.objects.create_metric_data(
                    session=session, metrics=metric_data.metrics,
                    data=metric_data.data, dbms=metric_data.dbms)

                workload, created = Workload.objects.get_or_create(
                    dbms=session.dbms, hardware=session.hardware,
                    name=res.workload.name, project=session.project)
                if res.workload.project != workload.project and workload.status != WorkloadStatusType.MODIFIED:
                    workload.status = WorkloadStatusType.MODIFIED
                    workload.save()

                res.pk = None
                res.session = session
                res.workload = workload
                res.knob_data = knob_data
                res.metric_data = metric_data
                res.save()

        self.stdout.write(self.style.SUCCESS(
            "Successfully created session '{}' under project '{}'\nupload_code: {}".format(
                session.name, session.project, session.upload_code)))
