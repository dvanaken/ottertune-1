import string

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from website.models import (KnobData, MetricData, ModelData, Project, Result, Session,
                            SessionKnob, Workload)
from website.types import AlgorithmType, WorkloadStatusType
from website.utils import MediaUtil


class Command(BaseCommand):
    help = 'Create a new user.'
    upload_code_valid_chars = string.ascii_letters + string.digits
    upload_code_max_length = 30 

    def add_arguments(self, parser):
        parser.add_argument(
            'upload_code',
            metavar='UPLOAD_CODE',
            help='Specifies the upload code of the existing session to be copied.')
        parser.add_argument(
            'results',
            metavar='RESULTS',
            type=int,
            nargs='*',
            help='Copy only these result IDs.')
        parser.add_argument(
            '-s',
            '--sessionname',
            metavar='SESSIONNAME',
            help='Specifies what to name the NEW session. '
                 'Default: same name as OLD session.')
        parser.add_argument(
            '-p',
            '--projectname',
            metavar='PROJECTNAME',
            default=None,
            help='Specifies which existing project the new session will belong to. '
                 'Default: same as project name of original session.')
        parser.add_argument(
            '-u',
            '--uploadcode',
            metavar='UPLOAD_CODE',
            default=None,
            type=self.upload_code_type,
            help='Specifies the upload code for the new session. '
                 'Default: SESSIONNAME with invalid chars removed, or '
                 '[unique_id] + SESSIONNAME if it already exists.')
        parser.add_argument(
            '-t',
            '--target-obj',
            metavar='TARGET_OBJ',
            default=None,
            choices=('throughput', 'latency_99', 'latency_95'),
            help='Specifies the target objective for the new session. '
                 'Default: same as original session.')
        parser.add_argument(
            '-a',
            '--algorithm',
            metavar='ALGORITHM',
            default=None,
            choices=('gpr', 'dnn', 'ddpg'),
            help='Specifies which algorithm the new session will use. '
                 'Default: same as original session.')
        parser.add_argument(
            '-w',
            '--workload',
            metavar='WORKLOAD',
            default=None,
            help='Specifies the workload name for the new session results. '
                 'Default: same workload as original results.')
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
        upload_code = options['upload_code']
        new_sessionname = options['sessionname']
        projectname = options['projectname']
        new_upload_code = options['uploadcode']
        target_obj = options['target_obj']
        algorithm = options['algorithm']
        quiet = options['quiet']

        with transaction.atomic():
            session = Session.objects.get(upload_code=upload_code)
            session_knobs = SessionKnob.objects.filter(session=session)

            session.pk = None
            if new_sessionname:
                session.name = new_sessionname

            if not new_upload_code:
                new_upload_code = self.get_upload_code(session.name)
            session.upload_code = new_upload_code

            if target_obj:
                session.target_objective = target_obj

            if algorithm:
                session.algorithm = AlgorithmType.short_type(algorithm)

            if projectname and projectname != session.project.name:
                project = Project.objects.get(name=projectname)
                session.project = project

            session.save()

            for knob in session_knobs:
                knob.pk = None
                knob.session = session
                knob.save()

            call_command('appendsession', upload_code, new_upload_code, *options['results'],
                         '--quiet', workload=options['workload'],
                         workload_status=options['workload_status'])

        if not quiet:
            self.stdout.write(self.style.SUCCESS(
                "Successfully created session '{}.{}'\n  upload_code: {}\n  num_results: {}".format(
                    session.project, session.name, session.upload_code,
                    Result.objects.filter(session=session).count())))

    def get_upload_code(self, sessionname):
        max_len = self.upload_code_max_length
        upload_code = ''.join(c for c in sessionname if c in self.upload_code_valid_chars)[:max_len]
        i = 0
        while True:
            exists = Session.objects.filter(upload_code=upload_code).exists()
            if not exists:
                break
            upload_code = '{}{}'.format(i, upload_code)[:max_len] 
            i += 1

        return upload_code

    def upload_code_type(self, s):
        # Note: this method is only called if the user explicitly sets the --uploadcode option so
        # an unset upload code cannot be generated here since the method never gets called.
        if len(s) > 30:
            raise ValueError("Upload code is too long: {} (expected <= 30 characters)".format(len(s)))
        for c in s:
            valid_chars = self.upload_code_valid_chars
            if c not in valid_chars:
                raise ValueError("Upload code contains invalid character: {} (valid chars: {})".format(
                    c, valid_chars))
        return s
