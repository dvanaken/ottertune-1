import string

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import transaction

from website.models import Result, Session
from website.types import WorkloadStatusType


class Command(BaseCommand):
    help = 'Create a new user.'
    upload_code_valid_chars = string.ascii_letters + string.digits
    upload_code_max_length = 30 

    def add_arguments(self, parser):
        parser.add_argument(
            'upload_code',
            metavar='SRC_UPLOAD_CODE',
            help='Specifies the upload code of the existing session to be copied.')
        parser.add_argument(
            'new_upload_code',
            metavar='NEW_UPLOAD_CODE',
            type=self.upload_code_type,
            help='Specifies the upload code for the new session.')
        parser.add_argument(
            'results',
            metavar='RESULTS',
            type=int,
            nargs='*',
            help='Move and delete only these result IDs.')
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

    def handle(self, *args, **options):
        upload_code = options['upload_code']
        new_upload_code = options['new_upload_code']
        results = options['results']

        with transaction.atomic():
            call_command('copysession',
                         upload_code, 
                         *results,
                         '--quiet',
                         sessionname=options['sessionname'],
                         projectname=options['projectname'],
                         uploadcode=new_upload_code,
                         target_obj=options['target_obj'],
                         algorithm=options['algorithm'],
                         workload=options['workload'],
                         workload_status=options['workload_status'])
            orig_session = Session.objects.get(upload_code=upload_code)
            orig_sessionname='{}.{}'.format(orig_session.project.name, orig_session.name)
            #orig_nresults = Result.objects.filter(session=orig_session).count()
            new_session = Session.objects.get(upload_code=new_upload_code)
            new_sessionname='{}.{}'.format(new_session.project.name, new_session.name)
            new_nresults = Result.objects.filter(session=new_session).count()
            if results:
                call_command('deleteresults', *results)
            else:
                orig_session.delete()

        s = '{} results from '.format(len(results)) if results else ''
        self.stdout.write(self.style.SUCCESS(
            "Successfully moved {}session '{}' to '{}'\n  upload_code: {}\n  num_results: {}".format(
                s, orig_sessionname, new_sessionname, new_upload_code, new_nresults)))

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
