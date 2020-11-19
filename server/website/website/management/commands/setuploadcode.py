#
# OtterTune - setuploadcode.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.core.management.base import BaseCommand, CommandError

from website.models import Session
from website.utils import MediaUtil


class Command(BaseCommand):
    help = 'Set the upload code for a session.'

    def add_arguments(self, parser):
        parser.add_argument(
            'cur_upload_code',
            metavar='UPLOAD_CODE',
            help='The current upload code.')
        #parser.add_argument(
        #    'username',
        #    metavar='USERNAME',
        #    help='Specifies the username of the session owner.')
        #parser.add_argument(
        #    'projectname',
        #    metavar='PROJECTNAME',
        #    help='Specifies the name of the project that the session belongs to.')
        #parser.add_argument(
        #    'sessionname',
        #    metavar='SESSIONNAME',
        #    help='Specifies the name of the session.')
        parser.add_argument(
            '-u',
            '--uploadcode',
            metavar='UPLOADCODE',
            default=None,
            help='Specifies the value to set the upload code to.')

    def handle(self, *args, **options):
        try:
            session = Session.objects.get(upload_code=options['cur_upload_code'])
        except Session.DoesNotExist:
            raise CommandError(
                "No session with upload code '{}' exists.".format(options['cur_upload_code']))
        #username = options['username']
        #projectname = options['projectname']
        #sessionname = options['sessionname']
        #session = Session.objects.filter(user__username=username,
        #                                 project__name=projectname,
        #                                 name=sessionname).first()
        upload_code = options['uploadcode'] or MediaUtil.upload_code_generator()
        session.upload_code = upload_code
        session.save()
        self.stdout.write(self.style.SUCCESS(upload_code))
