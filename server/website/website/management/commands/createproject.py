#
# OtterTune - createuser.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils.timezone import now

from website.models import Project 


class Command(BaseCommand):
    help = 'Create a new project.'

    def add_arguments(self, parser):
        parser.add_argument(
            'name',
            metavar='NAME',
            help='Specifies the name of the project.')
        parser.add_argument(
            '-u',
            '--username',
            metavar='USERNAME',
            default='admin',
            help='Specifies the login for the user. Default: admin')
        parser.add_argument(
            '-d',
            '--description',
            metavar='DESCRIPTION',
            default='',
            help='Description for the project. Default: admin')

    def handle(self, *args, **options):
        name = options['name']
        user = User.objects.get(username=options['username'])
        desc = options['description']
        ts = now()

        project = Project(name=name, user=user, description=desc, creation_time=ts, last_update=ts)
        project.save()
        self.stdout.write(self.style.SUCCESS("Successfully created project '{}'.".format(name)))
