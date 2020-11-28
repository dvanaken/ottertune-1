from django.core.management.base import BaseCommand
from django.db import transaction

from website.models import Session


class Command(BaseCommand):
    help = 'Delete existing results.'

    def add_arguments(self, parser):
        parser.add_argument(
            'upload_codes',
            metavar='UPLOAD_CODES',
            nargs='+',
            help='Upload codes of sessions to delete.')

    def handle(self, *args, **options):
        upload_codes = set(options['upload_codes'])
        ndelete = len(upload_codes)
        self.stdout.write("\nAttempting to delete {} sessions...".format(ndelete))
        deleted = []

        with transaction.atomic():
            sessions = Session.objects.filter(upload_code__in=upload_codes)
            if ndelete != len(sessions):
                missing = sorted(upload_codes - set([s.upload_code for s in sessions]))
                self.stdout.write(self.style.NOTICE(
                    "\nWARNING: {}/{} sessions do not exist: {}".format(
                        len(missing), ndelete,
                        ', '.join(str(m) for m in missing))))

            for sess in sessions:
                sess_uc = sess.upload_code
                sess.delete()
                deleted.append(sess_uc)

        if deleted:
            self.stdout.write(self.style.SUCCESS(
                "\nSuccessfully deleted {} sessions: {}.".format(
                    len(deleted), ', '.join(sorted(deleted)))))
