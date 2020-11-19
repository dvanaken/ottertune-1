from django.core.management.base import BaseCommand
from django.db import transaction

from website.models import Result


class Command(BaseCommand):
    help = 'Delete existing results.'

    def add_arguments(self, parser):
        parser.add_argument(
            'result_ids',
            metavar='RESULT_IDS',
            type=int,
            nargs='+',
            help='Result IDs to delete.')

    def handle(self, *args, **options):
        result_ids = set(options['result_ids'])
        self.stdout.write("\nAttempting to delete {} results...".format(
            len(result_ids)))
        deleted = []

        with transaction.atomic():
            results = Result.objects.filter(id__in=result_ids)
            if len(results) != len(result_ids):
                missing = sorted(result_ids - set([r.pk for r in results]))
                self.stdout.write(self.style.NOTICE(
                    "\nWARNING: {}/{} results do not exist: {}".format(
                        len(missing), len(result_ids),
                        ', '.join(str(m) for m in missing))))

            for res in results:
                res_id = res.pk
                res.delete()
                deleted.append(str(res_id))

        if deleted:
            self.stdout.write(self.style.SUCCESS(
                "\nSuccessfully deleted {} results: {}.".format(
                    len(deleted), ', '.join(sorted(deleted)))))
