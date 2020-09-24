from django.core.management.base import BaseCommand
from website.tasks import run_background_tasks


class Command(BaseCommand):
    help = 'Run celery background tasks.'

    def handle(self, *args, **options):
        run_background_tasks.apply_async()
        self.stdout.write(self.style.SUCCESS(
            "Successfully started running celery background tasks."))
