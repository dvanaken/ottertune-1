import csv
import datetime

from django.core.management.base import BaseCommand, CommandError

from website.models import Result, Session, SessionKnobManager
from website.utils import JSONUtil


class Command(BaseCommand):
    help = "Saves the session's the knob and metric data to csvs."

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help="The session's upload code.")
        parser.add_argument(
            '-o', '--out',
            metavar='FILE',
            default=None,
            help='Basename of the file to write the session knob/metric data to. '
                 'Default: [sessionname]_[knobs/metrics].csv')
        parser.add_argument(
            '--real-values',
            action='store_true',
            help='Output the real knob/metric settings (i.e., non-numeric). '
                 'Default: False')
        parser.add_argument(
            '--tunable-only',
            action='store_true',
            help='Dump tunable knobs only. Default: False')

    def handle(self, *args, **options):
        upload_code = options['uploadcode']
        try:
            session = Session.objects.get(upload_code=upload_code)
        except Session.DoesNotExist:
            raise CommandError(
                "ERROR: Session with upload code '{}' not exist.".format(upload_code))

        out = options['out'] or session.name
        real_vals = options['real_values']
        results = Result.objects.filter(session=session).order_by('creation_time')

        knob_names = None
        metric_names = None
        knob_csv = []
        metric_csv = []
        for res in results:
            ts = res.observation_end_time.strftime('%Y-%m-%d %H:%M:%S')
            if real_vals:
                knob_data = res.knob_data.knobs
                metric_data = res.metric_data.metrics
            else:
                knob_data = res.knob_data.data
                metric_data = res.metric_data.data

            knob_data = JSONUtil.loads(knob_data)
            metric_data = JSONUtil.loads(metric_data)
            if knob_names is None:
                if options['tunable_only']:
                    knob_names = SessionKnobManager.get_knob_min_max_tunability(
                        session, tunable_only=options['tunable_only']).keys()
                else:
                    knob_names = knob_data.keys()
                knob_names = sorted(knob_names)
                metric_names = sorted(metric_data.keys())

            knob_csv.append([ts] + [knob_data[k] for k in knob_names])
            metric_csv.append([ts] + [metric_data[m] for m in metric_names])

        knob_header = ['timestamp']
        for k in knob_names:
            if k.startswith('global.'):
                k = k.split('.', 1)[-1]
            knob_header.append(k)

        metric_header = ['timestamp']
        for m in metric_names:
            if m.startswith('global.'):
                m = m.split('.', 1)[-1]
            metric_header.append(m)

        filename = out + '__knobs.csv'
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(knob_header)
            writer.writerows(knob_csv)

        self.stdout.write(self.style.SUCCESS(
            "Successfully saved knob data to '{}'.".format(filename)))

        filename = out + '__metrics.csv'
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(metric_header)
            writer.writerows(metric_csv)

        self.stdout.write(self.style.SUCCESS(
            "Successfully saved metric data to '{}'.".format(filename)))
