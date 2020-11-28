import glob
import json
import os
import time
from collections import OrderedDict

from django.core.management.base import BaseCommand

from website.models import Session, Result
from website.utils import JSONUtil


class Command(BaseCommand):
    help = 'Find session results matching knob/metric upload data.'

    def add_arguments(self, parser):
        parser.add_argument(
            'upload_code',
            metavar='UPLOAD_CODE',
            help='Session upload code.')
        parser.add_argument(
            'directory',
            metavar='DIRECTORY',
            help='Path to the directory containing the upload data.')
        parser.add_argument(
            '-p',
            '--prefix',
            metavar='PREFIX',
            default='',
            help='Prefix of upload data files to match.')

    def handle(self, *args, **options):
        start_time = time.time()
        upload_code = options['upload_code']
        directory = options['directory']
        prefix = options['prefix']

        udm_files = sorted(glob.glob(os.path.join(directory, prefix + '*user_defined_metrics.json')))
        if not udm_files:
            self.stdout.write(self.style.NOTICE("\nNo upload files found!"))
            return

        session = Session.objects.get(upload_code=upload_code)
        results = Result.objects.filter(session=session)
        if not results:
            self.stdout.write(self.style.NOTICE(
                "\nNo result files found for session '{}'!".format(session.name)))
            return

        result_matches = OrderedDict()
        baselens = []

        self.stdout.write("\nFinding matches for {} upload files and {} web results...".format(
            len(udm_files), len(results)))

        for udm_file in udm_files:
            basepath = udm_file.replace('user_defined_metrics.json', '')
            basename = os.path.basename(basepath)
            baselens.append(len(basename))

            udata = {}
            with open(udm_file, 'r') as f:
                udms = json.load(f)

            for udm_name, entry in udms.items():
                udata['udm.' + udm_name] = int(entry['value'])

            matches = []

            for res in results:
                met_data = JSONUtil.loads(res.metric_data.metrics)
                match = True
                for k, v in udata.items():
                    if met_data[k] != v:
                        match = False
                        break
                if match:
                    matches.append(str(res.pk))

            result_matches[basename] = matches

        elapsed = time.time() - start_time

        fmt = ('    {: <' + str(max(baselens) + 3) + '} {: >6}\n').format
        out = "\nMatches for session '{}' ({:.1f} sec)\n\n".format(session.name, elapsed)
        for basename, matches in result_matches.items():
            out += fmt(basename + '*', ' '.join(matches) or '-')
        self.stdout.write(self.style.SUCCESS(out))
