import json
from collections import OrderedDict

from django.core.management.base import BaseCommand, CommandError

from website.models import Result, Session
from website.views import _tuner_status_helper

#<class 'djcelery.models.TaskMeta'>, ['DoesNotExist', 'MultipleObjectsReturned', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__unicode__', '__weakref__', '_check_column_name_clashes', '_check_field_name_clashes', '_check_fields', '_check_id_field', '_check_index_together', '_check_local_fields', '_check_long_column_names', '_check_m2m_through_same_relationship', '_check_managers', '_check_model', '_check_model_name_db_lookup_clashes', '_check_ordering', '_check_swappable', '_check_unique_together', '_do_insert', '_do_update', '_get_FIELD_display', '_get_next_or_previous_by_FIELD', '_get_next_or_previous_in_order', '_get_pk_val', '_get_unique_checks', '_meta', '_perform_date_checks', '_perform_unique_checks', '_save_parents', '_save_table', '_set_pk_val', '_state', 'check', 'clean', 'clean_fields', 'date_done', 'date_error_message', 'delete', 'from_db', 'full_clean', 'get_deferred_fields', 'get_next_by_date_done', 'get_previous_by_date_done', 'get_status_display', 'hidden', 'id', 'meta', 'objects', 'pk', 'prepare_database_save', 'refresh_from_db', 'result', 'save', 'save_base', 'serializable_value', 'status', 'task_id', 'to_dict', 'traceback', 'unique_error_message', 'validate_unique']

class Command(BaseCommand):
    help = "Saves the session's result tasks."
    task_keys = ['task_id', 'status', 'result', 'traceback']

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

    def handle(self, *args, **options):
        upload_code = options['uploadcode']
        try:
            session = Session.objects.get(upload_code=upload_code)
        except Session.DoesNotExist:
            raise CommandError(
                "ERROR: Session with upload code '{}' not exist.".format(upload_code))

        results = Result.objects.filter(session=session).order_by('creation_time')
        if not results:
            self.stdout.write(self.style.NOTICE(
                "Session '{}' does not have any results!".format(session.name)))
            return
        out = options.get('out', None)
        if not out:
            out = '{}-{}__tasks.json'.format(session.pk, session.name)
        task_info = OrderedDict()
        for res in results:
            res_context = _tuner_status_helper(session.project.pk, session.pk, res.pk)
            rmap = OrderedDict()
            for task_type, task in res_context['tasks']:
                rmap[task_type] = {k: getattr(task, k) for k in self.task_keys}
            task_info[res.id] = rmap
         
        with open(out, 'w') as f:
            json.dump(task_info, f)
        self.stdout.write(self.style.SUCCESS(
            "Successfully saved task data to '{}'.".format(out)))
