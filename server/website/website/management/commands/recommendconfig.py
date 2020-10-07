from celery import chain, signature, uuid
from django.core.management.base import BaseCommand
from website.models import Result
from website.types import AlgorithmType
from website.utils import JSONUtil


class Command(BaseCommand):
    help = 'Recommend the next configuration for the given result.'

    def add_arguments(self, parser):
        parser.add_argument(
            'result_id',
            metavar='RESULT_ID',
            type=int,
            help='The ID of the result for generating the recommendation.')

    def handle(self, *args, **options):
        result_id = options['result_id']
        result = Result.objects.get(id=result_id)
        session = result.session

        if session.algorithm == AlgorithmType.GPR:
            subtask_list = [
                ('preprocessing', (result_id, session.algorithm)),
                ('aggregate_target_results', ()),
                ('map_workload', ()),
                ('configuration_recommendation', ()),
            ]
        elif session.algorithm == AlgorithmType.DDPG:
            subtask_list = [
                ('preprocessing', (result_id, session.algorithm)),
                ('train_ddpg', ()),
                ('configuration_recommendation_ddpg', ()),
            ]
        elif session.algorithm == AlgorithmType.DNN:
            subtask_list = [
                ('preprocessing', (result_id, session.algorithm)),
                ('aggregate_target_results', ()),
                ('map_workload', ()),
                ('configuration_recommendation', ()),
            ]

        subtasks = []
        for name, args in subtask_list:
            task_id = '{}-{}'.format(name, uuid())
            s = signature(name, args=args, options={'task_id': task_id})
            subtasks.append(s)

        response = chain(*subtasks).apply_async()
        result.task_ids = JSONUtil.dumps(response.as_tuple())
        result.save()

        self.stdout.write(self.style.SUCCESS(
            "Successfully started running celery recommendation task "
            "for result {}.\n".format(result_id)))
