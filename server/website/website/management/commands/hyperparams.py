from collections import OrderedDict

from django.core.management.base import BaseCommand, CommandError

from analysis.gpr import gpr_models
from website.models import Session
from website.utils import JSONUtil


class Command(BaseCommand):
    help = 'Set the hyperparameters for a session'

    base_options = ('verbosity', 'settings', 'pythonpath', 'traceback', 'no_color',
                    'uploadcode', 'print', 'reset')

    tbl_fmt = '{k: <25}  {v1: >{vlen}}  {v2: >{vlen}}\n'.format

    def add_arguments(self, parser):
        parser.add_argument(
            'uploadcode',
            metavar='UPLOADCODE',
            help='The upload code of the session')
        parser.add_argument(
            '--print',
            action='store_true',
            help='Print the current hyperparameters for the session')
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Reset hyperparameters to their default values')
        # ************************
        # DDPG HYPERPARAMETERS
        # ************************
        parser.add_argument(
            '--ddpg-actor-hidden-sizes',
            nargs=3,
            required=False,
            default=None,  # [128, 128, 64]
            type=int,
            metavar='INT',
            help='Default: 128 128 64')
        parser.add_argument(
            '--ddpg-actor-learning-rate',
            required=False,
            default=None,  # 0.02
            type=float,
            metavar='FLOAT',
            help='Default: 0.02')
        parser.add_argument(
            '--ddpg-critic-hidden-sizes',
            nargs=3,
            required=False,
            default=None,  # [64, 128, 64]
            type=int,
            metavar='INT',
            help='Default: 64 128 64')
        parser.add_argument(
            '--ddpg-critic-learning-rate',
            required=False,
            default=None,  # 0.001
            type=float,
            metavar='FLOAT',
            help='Default: 0.001')
        parser.add_argument(
            '--ddpg-batch-size',
            required=False,
            default=None,  # 32
            type=int,
            metavar='INT',
            help='Default: 32')
        parser.add_argument(
            '--ddpg-gamma',
            required=False,
            default=None,  # 0.0
            type=float,
            metavar='FLOAT',
            help='Default: 0.0')
        parser.add_argument(
            '--ddpg-simple-reward',
            required=False,
            default=None,  # True
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: True')
        parser.add_argument(
            '--ddpg-update-epochs',
            required=False,
            default=None,  # 30
            type=int,
            metavar='INT',
            help='Default: 30')
        parser.add_argument(
            '--ddpg-use-default',
            required=False,
            default=None,  # False
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: False')
        # ************************
        # DNN HYPERPARAMETERS
        # ************************
        parser.add_argument(
            '--dnn-context',
            required=False,
            default=None,
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: False')
        parser.add_argument(
            '--dnn-debug',
            required=False,
            default=None,  # True 
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: True')
        parser.add_argument(
            '--dnn-debug-interval',
            required=False,
            default=None,  # 100
            type=int,
            metavar='INT',
            help='Default: 100')
        parser.add_argument(
            '--dnn-explore',
            required=False,
            default=None,  # False
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: False')
        parser.add_argument(
            '--dnn-explore-iter',
            required=False,
            default=None,  # 500
            type=int,
            metavar='INT',
            help='Default: 500')
        parser.add_argument(
            '--dnn-gd-iter',
            required=False,
            default=None,  # 100
            type=int,
            metavar='INT',
            help='Default: 100')
        parser.add_argument(
            '--dnn-noise-scale-begin',
            required=False,
            default=None,  # 0.1
            type=float,
            metavar='FLOAT',
            help='Default: 0.1')
        parser.add_argument(
            '--dnn-noise-scale-end',
            required=False,
            default=None,  # 0.0
            type=float,
            metavar='FLOAT',
            help='Default: 0.0')
        parser.add_argument(
            '--dnn-train-iter',
            required=False,
            default=None,  # 100
            type=int,
            metavar='INT',
            help='Default: 100')
        # ************************
        # GPR HYPERPARAMETERS
        # ************************
        parser.add_argument(
            '--gpr-context',
            required=False,
            default=None,
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: False')
        parser.add_argument(
            '--gpr-debug',
            required=False,
            default=None,  # True 
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: True')
        parser.add_argument(
            '--gpr-eps',
            required=False,
            default=None,  # 0.001
            type=float,
            metavar='FLOAT',
            help='The small bias to add to the starting points. Default: 0.001')
        parser.add_argument(
            '--gpr-hp-learning-rate',
            required=False,
            default=None,  # 0.001
            type=float,
            metavar='FLOAT',
            help='Default: 0.001')
        parser.add_argument(
            '--gpr-hp-max-iter',
            required=False,
            default=None,  # 5000
            type=int,
            metavar='INT',
            help='Default: 5000')
        parser.add_argument(
            '--gpr-learning-rate',
            required=False,
            default=None,  # 0.001
            type=float,
            metavar='FLOAT',
            help='Default: 0.001')
        parser.add_argument(
            '--gpr-max-iter',
            required=False,
            default=None,  # 500
            type=int,
            metavar='INT',
            help='Default: 500')
        parser.add_argument(
            '--gpr-model-name',
            required=False,
            choices=gpr_models._MODEL_MAP.keys(),
            default=None,  # BasicGP
            metavar='NAME',
            help='Default: BasicGP')
        parser.add_argument(
            '--gpr-optimize-model',
            required=False,
            default=None,
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Default: False')
        parser.add_argument(
            '--gpr-ucb-beta',
            required=False,
            default=None,  # get_beta_td
            type=self.parse_gpr_ucb_beta,
            metavar='VALUE',
            help='Default: get_beta_td')
        parser.add_argument(
            '--gpr-ucb-scale',
            required=False,
            default=None,  # 5.0
            type=float,
            metavar='FLOAT',
            help='Default: 5.0')
        # ************************
        # COMMON HYPERPARAMETERS
        # ************************
        parser.add_argument(
            '--context-metrics',
            required=False,
            default=None,
            metavar='VALUE',
            help='Default: None')
        parser.add_argument(
            '--num-samples',
            required=False,
            default=None,  # 100
            type=int,
            metavar='INT',
            help="Number of grid samples to use for the GPR/DNN starting points. Default: 100")
        parser.add_argument(
            '--top-num-config',
            required=False,
            default=None,  # 10
            type=int,
            metavar='INT',
            help="Number of top-performing samples to use for the GPR/DNN starting points. Default: 10")
        parser.add_argument(
            '--checkpoint',
            required=False,
            default=None,
            type=self.parse_bool,
            choices=(True, False),
            metavar='BOOL',
            help='Checkpoint the ML model. Default: False')
        parser.add_argument(
            '--checkpoint-interval',
            required=False,
            default=None,  # 5
            type=int,
            metavar='INT',
            help="Model checkpoint frequency. Default: 5")
        parser.add_argument(
            '--checkpoint-start',
            required=False,
            default=None,  # 15
            type=int,
            metavar='INT',
            help="Model checkpoint start iteration. Default: 15")
        # ************************
        # UNUSED HYPERPARAMETERS
        # ************************
        # "GPR_BATCH_SIZE": 3000,
        # "GPR_EPSILON": 1e-06,
        # "GPR_LENGTH_SCALE": 2.0,
        # "GPR_MAGNITUDE": 1.0,
        # "GPR_MAX_TRAIN_SIZE": 7000,
        # "GPR_MU_MULTIPLIER": 1.0,
        # "GPR_RIDGE": 1.0,
        # "GPR_SIGMA_MULTIPLIER": 1.0,
        # "GPR_USE_GPFLOW": true,
        # "IMPORTANT_KNOB_NUMBER": 10000,
        # "TF_NUM_THREADS": 4,
        # # Dummy encoding
        # "FLIP_PROB_DECAY": 0.5,
        # "INIT_FLIP_PROB": 0.3,
        self.hyperparam_options = [k for k in sorted(vars(parser.parse_args(['x'])).keys()) if k not in self.base_options]

    def handle(self, *args, **options):
        upload_code = options.pop('uploadcode')
        session = Session.objects.get(upload_code=upload_code)
        print_ = options.pop('print')
        reset = options.pop('reset')
        default_hyperparams = JSONUtil.loads(Session._meta.get_field('hyperparameters').get_default())
        sess_hyperparams = JSONUtil.loads(session.hyperparameters)

        if print_ is True:
            t = self.tbl_fmt(k='Name', v1='Value', v2='Default', vlen=15)
            s = "{0}\n{1}{0}\n".format('*' * len(t), t)
            for name, val in sess_hyperparams.items():
                if name.lower() not in self.hyperparam_options:
                    continue
                default = default_hyperparams[name]
                v2 = '-' if default == val else str(default)
                s += self.tbl_fmt(k=name, v1=str(val), v2=v2, vlen=15)
            self.stdout.write(self.style.SUCCESS(s))

        else:
            if reset is True:
                new_hyperparams = default_hyperparams
                action = 'reset'
            else:
                new_hyperparams = OrderedDict([
                    (k.upper(), v) for k, v in options.items() if k in self.hyperparam_options and v is not None])
                action = 'updated'

            updated = []
            vlen = 10
            for name, val in new_hyperparams.items():
                cur_val = sess_hyperparams[name]
                if val != cur_val:
                    updated.append((name, str(cur_val), str(val)))
                    cur_len = max(len(str(cur_val)), len(str(val)))
                    if cur_len > vlen:
                        vlen = cur_len
                    sess_hyperparams[name] = val

            if updated:
                session.hyperparameters = JSONUtil.dumps(sess_hyperparams, pprint=True)
                session.save()
                t = self.tbl_fmt(k='Name', v1='Initial', v2='Final', vlen=vlen)
                s = "{0}\n{1}{0}\n".format('*' * len(t), t)
                for k, v1, v2 in updated:
                    s += self.tbl_fmt(k=k, v1=v1, v2=v2, vlen=vlen)
                self.stdout.write(self.style.SUCCESS(s))
            else:
                self.stdout.write(self.style.NOTICE("No hyperparameters updated"))

    @staticmethod
    def parse_bool(s):
        return str(s).lower() == 'true'

    @staticmethod
    def parse_gpr_ucb_beta(s):
        if s not in ('get_beta_const', 'get_beta_t', 'get_beta_td'):
            s = float(s)
        return s
