#
# OtterTune - target_objective.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#

from ..base.target_objective import (BaseThroughput, BaseUserDefinedTarget,
                                     LESS_IS_BETTER, MORE_IS_BETTER)  # pylint: disable=relative-beyond-top-level
from website.types import DBMSType

target_objective_list = tuple((DBMSType.POSTGRES, target_obj) for target_obj in [  # pylint: disable=invalid-name
    BaseThroughput(transactions_counter='pg_stat_database.xact_commit'),
    BaseUserDefinedTarget(target_name='latency_99', improvement=LESS_IS_BETTER,
                          unit='milliseconds', short_unit='ms'),
    BaseUserDefinedTarget(target_name='throughput', improvement=MORE_IS_BETTER,
                          unit='transactions / seconds', short_unit='txn/s'),
])
