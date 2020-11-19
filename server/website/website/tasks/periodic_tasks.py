#
# OtterTune - periodic_tasks.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
import copy
import time
import traceback
import numpy as np
from pytz import timezone

from celery import shared_task
from celery.utils.log import get_task_logger
from django.db import transaction
from django.db.models import Count
from django.utils.timezone import now
from django.utils.datetime_safe import datetime
from sklearn.preprocessing import StandardScaler

from analysis.cluster import KMeansClusters, create_kselection_model
from analysis.factor_analysis import FactorAnalysis
from analysis.lasso import LassoPath
from analysis.preprocessing import (Bin, get_shuffle_indices,
                                    DummyEncoder,
                                    consolidate_columnlabels)
from website.models import PipelineData, PipelineRun, Result, Session, Workload, ExecutionTime
from website.settings import (ENABLE_DUMMY_ENCODER, KNOB_IDENT_USE_PRUNED_METRICS,
                              MIN_WORKLOAD_RESULTS_COUNT, TIME_ZONE, VIEWS_FOR_PRUNING,
                              PRUNED_METRICS_MIN_CLUSTERS, PRUNED_METRICS_MAX_CLUSTERS,
                              BG_TASKS_PROCESS_COMBINED_DATA)
from website.types import PipelineTaskType, WorkloadStatusType
from website.utils import DataUtil, JSONUtil

# Log debug messages
LOG = get_task_logger(__name__)


def save_execution_time(start_ts, fn):
    end_ts = time.time()
    exec_time = end_ts - start_ts
    start_time = datetime.fromtimestamp(int(start_ts), timezone(TIME_ZONE))
    ExecutionTime.objects.create(module="celery.periodic_tasks", function=fn, tag="",
                                 start_time=start_time, execution_time=exec_time, result=None)


def get_workload_name(workload):
    name = workload.name
    if workload.project.name != workload.name:
        name = '{}.{}'.format(workload.project.name, name)
    return '{}@{}'.format(workload.dbms.key, name)


@shared_task(name="run_background_tasks")
def run_background_tasks():
    start_ts = time.time()
    LOG.info("Starting background tasks...")

    with transaction.atomic():
        modified_workloads = Workload.objects.filter(status=WorkloadStatusType.MODIFIED)
        modified_workload_ids = list(modified_workloads.values_list('id', flat=True))
        if len(modified_workloads) == 0:
            # No previous workload data yet. Try again later.
            LOG.info("No modified workload data yet. Ending background tasks.")
            return

        # Create new entry in PipelineRun table to store the output of each of
        # the background tasks
        pipeline_run_obj = PipelineRun.objects.create(start_time=now(), end_time=None)
        pipeline_run_id = pipeline_run_obj.id
        modified_workloads.update(status=WorkloadStatusType.PROCESSING)

    LOG.info("Starting pipeline run %s (modified=%s, process_combined_data=%s)",
             pipeline_run_id, len(modified_workload_ids), BG_TASKS_PROCESS_COMBINED_DATA)

    try:
        with transaction.atomic():
            bg_wkld = Workload.objects.get(name='backgroundtasks')
            modified_workloads = Workload.objects.filter(id__in=modified_workload_ids)
            if BG_TASKS_PROCESS_COMBINED_DATA:
                modified_workloads = list(modified_workloads) + [bg_wkld]
            num_modified = len(modified_workloads)

            # Process modified workloads
            LOG.info("Processing %s workloads: %s", num_modified,
                     ', '.join(get_workload_name(w) for w in modified_workloads))
            for i, workload in enumerate(modified_workloads):
                workload_name = get_workload_name(workload)

                if bg_wkld == workload:
                    wkld_results = Result.objects.all()
                else:
                    wkld_results = Result.objects.filter(workload=workload)

                num_wkld_results = wkld_results.count()

                LOG.info("Starting workload %s (%s/%s, # results: %s)...", workload_name,
                         i + 1, num_modified, num_wkld_results)

                if num_wkld_results == 0:
                    # delete the workload
                    LOG.info("Deleting workload %s because it has no results.", workload_name)
                    workload.delete()
                    continue

                if num_wkld_results < MIN_WORKLOAD_RESULTS_COUNT:
                    # Check that there are enough results in the workload
                    LOG.info("Not enough results in workload %s (# results: %s, # required: %s).",
                             workload_name, num_wkld_results, MIN_WORKLOAD_RESULTS_COUNT)
                    workload.status = WorkloadStatusType.PROCESSED
                    workload.save()
                    continue

                LOG.info("Aggregating data for workload %s...", workload_name)
                # Aggregate the knob & metric data for this workload
                knob_data, metric_data = aggregate_data(wkld_results)
                LOG.info("Done aggregating data for workload %s.", workload_name)

                num_valid_results = knob_data['data'].shape[0]  # pylint: disable=unsubscriptable-object
                if num_valid_results < MIN_WORKLOAD_RESULTS_COUNT:
                    # Check that there are enough valid results in the workload
                    LOG.info("Not enough valid results in workload %s (# valid results: "
                             "%s, # required: %s).", workload_name, num_valid_results,
                             MIN_WORKLOAD_RESULTS_COUNT)
                    workload.status = WorkloadStatusType.PROCESSED
                    workload.save()
                    continue

                # Knob_data and metric_data are 2D numpy arrays. Convert them into a
                # JSON-friendly (nested) lists and then save them as new PipelineData
                # objects.
                knob_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                          task_type=PipelineTaskType.KNOB_DATA,
                                          workload=workload,
                                          data=JSONUtil.dumps(knob_data),
                                          creation_time=now())
                knob_entry.save()

                metric_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                            task_type=PipelineTaskType.METRIC_DATA,
                                            workload=workload,
                                            data=JSONUtil.dumps(metric_data),
                                            creation_time=now())
                metric_entry.save()

                # Execute the Workload Characterization task to compute the list of
                # pruned metrics for this workload and save them in a new PipelineData
                # object.
                LOG.info("Pruning metrics for workload %s...", workload_name)
                pruned_metrics = run_workload_characterization(
                    metric_data=metric_data, dbms=workload.dbms)
                LOG.info("Done pruning metrics for workload %s (# pruned metrics: %s).\n\n"
                         "Pruned metrics: %s\n", workload_name, len(pruned_metrics),
                         pruned_metrics)
                pruned_metrics_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                                    task_type=PipelineTaskType.PRUNED_METRICS,
                                                    workload=workload,
                                                    data=JSONUtil.dumps(pruned_metrics),
                                                    creation_time=now())
                pruned_metrics_entry.save()

                ranked_knobs = run_knob_identification(knob_data,
                                                       metric_data,
                                                       wkld_results,
                                                       pruned_metrics,
                                                       workload_name=workload_name)
                LOG.info("Done ranking knobs for workload %s (# ranked knobs: %s).\n\n"
                         "Ranked knobs: %s\n", workload_name, len(ranked_knobs), ranked_knobs)
                ranked_knobs_entry = PipelineData(pipeline_run=pipeline_run_obj,
                                                  task_type=PipelineTaskType.RANKED_KNOBS,
                                                  workload=workload,
                                                  data=JSONUtil.dumps(ranked_knobs),
                                                  creation_time=now())
                ranked_knobs_entry.save()

                workload.status = WorkloadStatusType.PROCESSED
                workload.save()
                LOG.info("Done processing workload %s (%s/%s).", workload_name, i + 1,
                         num_modified)

            LOG.info("Finished processing %s modified workloads.", num_modified)

            exclude_ids = modified_workload_ids + [bg_wkld.id]
            if bg_wkld.id not in exclude_ids:
                exclude_ids.append(bg_wkld.id)

            non_modified_workloads = Workload.objects.exclude(id__in=exclude_ids)
            for wkld in non_modified_workloads:
                run_id = PipelineData.objects.filter(workload=wkld).values(
                    'pipeline_run').annotate(total=Count('pipeline_run')).filter(
                        total=4).order_by('pipeline_run').values_list(
                            'pipeline_run',flat=True).first()    
                if run_id is not None:
                    PipelineData.objects.filter(workload=wkld, pipeline_run__id=run_id).update(
                        pipeline_run=pipeline_run_obj)

            # Set the end_timestamp to the current time to indicate that we are done running
            # the background tasks
            pipeline_run_obj.end_time = now()
            pipeline_run_obj.save()
            LOG.info("Saved pipeline run end time: %s", pipeline_run_obj.end_time)

    finally:
        tb = traceback.format_exc()
        if tb.startswith('NoneType'):
            LOG.info("No exceptions") 
        else:
            LOG.error(tb)
        pipeline_run_obj = PipelineRun.objects.get(id=pipeline_run_id)
        if pipeline_run_obj.end_time is None:
            LOG.warning("PipelineRun %s failed (end_time=None). Deleting...", pipeline_run_id)
            for workload in modified_workloads:
                if workload.status == WorkloadStatusType.PROCESSING:
                    workload.status = WorkloadStatusType.MODIFIED
                    workload.save()
            pipeline_run_obj.delete()

    save_execution_time(start_ts, "run_background_tasks")
    LOG.info("Finished background tasks (%.0f seconds).", time.time() - start_ts)


def aggregate_data(wkld_results, combine_duplicate_rows=True):
    # Aggregates both the knob & metric data for the given workload.
    #
    # Parameters:
    #   wkld_results: result data belonging to this specific workload
    #
    # Returns: two dictionaries containing the knob & metric data as
    # a tuple

    # Now call the aggregate_data helper function to combine all knob &
    # metric data into matrices and also create row/column labels
    # (see the DataUtil class in website/utils.py)
    #
    # The aggregate_data helper function returns a dictionary of the form:
    #   - 'X_matrix': the knob data as a 2D numpy matrix (results x knobs)
    #   - 'y_matrix': the metric data as a 2D numpy matrix (results x metrics)
    #   - 'rowlabels': list of result ids that correspond to the rows in
    #         both X_matrix & y_matrix
    #   - 'X_columnlabels': a list of the knob names corresponding to the
    #         columns in the knob_data matrix
    #   - 'y_columnlabels': a list of the metric names corresponding to the
    #         columns in the metric_data matrix
    start_ts = time.time()
    agg_data = DataUtil.aggregate_data(
        wkld_results, ignore=['range_test', 'default', '*'])

    X, y = agg_data['X_matrix'], agg_data['y_matrix']
    X_cls, y_cls = agg_data['X_columnlabels'], agg_data['y_columnlabels']
    rls = agg_data['rowlabels']

    X, y, rls = DataUtil.combine_duplicate_rows(X, y, np.array(rls), dup_test='Xy')
    if isinstance(rls, np.ndarray):
        rls = rls.tolist()

    LOG.debug("Aggregated data: X_matrix=%s, y_matrix=%s, X_columnlabels=%s, "
              "y_columnlabels=%s, rowlabels=%s", X.shape, y.shape, len(X_cls),
              len(y_cls), len(rls))

    # Separate knob & workload data into two "standard" dictionaries of the
    # same form
    knob_data = {
        'data': X,
        'rowlabels': rls,
        'columnlabels': X_cls 
    }

    metric_data = {
        'data': y,
        'rowlabels': copy.deepcopy(rls),
        'columnlabels': y_cls 
    }

    # Return the knob & metric data
    save_execution_time(start_ts, "aggregate_data")
    return knob_data, metric_data


def run_workload_characterization(metric_data, dbms=None):
    # Performs workload characterization on the metric_data and returns
    # a set of pruned metrics.
    #
    # Parameters:
    #   metric_data is a dictionary of the form:
    #     - 'data': 2D numpy matrix of metric data (results x metrics)
    #     - 'rowlabels': a list of identifiers for the rows in the matrix
    #     - 'columnlabels': a list of the metric names corresponding to
    #                       the columns in the data matrix
    start_ts = time.time()

    matrix = metric_data['data']
    columnlabels = metric_data['columnlabels']
    LOG.debug("Workload characterization ~ initial data size: %s", matrix.shape)

    views = None if dbms is None else VIEWS_FOR_PRUNING.get(dbms.type, None)
    matrix, columnlabels = DataUtil.clean_metric_data(matrix, columnlabels, views)
    LOG.debug("Workload characterization ~ cleaned data size: %s", matrix.shape)

    # Bin each column (metric) in the matrix by its decile
    binner = Bin(bin_start=1, axis=0)
    binned_matrix = binner.fit_transform(matrix)

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, cl in zip(binned_matrix.T, columnlabels):
        if np.any(col != col[0]):
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(cl)
    #assert len(nonconst_matrix) > 0, "Need more data to train the model"
    if len(nonconst_matrix) == 0:
        return []
    nonconst_matrix = np.hstack(nonconst_matrix)
    LOG.debug("Workload characterization ~ nonconst data size: %s", nonconst_matrix.shape)

    # Remove any duplicate columns
    unique_matrix, unique_idxs = np.unique(nonconst_matrix, axis=1, return_index=True)
    unique_columnlabels = [nonconst_columnlabels[idx] for idx in unique_idxs]

    LOG.debug("Workload characterization ~ final data size: %s", unique_matrix.shape)
    n_rows, n_cols = unique_matrix.shape

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix = unique_matrix[shuffle_indices, :]

    # Fit factor analysis model
    fa_model = FactorAnalysis()
    # For now we use 5 latent variables
    fa_model.fit(shuffled_matrix, unique_columnlabels, n_components=5)

    # Components: metrics * factors
    components = fa_model.components_.T.copy()
    LOG.info("Workload characterization first part costs %.0f seconds.", time.time() - start_ts)

    # Run Kmeans for # clusters k in range(1, num_nonduplicate_metrics - 1)
    # K should be much smaller than n_cols in detK, For now max_cluster <= 20
    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=PRUNED_METRICS_MIN_CLUSTERS,
                      max_cluster=min(n_cols - 1, PRUNED_METRICS_MAX_CLUSTERS),
                      sample_labels=unique_columnlabels,
                      estimator_params={'n_init': 50})

    # Compute optimal # clusters, k, using gap statistics
    gapk = create_kselection_model("gap-statistic")
    gapk.fit(components, kmeans_models.cluster_map_)

    LOG.debug("Found optimal number of clusters: %d", gapk.optimal_num_clusters_)

    # Get pruned metrics, cloest samples of each cluster center
    pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()

    # Return pruned metrics
    save_execution_time(start_ts, "run_workload_characterization")
    LOG.info("Workload characterization finished in %.0f seconds.", time.time() - start_ts)
    return pruned_metrics


def run_knob_identification(knob_data, metric_data, results, pruned_metrics,
                            workload_name=None):
    # Performs knob identification on the knob & metric data and returns
    # a set of ranked knobs.
    #
    # Parameters:
    #   knob_data & metric_data are dictionaries of the form:
    #     - 'data': 2D numpy matrix of knob/metric data
    #     - 'rowlabels': a list of identifiers for the rows in the matrix
    #     - 'columnlabels': a list of the knob/metric names corresponding
    #           to the columns in the data matrix
    #   dbms is the foreign key pointing to target dbms in DBMSCatalog
    #
    # When running the lasso algorithm, the knob_data matrix is set of
    # independent variables (X) and the metric_data is the set of
    # dependent variables (y).
    start_ts = time.time()

    workload_name = workload_name or results.first().workload.name
    dbms = results.first().dbms

    ranked_knob_metrics = sorted(results.values_list(
        'session__target_objective', flat=True).distinct())
    LOG.debug("Target objectives for workload %s: %s", workload_name,
              ', '.join(ranked_knob_metrics))

    if KNOB_IDENT_USE_PRUNED_METRICS:
        ranked_knob_metrics = sorted(set(ranked_knob_metrics) | set(pruned_metrics))

    # Use the set of metrics to filter the metric_data
    metric_idxs = [i for i, metric_name in enumerate(metric_data['columnlabels'])
                   if metric_name in ranked_knob_metrics]
    rank_metric_data = {
        'data': metric_data['data'][:, metric_idxs],
        'rowlabels': copy.deepcopy(metric_data['rowlabels']),
        'columnlabels': [metric_data['columnlabels'][i] for i in metric_idxs]
    }

    LOG.info("Ranking knobs for workload %s (use pruned metric data: %s)...",
             workload_name, KNOB_IDENT_USE_PRUNED_METRICS)
    sessions = [r.session for r in results.distinct('session')]

    rank_knob_data = copy.deepcopy(knob_data)
    rank_knob_data['data'], rank_knob_data['columnlabels'] =\
        DataUtil.clean_knob_data(knob_data['data'], knob_data['columnlabels'], sessions)

    knob_matrix = rank_knob_data['data']
    knob_columnlabels = rank_knob_data['columnlabels']

    metric_matrix = rank_metric_data['data']
    metric_columnlabels = rank_metric_data['columnlabels']

    # remove constant columns from knob_matrix and metric_matrix
    nonconst_knob_matrix = []
    nonconst_knob_columnlabels = []

    for col, cl in zip(knob_matrix.T, knob_columnlabels):
        #if np.any(col != col[0]):
        nonconst_knob_matrix.append(col.reshape(-1, 1))
        nonconst_knob_columnlabels.append(cl)
    #assert len(nonconst_knob_matrix) > 0, "Need more data to train the model"
    if len(nonconst_knob_matrix) == 0:
        return []
    nonconst_knob_matrix = np.hstack(nonconst_knob_matrix)

    nonconst_metric_matrix = []
    nonconst_metric_columnlabels = []

    for col, cl in zip(metric_matrix.T, metric_columnlabels):
        if np.any(col != col[0]):
            nonconst_metric_matrix.append(col.reshape(-1, 1))
            nonconst_metric_columnlabels.append(cl)
    if len(nonconst_metric_matrix) == 0:
        return []
    nonconst_metric_matrix = np.hstack(nonconst_metric_matrix)

    if ENABLE_DUMMY_ENCODER:
        # determine which knobs need encoding (enums with >2 possible values)

        categorical_info = DataUtil.dummy_encoder_helper(nonconst_knob_columnlabels,
                                                         dbms)
        # encode categorical variable first (at least, before standardize)
        dummy_encoder = DummyEncoder(categorical_info['n_values'],
                                     categorical_info['categorical_features'],
                                     categorical_info['cat_columnlabels'],
                                     categorical_info['noncat_columnlabels'])
        encoded_knob_matrix = dummy_encoder.fit_transform(
            nonconst_knob_matrix)
        encoded_knob_columnlabels = dummy_encoder.new_labels
    else:
        encoded_knob_columnlabels = nonconst_knob_columnlabels
        encoded_knob_matrix = nonconst_knob_matrix

    # standardize values in each column to N(0, 1)
    standardizer = StandardScaler()
    standardized_knob_matrix = standardizer.fit_transform(encoded_knob_matrix)
    standardized_metric_matrix = standardizer.fit_transform(nonconst_metric_matrix)

    # shuffle rows (note: same shuffle applied to both knob and metric matrices)
    shuffle_indices = get_shuffle_indices(standardized_knob_matrix.shape[0], seed=17)
    shuffled_knob_matrix = standardized_knob_matrix[shuffle_indices, :]
    shuffled_metric_matrix = standardized_metric_matrix[shuffle_indices, :]

    # run lasso algorithm
    lasso_model = LassoPath()
    lasso_model.fit(shuffled_knob_matrix, shuffled_metric_matrix, encoded_knob_columnlabels)

    # consolidate categorical feature columns, and reset to original names
    encoded_knobs = lasso_model.get_ranked_features()
    consolidated_knobs = consolidate_columnlabels(encoded_knobs)

    save_execution_time(start_ts, "run_knob_identification")
    LOG.info("Knob identification finished in %.0f seconds.", time.time() - start_ts)
    return consolidated_knobs
