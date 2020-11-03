#
# OtterTune - analysis/optimize.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

import numpy as np
import tensorflow as tf
from gpflow import settings
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.validation import FLOAT_DTYPES

from analysis.util import get_analysis_logger

LOG = get_analysis_logger(__name__)

_DEBUG_ROWS = [-2, -1]
_DEBUG_NCOLS = 5


class GPRGDResult():

    def __init__(self, ypreds=None, sigmas=None, minl=None, minl_conf=None, minl_context=None):
        self.ypreds = ypreds
        self.sigmas = sigmas
        self.minl = minl
        self.minl_conf = minl_conf
        self.minl_context = minl_context

    def __str__(self):
        attrs = []
        for k, v in sorted(self.__dict__.items()):
            if not k.startswith('_'):
                v = v.shape if isinstance(v, np.ndarray) else v
                attrs.append('{}={}'.format(k, v))
        return "{}({})".format(self.__class__.__name__, ', '.join(attrs))

    def __repr__(self):
        return self.__str__()


def tf_optimize(model, Xnew_arr, learning_rate=0.01, maxiter=100, ucb_beta=3.,
                Xctx_arr=None, bounds=(-np.infty, np.infty), debug=True):
    try:
        Xnew_arr = check_array(Xnew_arr, copy=False, warn_on_dtype=True, dtype=FLOAT_DTYPES)
    except ValueError:
        np.savetxt('ERROR_Xnew_arr.csv', Xnew_arr, delimiter=',')
        raise

    nsamples, X_dim = Xnew_arr.shape
    if Xctx_arr is None:
        Xctx_arr = np.empty((nsamples, 0))
    ctx_dim = Xctx_arr.shape[1]
    Xctx_arr = check_array(Xctx_arr, copy=False, warn_on_dtype=True, dtype=FLOAT_DTYPES,
                           ensure_min_features=0)
    if bounds is None:
        bounds = [-np.infty, np.infty]

    LOG.info("learning_rate=%s, maxiter=%s, ucb_beta=%s, Xctx_arr=%s, bounds=%s, debug=%s\n",
             learning_rate, maxiter, ucb_beta, Xctx_arr.shape, np.asarray(bounds).shape, debug)
    Xnew = tf.Variable(Xnew_arr, name='Xnew', dtype=settings.float_type)

    lower_bound = tf.constant(bounds[0], dtype=settings.float_type)
    upper_bound = tf.constant(bounds[1], dtype=settings.float_type)
    Xnew_bounded = tf.minimum(tf.maximum(Xnew, lower_bound), upper_bound)

    #if Xctx_arr.size == 0:
    #    Xin = Xnew_bounded
    #else:
    #    Xctx = tf.constant(Xctx_arr, name='Xctx', dtype=settings.float_type)
    #    Xin = tf.concat([Xnew_bounded, Xctx], axis=1) 
    Xctx = tf.constant(Xctx_arr, name='Xctx', dtype=settings.float_type)
    Xin = tf.concat([Xnew_bounded, Xctx], axis=1) 

    #if ctx_ver == 0:
    #    Xnew = tf.Variable(Xnew_arr, name='Xnew', dtype=settings.float_type)
    #    if bounds is None:
    #        lower_bound = tf.constant(-np.infty, dtype=settings.float_type)
    #        upper_bound = tf.constant(np.infty, dtype=settings.float_type)
    #    else:
    #        lower_bound = tf.constant(bounds[0], dtype=settings.float_type)
    #        upper_bound = tf.constant(bounds[1], dtype=settings.float_type)
    #    Xnew_bounded = tf.minimum(tf.maximum(Xnew, lower_bound), upper_bound)

    #    if active_dims is None:
    #        Xin = Xnew_bounded
    #    else:
    #        indices = []
    #        updates = []
    #        n_rows = Xnew_arr.shape[0]
    #        for c in active_dims:
    #            for r in range(n_rows):
    #                indices.append([r, c])
    #                updates.append(Xnew_bounded[r, c])
    #        part_X = tf.scatter_nd(indices, updates, Xnew_arr.shape)
    #        Xin = part_X + tf.stop_gradient(-part_X + Xnew_bounded)

    #else:
    #    Xnew = tf.Variable(Xnew_arr, name='Xnew', dtype=settings.float_type)
    #    Xctx = tf.constant(Xctx_arr, name='Xctx', dtype=settings.float_type)

    #    if bounds is None:
    #        lower_bound = tf.constant(-np.infty, dtype=settings.float_type)
    #        upper_bound = tf.constant(np.infty, dtype=settings.float_type)
    #    else:
    #        lower_bound = tf.constant(bounds[0][:43], dtype=settings.float_type)
    #        upper_bound = tf.constant(bounds[1][:43], dtype=settings.float_type)
    #    Xnew_bounded = tf.minimum(tf.maximum(Xnew, lower_bound), upper_bound)

    #    if active_dims is None:
    #        Xin = Xnew_bounded
    #    else:
    #        Xin = tf.concat([Xnew_bounded, Xctx], axis=1) 

    beta_t = tf.constant(ucb_beta, name='ucb_beta', dtype=settings.float_type)
    ##fmean, fvar, kvar, kls, lvar = model._build_predict(Xin)  # pylint: disable=protected-access
    ##y_mean_var = model.likelihood.predict_mean_and_var(fmean, fvar)
    ##y_mean, y_var = model.predict_y(Xin)
    fmean, fvar = model._build_predict(Xin)  # pylint: disable=protected-access
    y_mean, y_var = model.likelihood.predict_mean_and_var(fmean, fvar)
    y_std = tf.sqrt(y_var)
    loss = tf.subtract(y_mean, tf.multiply(beta_t, y_std), name='loss_fn')
    opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-6)
    train_op = opt.minimize(loss)
    variables = opt.variables()
    init_op = tf.variables_initializer([Xnew] + variables)
    session = model.enquire_session(session=None)
    with session.as_default():
        session.run(init_op)

        if debug:
            Xin0 = session.run(Xin)
            Xnew0 = session.run(Xnew_bounded)
            Xctx0 = session.run(Xctx)
            out = "\n{0}\n INITIAL VALUES\n{0}\n".format('*' * 100)
            out += "XNEW:\n{}\n\nXCTX:\n{}\n\n".format(
                     Xnew0[_DEBUG_ROWS, :_DEBUG_NCOLS],
                     Xctx0[_DEBUG_ROWS, :_DEBUG_NCOLS])
            out += "MODEL INFO:\n{}\nbeta_t        {:.3f}\n{}\n\n".format(
                model.as_pandas_table(), session.run(beta_t), '*' * 100)
            LOG.info(out)
            #LOG.info("\n\nINITIAL Xnew: %s\nINITIAL Xctx: %s\n",
            #         Xnew0[_DEBUG_ROWS, :_DEBUG_NCOLS],
            #         Xctx0[_DEBUG_ROWS, :_DEBUG_NCOLS])
            Xin0_new = Xin0[:, :X_dim]
            Xin0_ctx = Xin0[:, X_dim:]
            if not np.allclose(Xin0_new, Xnew0):
                LOG.warning("INITIAL Xin-new and Xnew differ. Diffs: %s/%s",
                            np.sum(Xin0_new != Xnew0), Xnew0.size)
            if not np.allclose(Xin0_ctx, Xctx0):
                LOG.warning("INITIAL Xin-ctx and Xctx differ. Diffs: %s/%s",
                            np.sum(Xin0_ctx != Xctx0), Xctx0.size)
            #LOG.info('\nINITIAL MODEL:\n%s\nucb_beta          %.3f\n',
            #         model.as_pandas_table(), session.run(beta_t))

        for i in range(maxiter):
            session.run(train_op)

        Xnew_value = session.run(Xnew_bounded)
        Xctx_value = session.run(Xctx)
        y_mean_value = session.run(y_mean)
        y_std_value = session.run(y_std)
        loss_value = session.run(loss)

        if debug:
            Xin1 = session.run(Xin)
            out = "\n{0}\n FINAL VALUES\n{0}\n".format('*' * 100)
            out += "XNEW:\n{}\n\nXCTX:\n{}\n\n".format(
                     Xnew_value[_DEBUG_ROWS, :_DEBUG_NCOLS],
                     Xctx_value[_DEBUG_ROWS, :_DEBUG_NCOLS])
            out += "MODEL INFO:\n{}\nbeta_t        {:.3f}\n{}\n\n".format(
                model.as_pandas_table(), session.run(beta_t), '*' * 100)
            LOG.info(out)
            #LOG.info("\n\nFINAL Xnew: %s\nFINAL Xctx: %s\n",
            #         Xnew_value[_DEBUG_ROWS, :_DEBUG_NCOLS],
            #         Xctx_value[_DEBUG_ROWS, :_DEBUG_NCOLS])
            Xin1_new = Xin1[:, :X_dim]
            Xin1_ctx = Xin1[:, X_dim:]
            if not np.allclose(Xin1_new, Xnew_value):
                LOG.warning("FINAL Xin-new and Xnew differ. Diffs: %s/%s",
                            np.sum(Xin1_new != Xnew_value), Xnew_value.size)
            if not np.allclose(Xin1_ctx, Xctx_value):
                LOG.warning("FINAL Xin-ctx and Xctx differ. Diffs: %s/%s",
                            np.sum(Xin1_ctx != Xctx_value), Xctx_value.size)
            #LOG.info('\nFINAL MODEL:\n%s\nucb_beta          %.3f\n',
            #         model.as_pandas_table(), session.run(beta_t))

        assert_all_finite(Xnew_value)
        assert_all_finite(Xctx_value)
        assert_all_finite(y_mean_value)
        assert_all_finite(y_std_value)
        assert_all_finite(loss_value)
        result = GPRGDResult(y_mean_value, y_std_value, loss_value, Xnew_value, Xctx_value)

        return result
