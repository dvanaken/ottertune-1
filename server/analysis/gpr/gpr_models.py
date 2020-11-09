#
# OtterTune - analysis/gpr_models.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
# Author: Dana Van Aken

import copy
import json
import os

import gpflow
from gpflow.models import GPR
import numpy as np
import tensorflow as tf

#from .gprc import GPRC as GPR
from analysis.util import get_analysis_logger

LOG = get_analysis_logger(__name__)


class BaseModel(object):

    # Min/max bounds for the kernel lengthscales
    _LENGTHSCALE_BOUNDS = (0.1, 10.)

    _DEFAULT_OPTIMIZE = False
    _DEFAULT_LEARNING_RATE = 0.001
    _DEFAULT_MAXITER = 5000
    _DEFAULT_LENGTHSCALES = 2.0
    _DEFAULT_VARIANCE = 1.0
    _DEFAULT_ARD = False
    #_DEFAULT_LIKELIHOOD = 1.0

    def __init__(self, X, y, **kwargs):
        self.optimize_hyperparameters = kwargs.pop('optimize_hyperparameters', self._DEFAULT_OPTIMIZE)
        self.learning_rate = kwargs.pop('learning_rate', self._DEFAULT_LEARNING_RATE)
        self.maxiter = kwargs.pop('maxiter', self._DEFAULT_MAXITER)
        self.lengthscales = kwargs.pop('lengthscales', self._DEFAULT_LENGTHSCALES)
        self.variance = kwargs.pop('variance', self._DEFAULT_VARIANCE)
        self.ARD = kwargs.pop('ARD', self._DEFAULT_ARD)
        self.X_dim = X.shape[1]
        #self.likelihood = kwargs.pop('likelihood', self._DEFAULT_LIKELIHOOD)

        LOG.info("module=%s, optimize=%s, learning_rate=%s, maxiter=%s, lengthscales=%s, variance=%s, "
                 "ARD=%s, X_dim=%s", GPR.__module__, self.optimize_hyperparameters, self.learning_rate,
                 self.maxiter, self.lengthscales, self.variance, self.ARD, self.X_dim)

        # Build the kernels and the model
        with gpflow.defer_build():
            kernel, subkernels = self._build_kernel(**kwargs)
            m = GPR(X, y, kern=kernel)
            #m.likelihood.variance = self.likelihood 
            #m.likelihood.variance.trainable = self.optimize_hyperparameters
            for k in (kernel,) + subkernels:
                if hasattr(k, 'ARD'):
                    # If ARD is enabled, the kernel's lengthscales arg must be an array of length X_dim
                    k.ARD = self.ARD
                if hasattr(k, 'variance'):
                    k.variance = self.variance
                if hasattr(k, 'lengthscales'):
                    k.lengthscales = self.lengthscales
            for param in m.parameters:
                if self.optimize_hyperparameters and param.pathname.endswith('lengthscales'):
                    param.transform = gpflow.transforms.Logistic(*self._LENGTHSCALE_BOUNDS)
                param.trainable = self.optimize_hyperparameters
        m.compile()
        LOG.info("MODEL HYPERPARAMETERS (optimize=%s):\n%s\n",
                 self.optimize_hyperparameters, m.as_pandas_table())

        # If enabled, optimize the hyperparameters
        if self.optimize_hyperparameters:
            opt = gpflow.train.AdamOptimizer(self.learning_rate)
            opt.minimize(m, maxiter=self.maxiter)
            for param in m.parameters:
                param.trainable = False
            LOG.info("OPTIMIZED HYPERPARAMETERS:\n%s\n", m.as_pandas_table())
        self.model = m
        self.kernel = kernel
        self.subkernels = subkernels

    def _build_kernel(self, **kwargs):
        return None


class BasicGP(BaseModel):

    def _build_kernel(self, **kwargs):
        k = gpflow.kernels.Matern12(
            input_dim=self.X_dim,
            #variance=self.variance,
            #lengthscales=self.lengthscales,
            #ARD=False,
        )
        return k, ()


class ExpWhiteGP(BasicGP):

    def _build_kernel(self, **kwargs):
        k0, _ = super()._build_kernel(**kwargs)
        #k0 = gpflow.kernels.Exponential(**kernel_kwargs[0])
        k1 = gpflow.kernels.White(input_dim=self.X_dim)
        k = k0 + k1
        return k, (k0, k1)


class ContextualGP(BaseModel):

    def _build_kernel(self, **kwargs):
        k0_active_dims = kwargs.pop('k0_active_dims')
        k1_active_dims = kwargs.pop('k1_active_dims')
        k0 = gpflow.kernels.Matern12(
            input_dim=len(k0_active_dims),
            active_dims=k0_active_dims,
            #input_dim=self.X_dim,
            #variance=self.variance,
            #lengthscales=self.lengthscales,
            #ARD=False,
        )
        k1 = gpflow.kernels.Matern12(
            input_dim=len(k1_active_dims),
            active_dims=k1_active_dims,
            #input_dim=self.X_dim,
            #variance=self.variance,
            #lengthscales=self.lengthscales,
            #ARD=False,
        )
        k = k0 * k1
        return k, (k0, k1)


class ContextualWhiteGP(ContextualGP):

    def _build_kernel(self, **kwargs):
        _, (k0, k1) = super()._build_kernel(**kwargs)
        k2 = gpflow.kernels.White(input_dim=self.X_dim)
        k = k0 * k1 + k2
        return k, (k0, k1, k2)


_MODEL_MAP = {
    'BasicGP': BasicGP,
    'ExpWhiteGP': ExpWhiteGP,
    'ContextualGP': ContextualGP,
    'ContextualWhiteGP': ContextualWhiteGP,
}


def create_model(model_name, **kwargs):
    # Update tensorflow session settings to enable GPU sharing
    gpflow.settings.session.update(gpu_options=tf.GPUOptions(allow_growth=True))
    check_valid(model_name)
    return _MODEL_MAP[model_name](**kwargs)


def check_valid(model_name):
    if model_name not in _MODEL_MAP:
        raise ValueError('Invalid GPR model name: {}'.format(model_name))
