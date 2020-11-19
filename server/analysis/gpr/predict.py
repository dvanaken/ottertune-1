import numpy as np
from sklearn.utils import assert_all_finite, check_array
from sklearn.utils.validation import FLOAT_DTYPES


class GPRResult():

    def __init__(self, ypreds=None, sigmas=None):
        self.ypreds = ypreds
        self.sigmas = sigmas


def gpflow_predict(model, Xin):
    Xin = check_array(Xin, copy=False, warn_on_dtype=True, dtype=FLOAT_DTYPES)
    y_mean, y_var = model.predict_y(Xin)
    assert_all_finite(y_mean)
    assert_all_finite(y_var)
    y_std = np.sqrt(y_var)
    return GPRResult(y_mean, y_std)
