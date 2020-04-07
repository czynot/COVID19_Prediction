r"""This module implements the ExponentialGeneralizedGrowth class.

Inheriting from the base Growth class, the ExponentialGeneralizedGrowth class 
implements a generalized exponential growth model that can be fit on some data 
using non-linear least square optimisation.
"""
from growth_modeling import Growth
import numpy as np

class ExponentialGeneralizedGrowth(Growth):
    r"""Implement an generalized exponential growth model.
    
    Attributes
    ----------
    params_signature : array_like
        An array containing the name of each parameter sorted by how they 
        are called in compute_t and compute_y.
    params : dict
        A dictionary of parameter fit and used by the model to predict.
        The params dictionary should be ordered and have the following
        key: "a" corresponding to the parameters of self.compute_y method.
    bounds : array_like
        Bounds for each parameter similar to the bounds parameter of 
        scipy.curve_fit function.
    y_0 : int
        The response value `y` at time 0. **must be set before calling compute_y**.
    """ 
    def __init__(self, params, bounds):
        r"""Initialize an Exponential Growth Model.
        
        Parameters
        ----------
        params: dict
            dict with the keys corresponding to the params_signature attribute.
        bounds : array_like
            Bounds for each parameter similar to the bounds parameter of 
            scipy.curve_fit function should be order as params_signature.
        """
        super().__init__(params, bounds)
        self.params_signature = ("a", "p")
        self._check_params()
        self.y_0 = NotImplemented
        
    def compute_y(self, t, *args):
        r"""Compute growth cumulated values using a generalized exponential growth.
        
        If the parameters in \*args are not specified the values from
        self.params are used. If one value from \*args is specified then
        all other values must be specified.
            
        Parameters
        ----------
        t : array_like
            time values for which to compute the response values.
        a : float
            The maximum intrinsic rate of increase (RGR) of the response.
            (a > 0)
        p : float
            allow the model to grow sub-exponentially (p < 1).

        Returns
        -------
        array_like
            the response values corresponding to the growth of t.
        """
        if self.y_0 == NotImplemented:
            raise Exception('"y_0" attribute must be set before calling compute_y.')
        (a, p) = self._get_compute_parameters(args)
        return ((1 - p) * a * t + (1 / self.y_0) ** (p - 1)) ** (-1/(p-1))
