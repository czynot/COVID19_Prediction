r"""This module implements the LogisticGrowth class.

Inheriting from the base Growth class, the LogisticGrowth class implements a 
logistic growth model that can be fit on some data using non-linear least square
 optimisation.

"""
from growth_modeling import Growth
import numpy as np

class LogisticGrowth(Growth):
    r"""Implement a logistic growth model.
    
    Attributes
    ----------
    params_signature : array_like
        An array containing the name of each parameter sorted by how they 
        are called in compute_t and compute_y.
    params : dict
        A dictionary of parameter fit and used by the model to predict.
        The params dictionary should be ordered and have the following
        keys: "a", "t_0", "K" corresponding to the parameters of self.compute_y 
        method.
    bounds : array_like
        Bounds for each parameter similar to the bounds parameter of 
        scipy.curve_fit function.
    """ 
    def __init__(self, params, bounds):
        r"""Initialize a Logistic Growth Model.
        
        Parameters
        ----------
        params: dict
            dict with the keys corresponding to the params_signature attribute.
        bounds : array_like
            Bounds for each parameter similar to the bounds parameter of 
            scipy.curve_fit function should be order as params_signature.
        """
        super().__init__(params, bounds)
        self.params_signature = ("a", "t_0", "K")
        self._check_params()
        
    def compute_y(self, t, *args):
        r"""Compute growth cumulated values using the logisitc growth equation.
        
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
        t_0 : int
            time at which y = K/2.
        K : int
            The upper asymptote of the response y.

        Returns
        -------
        array_like
            the response values corresponding to the growth of t.
        """
        a, t_0, K = self._get_compute_parameters(args)
        return K / (1 + np.exp(- a * (t - t_0)))
