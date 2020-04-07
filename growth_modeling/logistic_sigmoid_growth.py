r"""This module implements the LogisticSigmoidGrowth class.

Inheriting from the base Growth class, this class implements a logistic sigmoid
 growth from [1]_.

References
----------
.. [1] Colin P. D. Birch. 1999 "A New Generalized Logistic Sigmoid Growth 
   Equation Compared with the Richards Growth Equation."
"""
from growth_modeling import Growth
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

class LogisticSigmoidGrowth(Growth):
    r"""Implement the "Logistic Sigmoid Growth".
    
    Attributes
    ----------
    params_signature : array_like
        An array containing the name of each parameter sorted by how they 
        are called in compute_t and compute_y.
    params : dict
        A dictionary of parameter fit and used by the model to predict. The 
        params dictionary should be ordered and have the following keys: "a", 
        "c", "K" corresponding to the parameters of self.compute_y method.
    bounds : array_like
        Bounds for each parameter similar to the bounds parameter of 
        scipy.curve_fit function.
    y_0 : int
        The response value `y` at time 0. **must be set before calling compute_y**.

    References
    ----------
    .. [1] Colin P. D. Birch. 1999 "A New Generalized Logistic Sigmoid Growth 
       Equation Compared with the Richards Growth Equation."
    """
    def __init__(self, params, bounds):
        r"""Initialize a Logistic Sigmoid Growth Model.
        
        Parameters
        ----------
        params: dict
            dict with the keys corresponding to the params_signature attribute.
        bounds : array_like
            Bounds for each parameter similar to the bounds parameter of 
            scipy.curve_fit function should be order as params_signature.
        """
        super().__init__(params, bounds)
        self.params_signature = ("a", "c", "K")
        self._check_params()
        self.y_0 = NotImplemented
    
    def compute_y(self, t, *args):
        r"""Compute the growth response y at each time provided.
        
        Since this model has no closed form solution for y we need to integrate
        numerically the differential equation.
        
        Parameters
        ----------
        t: array_like
            the time values at which the response will be computed.
        a : float
            The maximum intrinsic rate of increase (RGR) of the response.
            (a > 0)
        c : float
            An additional parameter in the new sigmoid equation introduced
            so that it can define asymmetric curves.
        K : int
            The upper asymptote of the response y.
            
        Returns
        -------
        array_like
            the values of y at each time t provided.

        Notes
        -----
        The logistic Sigmoid Growth equation is studied exetensively in [1]_

        References
        ----------
        .. [1] Colin P. D. Birch. 1999 "A New Generalized Logistic Sigmoid 
           Growth Equation Compared with the Richards Growth Equation."
        """
        if self.y_0 == NotImplemented:
            raise Exception('"y_0" attribute must be set before calling compute_y.')

        a, c, K = self._get_compute_parameters(args)
        
        # differential equation to solve numerically
        def dydt(t, y): 
            return (a * y * (K - y)) / (K - y + c * y)
        
        out = solve_ivp(dydt, (0, max(t)), [self.y_0], t_eval=t)

        return out.y.flatten()
