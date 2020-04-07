r"""This module implements the RichardGrowth class.

Inheriting from the base Growth class, this class implements a Richard (1959) 
growth model.
"""
from growth_modeling import Growth
import numpy as np

class RichardGrowth(Growth):
    r"""Implement the Richard's equation growth model.
    
    Attributes
    ----------
    params_signature : array_like
        An array containing the name of each parameter sorted by how they 
        are called in compute_t and compute_y.
    params : dict
        A dictionary of parameter fit and used by the model to predict.
        The params dictionary should be ordered and have the following
        keys: "a", "b", "d", "K" corresponding to the parameters of 
        self.compute_y method.
    bounds : array_like
        Bounds for each parameter similar to the bounds parameter of 
        scipy.curve_fit function.
    """ 
    def __init__(self, params, bounds):
        r"""Initialize a Richard Growth Model.
        
        Parameters
        ----------
        params: dict
            dict with the keys corresponding to the params_signature attribute.
        bounds : array_like
            Bounds for each parameter similar to the bounds parameter of 
            scipy.curve_fit function should be order as params_signature.
        """
        super().__init__(params, bounds)
        self.params_signature = ("a", "b", "d", "K")
        self._check_params()
        
    def compute_y(self, t, *args):
        r"""Compute growth cumulated values using Richard's equation.
        
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
        b : float
            An additional parameter in the Richards equation introduced as
            a power law so that it can define asymmetric curves.(b > 0)
        d : float
            A parameter in the Richards equation which allows the time at 
            which y = K/2 to be varied.
        K : int
            The upper asymptote of the response y.

        Returns
        -------
        array_like
            the response values corresponding to the growth of t.  

        Notes
        -----
        The computation of the response values corresponds to the solution
        to the following differential equation introduced by [1]_:

        .. math:: \frac{\partial y}{\partial t} = ay[1 - (\frac{y}{K})^b]

        which as a solution for y when a > 0 and b > 0 [2]_:

        .. math:: y = K(1 + e^{(d − abt)})^{−1/b}

        References
        ----------
        .. [1] Richards FJ. 1959. "A flexible growth function for empirical use."
           Journal of Experimental Botany 10: 290–300.

        .. [2] Causton DR, Venus JC. 1981. "The biometry of plant growth."
           London: Edward Arnold.
        """
        a, b, d, K = self._get_compute_parameters(args)
        return K * (1 + np.exp(d - a * b * t)) ** (- 1 / b)