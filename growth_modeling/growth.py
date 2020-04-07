r"""The growth.py module implements the Growth base class.

This module implements the Growth class: a parent class that will be inherited 
by the different specific growth modules. It allows to encapsulate at a shared 
level most of the general behaviors of the different growth models implemented 
in this package.

"""
from scipy.optimize import curve_fit

class Growth:
    r"""A parent class that encapsulate common growth classes behaviors.
    
    Attributes
    ----------
    params_signature : array_like
        An array containing the name of each parameter sorted by how they are 
        called in compute_t and compute_y. This signature should be overriden 
        when creating the child class in the __init__.
    params : dict
        A dictionary of parameter fit and used by the model to predict.
    bounds : array_like
        Bounds for each parameter similar to the bounds parameter of 
        scipy.optimise.curve_fit function.
    """
    def __init__(self, params, bounds=()):
        r"""Initialize a growth class.
        
        Parameters
        ----------
        params : dict
            A dictionary of parameters fit and used by the model to predict. The
             values provided correspond to the initial values of the params 
             (used as initial point for fitting algorithm).
        bounds : array_like
            Bounds for each parameter similar to the bounds parameter of 
            scipy.curve_fit function.
        """
        self.params_signature = NotImplemented
        self.params = params
        self.bounds = bounds

    def compute_t(self, y, *args):
        r"""Compute the time values based on the observed response `y`.
        
        Parameters should be expanded with the specific parameters of the growth
         implemented.
        
        Parameters
        ----------
        y : array_like
            observed response values for which to compute the time values.
        *args : array_like, optional
            list of arguments that match the params_signature attribute if None
             provided then use params attributes.
            
        Returns
        -------
        t : array_like
            time values corresponding to the response values provided `y`.
        """
        raise NotImplementedError

    def compute_y(self, t, *args):
        r"""Compute the response values based on the observed time `t`.
        
        Parameters should be expanded with the specific parameters of the growth
         implemented.
        
        Parameters
        ----------
        t : array_like
            Observed time values `t` for which to compute the response values 
            `y`.
        *args : optional array_like
            list of arguments that match the params_signature attribute if None 
            provided then use params attributes.
            
        Returns
        -------
        y : array_like
            reponse values corresponding to the time values provided `t`.
        """
        raise NotImplementedError
        
    def fit(self, t, y, p0=[]):
        r"""Fit the growth to the provided data and update self.params.
        
        The implementation might varies depending on the growth model concerned 
        however here will be implemented the 
        most common scipy curve_fit strategy based on the compute_y.
        
        Parameters
        ----------
        t : array_like
            observed time values `t` for which to compute the response values.
        y : array_like
            response values `y` corresponding to each time `t`.
        p0: array_like, optional
            corresponding to the parameters specified in params_signature (order
             must match). 

        Returns
        -------
        None
        """
        fit_params, _ = curve_fit(self.compute_y, t, y, bounds=self.bounds, 
                                  p0=tuple(self._get_compute_parameters()))
        for i, key in enumerate(self.params.keys()):
            self.params[key] = fit_params[i]

    def _check_params(self):
        r"""Check that the parameters are compatible with the signature.
        
        Raises
        ------
        AssertionError
            when the keys from `self.params` are not matching with 
            `self.params_signature`.
        """
        assert set(self.params.keys()) == set(self.params_signature)

    def _get_compute_parameters(self, args=[]):
        r"""Get the compute parameters based on the args passed.
        
        This method check if args are provided if not return the stored 
        attributes params.
        
        Parameters
        ----------
        args : array_like
            An array of the parameters passed to `self.compute_y` or 
            `self.compute_t`.
        
        Returns
        -------
        array_like
            a sorted array of parameter value corresponding to 
            `self.params_signature`.
        """
        if len(args) > 0:
            assert len(args) == len(self.params_signature)
            return args
        else:
            self._check_params()
            return (self.params[k] for k in self.params_signature)
