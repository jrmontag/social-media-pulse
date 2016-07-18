#!/usr/bin/env python
from __future__ import print_function

__author__="Scott Hendrickson, Josh Montague"
__email__="shendrickson@twitter.com"
__license__="APL 2.0, http://www.apache.org/licenses/LICENSE-2.0"

from inspect import cleandoc
import sys
import math
import scipy.special
import numpy as np

##### 
# (optionally) turn off numpy warnings
#
disable_np_warn = False 
#
####

if disable_np_warn:
    np.seterr(all='ignore')
    sys.stderr.write("WARNING: all Numpy errors currently disabled (setting in models.py)") 


class BaseModel(object):
    """
    The BaseModel class defines the expected methods of custom model classes. 
    """
    def __init__(self):
        self.parameters = [ 1.0 ]

    def guess_parameters_from_data(self, data, update=False):
        """Override this method in a custom model class."""
        return self.parameters

    def evaluate(self, x):
        """
        Evaluate the model at each point in the input vector x and return 
        the result vector. This method should be overwritten in all models.
        """ 
        return [ self.evaluate_point(xi) for xi in x ] 

    def evaluate_point(self, x): 
        """
        Evaluate the model at point x, with model parameters as passed 
        to the function. 
        """
        return x
   
    def set_parameters(self, parameters):
        """Helper function to assign this model function object's parameters."""
        self.parameters = [ float(x) for x in parameters ]

    def __repr__(self):
        d = {'model': self.__class__.__name__, 'parameters': self.parameters} 
        return d 


class GammaModel(BaseModel): 
    """ 
    This is the gamma distribution function and associated unilities for evaluating and fitting.
    Expects that the model function parameter list is 
    [x0 (start of curve), alpha, beta, a0 (normalization coefficient), y0 (constant offset) ]. 
    """
    def __init__(self):
        """
        Construct a GammaModel model object. The model object has a parameters vector that is 
        specific to the functional form defined in the evaluate() method. We guess this vector 
        at the time of initialization and let the upstream code use the getters and setters 
        to override with better values. 
        """
        # placeholders for the fit results
        self.covariance = None
        self.parameters = [ 1.0, 1.0, 1.0, 1.0, 1.0 ] 

    def guess_parameters_from_data(self, data, update=False):
        """
        Use the pre-existing Data object to make an educated guess about the model parameters 
        based on the full data (ignore any fit windowing). Returns the resulting list of 
        guessed parameters. If update=True, the guess values overwrite the current 
        internal parameter values. 
        """
        # moderately-informed parameter initializations 
        #TODO: test other approaches  
        x0 = data.scaled_x[ np.argmax( data.scaled_y ) - 2 ] 
        alpha = 0.5
        beta = 0.5
        a0 = 0.01*max(data.scaled_y) 
        y0 = data.scaled_y[0] 

        parameter_guess = [ x0, alpha, beta, a0, y0 ] 
        
        # optionally overwrite the internal version with new guess 
        if update:
            self.set_parameters( parameter_guess )
        return parameter_guess 
      
    def evaluate(self, x, _x0, _alpha, _beta, _a0, _y0, remove_offset=False): 
        """
        Takes a vector of input values x and parameter values of the model, returns 
        the corresponding result vector, evaluated at each input value. This version has 
        been refactored to match the alpha/beta parametrization. 

        x: input vector 
        _x0: start of pulse
        _alpha: shape parameter
        _beta: rate parameter
        _a0: scaling coefficient
        _y0: constant background 
        """
        result = [ self.evaluate_point(xi, _x0, _alpha, _beta, _a0, _y0, remove_offset) for xi in x ]
        return result 
        
            
    def evaluate_point(self, x, _x0, _alpha, _beta, _a0, _y0, remove_offset=False): 
        """
        Evaluate the model at point x, with model parameters as passed 
        to the function.
    
        setting remove_offset to True subtracts the current value of _y0 from the
        returned value (for use with e.g. root finders). 
        """
        # use this return value to kludge a form of bounded optimization
        BIG_VAL = 1e12
        # start by assuming this point value is the current offset, then update 
        value = _y0 

        # kludge constrained optimization for use with scipy.curve_fit()
        #   - this will return a very value (leading to large SSQ error) when asked to 
        #       evaluate a single point with these ranges of parameters 
        if _a0 < 0 or _y0 < 0 or _alpha < 0 or _beta < 0 or 1 < _x0 < 0:
            return BIG_VAL 
        # now we're at least in the desired region of parameter space
        if x > _x0: 
            # the difference from the offset is the function argument 
            x_diff = x - _x0
            # dependent value = offset + scaling coefficient * gamma pdf 
            value = _y0 \
                    + _a0 * _beta**_alpha \
                    * x_diff**( _alpha - 1.0 ) / scipy.special.gamma( _alpha ) \
                    * math.exp( -_beta*x_diff ) 
        # in case we're using this function in a root-finding algorithm, normalize by the offset 
        if remove_offset:
            # subtract a tiny number (of counts) so the curve crosses zero 
            #   - assumes that we're ok with being ~1% off in total counts
            #   - TODO: test with other approaches 
            value = value - (1.01*_y0)

        return value 
       

    def __repr__(self):
        """Return the current state of the internal model parameters."""
        x0, alpha, beta, a0, y0 = self.parameters
        res = "# -- model parameters (for scaled input) \n" 
        res +=  cleandoc(
                    """
                    #  x0={:.4f}
                    #  alpha={:.4f} 
                    #  beta={:.4f} 
                    #  a0={:.4f} 
                    #  y0={:.4f}\n
                    """.format(x0, alpha, beta, a0, y0) 
                )
        return res

