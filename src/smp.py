#!/usr/bin/env python
from __future__ import print_function

__author__="Scott Hendrickson, Josh Montague"
__email__="shendrickson@twitter.com"
__license__="https://opensource.org/licenses/MIT"


import csv
from datetime import datetime
from inspect import cleandoc
import fileinput
from math import ceil
from math import floor 
import operator
import optparse
import re
import sys

import numpy as np
from scipy import integrate, special 
from scipy import optimize as sp_optimize 

import models 


class Data(object):
    """
    The Data class manages all aspects of the data to be used for fitting. 
    An instance of the Data object stores the raw data, a scaling factor to normalize the 
    input vectors to a range of 0-1.0, and (for now), the scaled vectors of floats as 
    np.array objects 
    """
    def __init__(self, x_col=1, y_col=2, window=None):
        """Initialize internal data structures needed for fitting."""
        # assign data columns. cmd-line column indices are 1-based 
        #   (for compatibility with eg cut), so convert to 0-based indices 
        self.indep_col = x_col - 1 
        self.dep_col = y_col - 1 
        # for datetime parsing
        self.time_re = re.compile("[0-9]{4}-[0-9]{2}-[0-9]{2}.[0-9]{2}:[0-9]{2}:[0-9]{2}")
        # Activity Streams format
        self.time_format = "%Y-%m-%dT%H:%M:%S"

        # raw input data lists (mutability now, np.array later)
        self.raw_x = [] 
        self.raw_y = [] 

        # scaled input data 
        self.scaled_x = np.array([]) 
        self.scaled_y = np.array([]) 

        # read data into object attributes via csv.reader & fileinput 
        self.read_input()

        # scale input data, calculate & store scaling factors
        self.initialize_scaled_data()
    
        # start by assuming that we aren't windowing the data
        self.window_idx = (0, len(self.raw_x))
        # if needed, calculate the windowed indices for fitting 
        if window != None:
            # read the input and ensure it is a properly formatted list 
            exec("window = %s" % window)       
            assert isinstance(window, list), "Time window argument must be in form of list: [0.1,0.9]"
            assert all([ 0.0 <= x <= 1.0 for x in window]), "Time window values must be in the range [0.0, 1.0]"
            self.calculate_window_idx(window)

    def read_input(self):
        """Read input data from either stdin or file, store raw data, copy"""
        # threshold for text headers or bad data 
        MAX_ERR_COUNT = 5
        error_count = 0
        for row in csv.reader(fileinput.FileInput(args, openhook=fileinput.hook_compressed)):
            if error_count >= MAX_ERR_COUNT:
                sys.stderr.write("\n# Exiting. Maximum invalid input reached (count={}).".format(error_count)) 
                sys.stderr.write("\n# Check input data.")
                sys.exit() 
            try:
                # is this an epoch timestamp?
                x_val = float( row[self.indep_col] )
            except ValueError as e:
                # is this an Activity Streams (UTC) timestamp?
                row_time_match = self.time_re.search(row[self.indep_col])
                if row_time_match is None:
                    error_count += 1 
                    sys.stderr.write("# could not parse row: {} (errors: {}) \n".format(','.join(row), error_count)) 
                    # skip to next row
                    continue
                row_time = row_time_match.group(0)
                # parse ts, convert to epoch secs
                x_ts = datetime.strptime(row_time, self.time_format)
                x_val = (x_ts - datetime(1970, 1, 1)).total_seconds() 
            y_val = float( row[self.dep_col] )
            # build lists of raw data 
            self.raw_x.append(x_val)
            self.raw_y.append(y_val)
        # the length of the input array is useful for other steps
        self.x_array_size = len(self.raw_x)

    def initialize_scaled_data(self):
        """Normalize raw input data ranges to 0-1."""

        # flag for later verification 
        self.initialized_scale = True
        
        # parameters for rescaling
        self.scale_offset_x = 0.0
        self.scale_offset_y = 0.0
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0

        # convert to np.array for calculations
        self.raw_x = np.array( self.raw_x )
        self.raw_y = np.array( self.raw_y )

        # calculate data ranges 
        min_x = min( self.raw_x )
        max_x = max( self.raw_x )
        min_y = min( self.raw_y )
        max_y = max( self.raw_y )

        # determine input transformation values 
        self.scale_offset_x = min_x 
        self.scale_factor_x = 1.0 / ( max_x - min_x ) 
        self.scale_offset_y = min_y 
        self.scale_factor_y = 1.0 / ( max_y - min_y ) 

        # calculate scaled arrays 
        self.scaled_x = np.dot( self.scale_factor_x, self.raw_x - self.scale_offset_x )
        self.scaled_y = np.dot( self.scale_factor_y, self.raw_y - self.scale_offset_y )

    def scale_data(self, array, axis='x', make_larger=False):
        """Convert data array between original (raw) range of values and scaled 0-1 
        range. By default, scales from arbitrary (assumed to be larger) range to 
        0-1 scale. Use make_larger=True to scale from 0-1 to original range. 

        Parameters
        ----------
        array : array-like of numerics or numeric 
            The data array (or single value) to scale. 
        axis : string, optional
            The axis of data to which array corresponds. Values are 'x' and 'y'. 
        make_larger : boolean, optional
            Determines whether 'array' is scaled to larger (raw data) or 
            smaller (0-1, "scaled") values.

        Returns
        -------
        scaled_data : array-like of numerics or numeric  
            Array of values, scaled as guided by method arguments 
            (same return type as 'array'). 
        """ 
        if not self.initialized_scale:
            raise Exception('Cannot call scale_data() without having first used initialize_scaled_data()') 
        if axis == 'x':
            if make_larger:
                scaled_data = (array / self.scale_factor_x) + self.scale_offset_x 
                # raw x is always integer epoch seconds 
                scaled_data = np.floor(scaled_data)
            elif not make_larger:
                scaled_data = (array - self.scale_offset_x) * self.scale_factor_x 
            else: 
                raise Exception('Invalid "make_larger" argument. Must be True or False.') 
        elif axis =='y':
            if make_larger:
                scaled_data = (array / self.scale_factor_y) + self.scale_offset_y 
                # raw y is always integer counts 
                scaled_data = np.floor(scaled_data)
            elif not make_larger:
                scaled_data = (array - self.scale_offset_y) * self.scale_factor_y 
            else: 
                raise Exception('Invalid "make_larger" argument. Must be True or False.') 
        else:
            raise Exception('Invalid data axis. Must be "x" or "y".') 
        return scaled_data


    def calculate_window_idx(self, window):
        """
        Use the given window list (in scaled time) to find and store the 
        appropriate indices for slicing within the scaled_x data array. 
        """
        for i,x in enumerate(self.scaled_x):
            if x <= window[0]:
                start_idx = i
            if x <= window[1]:
                end_idx = i

        self.window_idx = (start_idx, end_idx)


    def get_data_to_fit(self): 
        """
        Return the appropriate independent and dependent data arrays, 
        possibly windowed to smaller span. 
        """
        start, end = self.window_idx

        windowed_data = ( 
                        self.scaled_x[start:end], 
                        #self.raw_y[start:end] 
                        self.scaled_y[start:end] 
                        )

        return windowed_data


    def __repr__(self):
        """
        Used mostly for debugging. Potentially convert to more useful format.
        """
        
        return cleandoc("""
                        raw_x: {}
                        scaled_x: {}
                        raw_y: {}
                        scaled_y: {}
                        x (min, max): {}
                        scaled_x (min, max): {}
                        y (min, max): {}
                        scaled_y (min, max): {}
                        self.window_idx: {} 
                        self.scale_factor_x: {} 
                        self.scale_factor_y: {} 
                        self.scale_offset_x: {} 
                        self.scale_offset_y: {} 
                        """.format( 
                                self.raw_x,
                                self.scaled_x,
                                self.raw_y,
                                self.scaled_y,
                                (min(self.raw_x), max(self.raw_x)),
                                (min(self.scaled_x), max(self.scaled_x)),
                                (min(self.raw_y), max(self.raw_y)),
                                (min(self.scaled_y), max(self.scaled_y)),
                                self.window_idx,
                                self.scale_factor_x,
                                self.scale_factor_y,
                                self.scale_offset_x,
                                self.scale_offset_y
                            )
                    )


class Fit(object):
    """
    The Fit object applies a curve fit based on a model function 
    (from a Model() object) to data (from a Data() object). 
    """
    def __init__(self, data_obj, model_obj): 
        """
        Creates an instance of type Fit. Requires a model object 
        defining the model function, and a data object containing 
        the input data on which to fit the model.
        """
        self.data = data_obj
        self.model = model_obj

    def optimize(self):
        """
        Applies the chosen optimization technique to the model's evaluate method and 
        the appropriate data structures.  
        """ 
        # use helper method to possibly narrow data by user-entered window 
        x, y = self.data.get_data_to_fit()
        # optimize based on model-specific evaluate() 
        optimal_params, covar = sp_optimize.curve_fit( 
                                                    self.model.evaluate,
                                                    x, 
                                                    y, 
                                                    self.model.parameters 
                                                    )

        # store the final fit parameters and covariance in the model object for JIT calculations
        self.model.parameters = optimal_params
        self.model.covariance = covar

        return optimal_params, covar

    def get_fit_data(self, kind=None):
        """
        Returns the input data and corresponding output data associated with the 
        model fit. Optionally, convert the output format between scaled and raw values. 

        Parameters
        ----------
        kind : string, optional
            The type and kind of output to return. If specified, possible values are 
            'real' and 'both'. Default (None) returns scaled values for in/out/fit. 

        Returns
        -------
        return_data : list 
            A list comprising the rows of requested output data. 
            
        """ 
        # this list of iterables contains the result rows (subsequently given to csvwriter) 
        return_data = []
        # choose how many digits to keep (after decimal point) 
        digits = 5
        # build a formater with that digit setting for use later
        float_formatter = '{}:.{}f{}'.format('{',digits,'}')

        # evaluate the current model on scaled x - needed for every case
        model_values = model.evaluate(self.data.scaled_x, *model.parameters) 

        if kind == 'real':
            ## options.real_time output: raw_x, raw_y, raw_fit 
            # scale data up to raw values 
            raw_values = self.data.scale_data(model_values, axis='y', make_larger=True) 
            # include a header row
            return_data.append( ['raw_x','raw_y','raw_fit'] )

            for rx,ry,rf in zip(self.data.raw_x, 
                                self.data.raw_y, 
                                raw_values ): 
                # in this case, outputs are all integers, no need to format floats 
                return_data.append( [rx,ry,rf] )
        elif kind == 'both':
            ## options.both_times output: raw_x, raw_y, raw_fit, scaled_x, scaled_y, scaled_fit 
            # nb: this output is very helpful for debugging, but is hidden from 
            #   the command-line options (invoke via '-b')
            # scale data up to raw values 
            raw_values = self.data.scale_data(model_values, axis='y', make_larger=True) 
            # include a header row
            return_data.append( ['raw_x','raw_y','raw_fit','scaled_x','scaled_y','scaled_fit'] )

            for rx,ry,rf,sx,sy,sf in zip(self.data.raw_x, 
                                        self.data.raw_y, 
                                        raw_values,
                                        self.data.scaled_x, 
                                        self.data.scaled_y, 
                                        model_values): 
                # format floats before appending to the data list 
                return_data.append( [float_formatter.format(col) for col in (rx,ry,rf,sx,sy,sf)] ) 
        else:
            ## (default) only scaled output: scaled_x, scaled_y, scaled_fit
            # include a header row
            return_data.append( ['scaled_x','scaled_y','scaled_fit'] )
            for sx,sy,sf in zip(self.data.scaled_x, 
                                self.data.scaled_y,
                                model_values): 
                # format floats before appending to the data list 
                return_data.append( [float_formatter.format(col) for col in (sx,sy,sf)] ) 
        return return_data

    #####
    # helper methods to calculate derived parameters
    #####

    def get_rise_time(self):
        """ 
        The rise time of the SMP is the time duration between the onset of the 
        pulse and the maximum rate estimate. 

        If the peak predicted by the model is prior to the onset of the SMP (typically the 
        result of a very sharp rise), a rise time of zero is returned.  

        Returns
        -------
        rise_time: 2-tuple  
            Calculated rise time (scaled time), rise time (real time).
        """
        # unpack optimized model parameters
        x0, alpha, beta, a0, y0 = self.model.parameters
        # Eqn 4 
        t_rise_scaled = ( alpha - 1.0 ) / beta  
        if t_rise_scaled < 0: 
            t_rise_scaled = 0.0
        # de-scale time delta back up using appropriate scale factor 
        t_rise_real = t_rise_scaled / self.data.scale_factor_x

        rise_time = ( 
                    t_rise_scaled,
                    t_rise_real 
                    ) 

        return rise_time


    def get_balanced_time(self):
        """ 
        The balanced time of the SMP is a point that tries to strike a balance between the 
        tall head and long tail of the distribution. 

        The method returns a tuple of the calculated balanced time (in scaled time), 
        and the value of the model evaluated at these points. 

        Returns
        -------
        balanced_time: 3-tuple
            Calculated balanced time (scaled), calculated balanced time (real), value of model 
            evaluated at balanced time. 
        """
        # unpack optimized model parameters
        x0, alpha, beta, a0, y0 = self.model.parameters
        # Eqn 5 
        t_balanced_scaled = float(alpha) / beta + x0
        # invert time transformation
        #t_balanced_real = float(t_balanced_scaled) / self.data.scale_factor_x + self.data.scale_offset_x 
        t_balanced_real = self.data.scale_data(t_balanced_scaled, 
                                                axis='x', 
                                                make_larger=True) 
        balanced_time = ( 
                        t_balanced_scaled,
                        t_balanced_real,
                        self.model.evaluate_point( t_balanced_scaled, *self.model.parameters ) 
                        )
        return balanced_time 
    

    def get_event_volume(self):
        """ 
        Calculate an estimate for the total count of activities solely due to the SMP model. 

        Returns
        -------
        totals: 2-tuple 
            Sum of model counts evaluation over the duration of the event with the background 
            (baseline), and same but without the baseline included. 
        """
        # unpack optimized model parameters
        x0, alpha, beta, a0, y0 = self.model.parameters

        x_root = sp_optimize.brentq(self.model.evaluate_point, 
                                    1.1*x0, 
                                    10.0, 
                                    args=(x0, alpha, beta, a0, y0, True)
                                    )

        # create the x values, step size is same as original 
        x_array_step = self.data.scaled_x[1] - self.data.scaled_x[0]
        x_array = np.arange(x0, x_root, x_array_step)
        # create array of point-by-point (scaled) model evaluations (w/ + w/o baselines) 
        point_evals_with_baseline = self.model.evaluate(x_array, x0, alpha, beta, a0, y0)
        point_evals_without_baseline = self.model.evaluate(x_array, x0, alpha, beta, a0, y0, remove_offset=True)
        
        # de-scale the y values to actual counts 
        baseline_counts = sum( self.data.scale_data(point_evals_with_baseline, 
                                                    axis='y', 
                                                    make_larger=True) ) 
        no_baseline_counts = sum( self.data.scale_data(point_evals_without_baseline, 
                                                        axis='y', 
                                                        make_larger=True) ) 
    
        # combine the two for return 
        totals = (
                    baseline_counts, 
                    no_baseline_counts, 
                    )

        return totals 

    def __repr__(self):
        """
        Returns a string representation of the Fit object, including the derived parameters 
        using the information in the Data and Model objects. 
        """
        # unpack optimized model parameters
        x0, alpha, beta, a0, y0 = self.model.parameters

        # seed the result so it can be built arbitrarily 
        result = "\n"
        
        ## model fit parameters
        result += '{}'.format(self.model)

        # observed values
        # TODO: peak value, time of peak value 

        ## derived values 
        result += "\n# -- derived parameters" 

        # rise time
        scaled, real = self.get_rise_time()
        result += "\n# rise time "
        result += "\n# - t_rise (scaled) = {}".format(scaled) 
        result += "\n# - t_rise (real) = {}".format(real) 
        if scaled == 0.0:
            result += " [* t_rise = 0 indicates calculated rise time is smaller than data resolution *] " 

        # balanced time
        t_bal_scaled, t_bal_real, f_t_bal_scaled = self.get_balanced_time() 
        result += "\n# balanced time "
        result += "\n# - t_bal (scaled) = {:.5f}".format(t_bal_scaled)
        result += "\n# - t_bal (real) = {:.0f}".format(t_bal_real) 
        result += "\n# - f(t_bal) = {}".format( int(ceil(f_t_bal_scaled)) ) 

        # total event volume
        vol_with_baseline, vol_no_baseline = self.get_event_volume() 
        result += "\n# event volume "
        result += "\n# - V_event (no baseline) = {:.0f}".format(vol_no_baseline)
        result += "\n# - V_event (+ baseline) = {:.0f}".format(vol_with_baseline)

        return result


def fit_args():
    """
    This top-level function returns an opt parser in the event that this module 
    is run from the command-line.
    """
    parser = optparse.OptionParser()

    parser.add_option("-i", "--column-independent", dest="i_col", default=1, type="int",
            help="Column of independent variable for fit, 1-based count (default: %default)")
    parser.add_option("-d", "--column-dependent",  dest="d_col", default=2, type="int",
            help="Column of dependent variable for fit, 1-based count (default: %default)")
    parser.add_option("-p", "--init-parameters", dest="init_parameters", default=None,
            help="Initial guess of model function parameters e.g. '[x0, alpha, beta, a0, y0]' (use single-quotes). If not given, initial values are calculated from input data (default: %default)")
    parser.add_option("-w", "--fitting-window", dest="window", default=None, 
            help="Define a window of 0-1-scaled x values over which to fit the curve e.g. '[0.1,0.7]' (use single-quotes). If not given, entire input range is used (not recommended). (default: %default)")
    parser.add_option("-t", "--real-time", dest="real_time", default=False, action='store_true', 
            help="Return output with time data in actual time [epoch seconds, instead of 0-1 scaled] (default: %default)")
    parser.add_option("-b", "--both-times", dest="both_times", action='store_true', 
            help=optparse.SUPPRESS_HELP)

    return parser



if __name__ == "__main__":

    # read any cmd-line options 
    (options, args) = fit_args().parse_args() 

    # create the data object (reads from stdin)
    data = Data( options.i_col, options.d_col, options.window )

    # create the model object optionally using cmd-line arguments 
    model = models.GammaModel() 

    if options.init_parameters is not None:
        # convert optparser string to list 
        exec("init_parameters_list = %s"%options.init_parameters)
        assert( type(init_parameters_list) == list), 'init-parameters must be in list-like form: [4.2,5.1]'
        try:
            model.parameters = [ float(x) for x in init_parameters_list ]
        except ValueError as e:
            sys.stderr.write('Invalid parameter type. Models require numeric parameter values.\n') 
            sys.exit()
    else:
        # update internal parameters
        _ = model.guess_parameters_from_data(data, update=True) 

    # create the fit object 
    fit = Fit(data_obj=data, model_obj=model)
    
    # optimize based on the model and data 
    # - updates the state of the model object, returns final parameters and covariance matrix 
    _op, _cov = fit.optimize()

    # select the appropriate output format 
    if options.real_time:
        output_data = fit.get_fit_data(kind='real') 
    elif options.both_times: 
        output_data = fit.get_fit_data(kind='both') 
    else: 
        output_data = fit.get_fit_data() 

    # write the appropriate data to a csv on stdout 
    writer = csv.writer(sys.stdout, delimiter=',')
    writer.writerows(output_data)

    # finally, write the model repr (parameters+)
    # but, send to stderr to avoid contaminating a data file 
    sys.stderr.write( "{}\n".format( fit ) )

