# these are the different descriptors to be used as a function neighborhood
import numpy as np 

def identity(subsequence): 
    """
    This function returns the identity function of the given 1-D subsequence 
    :param subsequence: The univariate time series 
    """
    # return a copy of the given subsequence 
    return np.copy(subsequence)
    
def paa(subsequence, num_intervals = 5): 
    """
    This function applies Piecewise Aggregate Approximation on the 1-D 
    subsequence ( num_intervals should be a divider of len(subsequence) )
    :param subsequence: The univariate time series 
    :param num_intervals: The new length of the time series 
    """
    # get the number of time steps of the subsequence 
    n = len(subsequence)
    # window size 
    w = int(n/num_intervals)
    # init the new subsequence to return 
    res_paa = np.zeros((num_intervals),dtype=np.float64)
    # loop through each new data point 
    for i in range(num_intervals):
        # average the window to get one value 
        res_paa[i] = np.mean(subsequence[i*w:(i+1)*w])
    # return the resulting new reduced subsequence 
    return res_paa

def slope(subsequence, num_intervals = 5, xscale = 0.1): 
    """
    This function divides the subsequence in num_intervals equal-length intervals
    Each interval is then fitted with a line using Total Least Square
    The slope of each line is claculated thus the resulting transformed subsequence 
    has a length equal to num_intervals
    :param subsequence: The univariate time series
    :param num_intervals: The new length of the time series
    """
    # get the number of time steps of the subsequence 
    n = len(subsequence)
    # window size 
    w = int(n/num_intervals)
    # init the new subsequence to return 
    res = np.zeros((num_intervals),dtype=np.float64)
    # loop through each new data point 
    for i in range(num_intervals):
        # get the interval value 
        val = subsequence[i*w:(i+1)*w]
        # create the points to be fitted 
        x = np.arange(len(val))
        # scale them 
        x = x*xscale
        # fit the line y = a x + b
        fitted_line = np.polyfit(x, val, 1)
        # get the slope 
        res[i] = fitted_line[0]
    return res

def derivative(subsequence): 
    """
    This function returns the first order derivative of the subsequence 
    :param subsequence: The univariate time series 
    """
    # get the number of time steps of the subsequence
    n = len(subsequence)
    # define the result 
    res = np.zeros((n-2),dtype=np.float64)
    # loop through each time step and calculate the derivative 
    # note that the derivative of the first and last elements are not computed 
    for i in range (1,n-1): 
        # calculate the derivative according to "Derivative Dynamic Time Warping" SDM
        res[i-1] = ((subsequence[i]-subsequence[i-1])+((subsequence[i+1]-subsequence[i-1])/2))/2
    return res 