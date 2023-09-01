from averages.dba import dba

def shapedba(tseries, max_iter=10, verbose=False, init_avg_method='approx_medoid',
             init_avg_series=None, weights=None, tseries_pad=None):
    return dba(tseries, max_iter=max_iter, verbose=verbose, init_avg_method=init_avg_method,
               init_avg_series=init_avg_series, distance_algorithm='shapedtw', weights=weights, tseries_pad=tseries_pad)
