from tslearn.metrics import soft_dtw

def softdtw(x,y,gamma=0.01, max_iter=100):
    """
    This function returns the soft-dtw to compute the distance between two time series using the tslearn package
    """
    # simply returns dtw with warping window equal to zero
    return soft_dtw(x,y,gamma=gamma),None