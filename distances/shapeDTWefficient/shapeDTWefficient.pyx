# cython: profile=True

from __future__ import division
import numpy as np
cimport numpy as np
from libc.float cimport DBL_MAX

cdef inline int min_c_int(int a, int b): return a if a <= b else b
cdef inline int max_c_int(int a, int b): return a if a >= b else b

__author__ ="Francois Petitjean"

def shape_DTW(np.ndarray[double, ndim=1] s,np.ndarray[double, ndim=1] t,
              cost_mat, cost_mat_dtw, delta_mat, delta_mat_padded,
              shape_reach=0, w=1.0):
    cdef:
        double resu

    resu = np.sqrt(squared_shape_DTW(s,t,shape_reach,cost_mat,cost_mat_dtw,delta_mat,delta_mat_padded,w))
    return resu,np.pad(cost_mat,((1,0),(1,0)),mode='maximum')

cdef double squared_shape_DTW(double[:] s,double[:] t,int shape_reach,double[:,:] cost_mat,double[:,:] cost_mat_dtw,double[:,:] delta_mat,double[:,:] delta_mat_padded,double w):
    cdef:
       int reach2x = shape_reach*2
       int s_len = len(s)-reach2x
       int t_len = len(t)-reach2x
       int i,j,jstart,jstop,idx_inf_left,im,jm
       double res
       double[:,:] delta_mat_dtw
       int w_
    ''' We need 2 delta matrices and 2 accumulated cost matrices.
    This is because shapeDTW calculates the cost for the path
    using the temporal neighborhood, but then uses the 'normal' DTW matrix to
    calculate the cost. Here we do it all in one path so we don't have to
    retrace the path at the end (I think it's faster, even if more operations)
    '''
    fill_delta_mat_shape_dtw(s, t, shape_reach, delta_mat,delta_mat_padded)
    # 'normal' DTW matrix is in the middle of the padded one
    delta_mat_dtw = delta_mat_padded[shape_reach:shape_reach+s_len,shape_reach:shape_reach+t_len]

    cost_mat[0, 0] = delta_mat[0, 0]
    cost_mat_dtw[0, 0] = delta_mat_dtw[0, 0]

    w_ = int(w*s_len)

    for i in range(1, s_len):
       cost_mat[i, 0] = cost_mat[i-1, 0]+delta_mat[i, 0]
       cost_mat_dtw[i, 0] = cost_mat_dtw[i-1, 0]+delta_mat_dtw[i, 0]

    for j in range(1, t_len):
       cost_mat[0, j] = cost_mat[0, j-1]+delta_mat[0, j]
       cost_mat_dtw[0, j] = cost_mat_dtw[0, j-1]+delta_mat_dtw[0, j]

    for i in range(1, s_len):
       jstart = max_c_int(1 , i-w_)
       jstop = min_c_int(t_len , i+w_+1)
       idx_inf_left = i-w_-1
       if idx_inf_left >= 0:
           cost_mat[i,idx_inf_left] = DBL_MAX
       for j in range(jstart, jstop):
           im = i-1
           jm = j-1
           diag,left,top =cost_mat[im, jm], cost_mat[i, jm], cost_mat[im, j]

           if(diag <=left):
               if(diag<=top):
                   res = diag
                   res_dtw = cost_mat_dtw[im, jm]
               else:
                   res = top
                   res_dtw = cost_mat_dtw[im, j]
           else:
               if(left<=top):
                   res = left
                   res_dtw = cost_mat_dtw[i, jm]
               else:
                   res = top
                   res_dtw = cost_mat_dtw[im, j]

           cost_mat[i, j] = res + delta_mat[i, j]
           cost_mat_dtw[i, j] = res_dtw + delta_mat_dtw[i,j]
           if jstop < t_len:
               cost_mat[i][jstop] = DBL_MAX
    return cost_mat_dtw[s_len-1,t_len-1]

cdef void fill_delta_mat_dtw(double[:]center, double[:]s, double[:,:] delta_mat):
    delta_numpy = np.asarray(delta_mat)
    np.subtract.outer(np.asarray(center), np.asarray(s), out=delta_numpy)
    np.square(delta_numpy, out=delta_numpy)

cdef void fill_delta_mat_shape_dtw(double[:] center, double[:] s,int reach,double[:,:] delta_mat,double[:,:] delta_mat_padded):
    #Ok, quite a lot of thoughts went into this, be careful modifying!
    cdef:
        int length = len(center)-reach*2
        double[:] padded_center = np.asarray(center)
        double[:] padded_s = np.asarray(s)
        # double[:] padded_center = np.pad(np.asarray(center),(reach,reach),mode='edge')
        # double[:] padded_s = np.pad(np.asarray(s),(reach,reach),mode='edge')
        int w,start,stop, w_stop=2*reach+1
        double[:,:] slice_for_window
    fill_delta_mat_dtw(padded_center,padded_s,delta_mat_padded)
    delta_mat_numpy = np.asarray(delta_mat)
    #have all the numbers calculated, now just need to add the right things together
    delta_mat_numpy.fill(0.0)
    for w in range(w_stop):
        start= w
        stop = w+length
        slice_for_window = delta_mat_padded[start:stop,start:stop]
        #assert(delta_mat.shape[0] == slice_for_window.shape[0] and delta_mat.shape[1] == slice_for_window.shape[1] )
        np.add(delta_mat_numpy,slice_for_window,out=delta_mat_numpy)
