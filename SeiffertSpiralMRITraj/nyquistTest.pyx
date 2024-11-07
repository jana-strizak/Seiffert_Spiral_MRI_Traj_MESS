import numpy as np
cimport numpy as np

np.import_array()

ctypedef np.float32_t DTYPE_f
ctypedef np.int8_t DTYPE_i

# Define a function that operates on the 3D numpy array
def process_3d_array(np.ndarray[DTYPE_f, ndim = 4] gridPoints, np.ndarray nyquistPoints, np.ndarray[DTYPE_i, ndim = 3] dsphere, double dk_nyquist):
    
    # define size integers
    cdef Py_ssize_t i, j, p, m, k, i_n, j_n, p_n
    
    # inner array containing nyquist points
    cdef np.ndarray[DTYPE_f, ndim = 2] pointsNyquist_gp
    cdef np.ndarray[DTYPE_f, ndim=2] pointsNyquist_n
    
    # define points which will be populated by missing points once found
    #cdef np.ndarray[DTYPE_f, ndim = 2] requiredPoints
    cdef list requiredPoints = []
    cdef list checkedGridPoints = []
    
    # array to hold neighbouring indices
    cdef int possibleIndx[3][3]
    
    cdef np.ndarray[DTYPE_f, ndim = 1] d
    
    cdef bint nyquistResult
    
    # Loop through all gridPoints
    for i in range(1, gridPoints.shape[0] - 1):
        for j in range(1, gridPoints.shape[1] - 1):
            for p in range(1, gridPoints.shape[2] - 1):
                
                # don't check this point unless the value is 2 
                if dsphere[i, j, p] != 2:
                    continue
                
                # Access the inner 2D array, representing the nyquist points at this grid point
                pointsNyquist_gp = nyquistPoints[i, j, p]  
                
                checkedGridPoints.append(pointsNyquist_gp)
                
                # neighbour's indices 
                possibleIndx[0][0] = i + 1
                possibleIndx[0][1] = j
                possibleIndx[0][2] = p

                possibleIndx[1][0] = i
                possibleIndx[1][1] = j + 1
                possibleIndx[1][2] = p

                possibleIndx[2][0] = i
                possibleIndx[2][1] = j
                possibleIndx[2][2] = p + 1
                
                # itterate through neighbours 
                for m in range(3):
                    i_n = possibleIndx[m][0]
                    j_n = possibleIndx[m][1]
                    p_n = possibleIndx[m][2]
                    
                    if (dsphere[i_n, j_n, p_n] > 0):# this grid point is in the sphere and can be checked 
        
                        # acess nyquist points inside this neighbour's index 
                        pointsNyquist_n = nyquistPoints[i_n, j_n, p_n]  
                        
                        nyquistResult = False
                        
                        # compare with neighbouring points 
                        for k in range(pointsNyquist_gp.shape[0]):
                            # Calculate distance between the points using NumPy's norm function
                            d = np.linalg.norm(pointsNyquist_gp[k, :] - pointsNyquist_n, axis = 1)
                            
                            # Check condition using NumPy's any function
                            if np.any(d <= dk_nyquist): # passes
                                nyquistResult = True
                                break # don't check other points 
                                
                        # if it didn't pass up until now, it's a fail
                        if not(nyquistResult):
                            # nyquist condition not met
                            gridPoint_coord = np.squeeze(gridPoints[i, j, p, :])
                            n_coord = np.squeeze(gridPoints[i_n, j_n, p_n, :])
                            midPoint = np.mean(np.stack([gridPoint_coord, n_coord]), axis = 0)
                       
                            requiredPoints.append(list(midPoint))
    #return requiredPoints
    return requiredPoints
    #cdef int result = 1
    #return result
 