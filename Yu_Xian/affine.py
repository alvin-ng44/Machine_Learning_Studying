import numpy as np
from osgeo import gdal, osr

# GCP is of the form (X_geo, Y_geo, 0, X_pixel, Y_pixel)

# For our purposes here, we will do (X_geo, Y_geo, X_pixel, Y_pixel) instead

# This is what CRS_compute_georef_equations does
def solve_lss_from_gcps(gcps_lst):

    # solves pixel to georef first
    # matrix A12 is from pixel to georef
    A12 = np.array([[1, px, py] for (X, Y, px, py) in gcps_lst])
    bX12 = np.array([X for (X, Y, px, py) in gcps_lst])
    bY12 = np.array([Y for (X, Y, px, py) in gcps_lst])

    E12, *_ = np.linalg.lstsq(A12, bX12, rcond=None)
    N12, *_ = np.linalg.lstsq(A12, bY12, rcond=None)

    # solves georef to pixel
    A21 = np.array([[1, X, Y] for (X, Y, px, py) in gcps_lst])
    bX21 = np.array([px for (X, Y, px, py) in gcps_lst])
    bY21 = np.array([py for (X, Y, px, py) in gcps_lst])

    E21, *_ = np.linalg.lstsq(A21, bX21, rcond=None)
    N21, *_ = np.linalg.lstsq(A21, bY21, rcond=None)
    
    return E12, N12, E21, N21
    
# This is what CRS_georef does
def transform(coords, E, N):
    x, y = coords[0], coords[1]

    X = E[0] + E[1] * x + E[2] * y
    Y = N[0] + N[1] * x + N[2] * y
    return X, Y

def py_worst_outlier(gcps_lst, E, N, dfTolerance):

    worst_index = -1
    worst_residual = -1.0

    for i, gcp in enumerate(gcps_lst):
        X, Y, px, py = gcp

        X_pred, Y_pred = transform((px, py), E, N)
        X_res = (X_pred - X)
        Y_res = (Y_pred - Y)
    
        Res = np.sqrt((X_res)**2 + (Y_res)**2)
        if (Res >= dfTolerance) and (Res > worst_residual):
            worst_residual = Res
            worst_index = i

    if worst_index == -1:
        return "No outliers"
    else:
        return worst_index

def py_rm_outliers(gcps_lst, dfTolerance, nGCPmin = -1):

    gcps_lst_cp = list(gcps_lst)
    E12, N12, E21, N21 = solve_lss_from_gcps(gcps_lst_cp)

    while len(gcps_lst_cp) > nGCPmin:
        worst_index = py_worst_outlier(gcps_lst_cp, E12, N12, dfTolerance)
        if worst_index == "No outliers":
            break
        else:
            del gcps_lst_cp[worst_index]
            E12, N12, E21, N21 = solve_lss_from_gcps(gcps_lst_cp)

    return gcps_lst_cp, E12, N12, E21, N21

def py_GCPTransformer(gcps_lst, bRefine, dfTolerance, nGCPmin = -1):

    if (bRefine == True):
        if nGCPmin == -1:
            nGCPmin = 4
            if len(gcps_lst) < nGCPmin:
                return "Need 4 or more GCPs for bRefine to be True"
        else:
            if nGCPmin != -1: 
                gcps_lst_cp, E12, N12, E21, N21 = py_rm_outliers(gcps_lst, dfTolerance, nGCPmin)
                return gcps_lst_cp, E12, N12, E21, N21
    else:
        E12, N12, E21, N21 = solve_lss_from_gcps(gcps_lst)
        return gcps_lst, E12, N12, E21, N21

# def py_warp_destoutput(tif_file, py_GCPTransformer):

#     src_ds = gdal.Open(tif_file)
#     width, height = src_ds.RasterXSize, src_ds.RasterYSize

#     N_pixelstep = 50
#     nsteps = int( (min(width, height) / N_pixelstep) + 0.5)
#     if nsteps < 20:
#         nsteps = 20
#     elif nsteps > 100:
#         nsteps = 100

#     success_flag = None

#     while nsteps:
        
        