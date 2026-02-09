import os
import shutil
import tempfile
import numpy as np
from osgeo import gdal, osr
from numba import vectorize, float64

# GCP is of the form (X_geo, Y_geo, 0, X_pixel, Y_pixel)

# For our purposes here, we will do (X_geo, Y_geo, X_pixel, Y_pixel) instead
# te is (xmin, ymin, xmax, ymax)
# tr is (pixel_width, pixel_height)

# ============================================================================
# VECTORIZED TRANSFORMATION FUNCTIONS
# ============================================================================

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def _transform_forward_x(x, y, E0, E1, E2, x_mean, y_mean):
    """Vectorized forward X transformation: pixel -> geo (with mean-centering)"""
    return E0 + E1 * (x - x_mean) + E2 * (y - y_mean)

@vectorize([float64(float64, float64, float64, float64, float64, float64, float64)])
def _transform_forward_y(x, y, N0, N1, N2, x_mean, y_mean):
    """Vectorized forward Y transformation: pixel -> geo (with mean-centering)"""
    return N0 + N1 * (x - x_mean) + N2 * (y - y_mean)

@vectorize([float64(float64, float64, float64, float64, float64)])
def _transform_simple_x(x, y, E0, E1, E2):
    """Vectorized simple X transformation (no mean-centering)"""
    return E0 + E1 * x + E2 * y

@vectorize([float64(float64, float64, float64, float64, float64)])
def _transform_simple_y(x, y, N0, N1, N2):
    """Vectorized simple Y transformation (no mean-centering)"""
    return N0 + N1 * x + N2 * y

# ============================================================================
# GCP SOLVER
# ============================================================================
# This is what CRS_compute_georef_equations does
def solve_lss_from_gcps(gcps_lst):
    """
    Computes forward (px->geo) and inverse (geo->px) affine transforms using 
    mean-centering for numerical stability, matching GDAL's CRS_compute_georef_equations.
    """
    # Extract coordinates
    X_geo = np.array([g[0] for g in gcps_lst])
    Y_geo = np.array([g[1] for g in gcps_lst])
    px = np.array([g[2] for g in gcps_lst])
    py = np.array([g[3] for g in gcps_lst])

    # Compute means (required for stable LSS and subsequent transforms)
    px_mean, py_mean = np.mean(px), np.mean(py)
    X_mean, Y_mean = np.mean(X_geo), np.mean(Y_geo)

    # Solve px -> geo (Forward)
    # Formula: GeoX = E[0] + E[1]*(px - px_mean) + E[2]*(py - py_mean)
    A_fwd = np.column_stack([np.ones(len(gcps_lst)), px - px_mean, py - py_mean])
    E12, _, _, _ = np.linalg.lstsq(A_fwd, X_geo, rcond=None)
    N12, _, _, _ = np.linalg.lstsq(A_fwd, Y_geo, rcond=None)

    # Solve geo -> px (Inverse)
    # Formula: px = E[0] + E[1]*(GeoX - X_mean) + E[2]*(GeoY - Y_mean)
    A_inv = np.column_stack([np.ones(len(gcps_lst)), X_geo - X_mean, Y_geo - Y_mean])
    E21, _, _, _ = np.linalg.lstsq(A_inv, px, rcond=None)
    N21, _, _, _ = np.linalg.lstsq(A_inv, py, rcond=None)

    return E12, N12, E21, N21, px_mean, py_mean, X_mean, Y_mean

def transform_with_means(x, y, E, N, x_mean, y_mean):
    """Applies affine coefficients using mean-centering (now using vectorized functions)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    X = _transform_forward_x(x, y, E[0], E[1], E[2], x_mean, y_mean)
    Y = _transform_forward_y(x, y, N[0], N[1], N[2], x_mean, y_mean)
    return X, Y

def py_worst_outlier(gcps_lst, E12, N12, px_mean, py_mean, dfTolerance):
    """Finds the GCP with the highest residual exceeding tolerance (vectorized)."""
    # Extract coordinates for vectorized computation
    px_arr = np.array([g[2] for g in gcps_lst], dtype=np.float64)
    py_arr = np.array([g[3] for g in gcps_lst], dtype=np.float64)
    X_arr = np.array([g[0] for g in gcps_lst], dtype=np.float64)
    Y_arr = np.array([g[1] for g in gcps_lst], dtype=np.float64)

    # Vectorized prediction
    X_pred = _transform_forward_x(px_arr, py_arr, E12[0], E12[1], E12[2], px_mean, py_mean)
    Y_pred = _transform_forward_y(px_arr, py_arr, N12[0], N12[1], N12[2], px_mean, py_mean)
    
    # Compute residuals
    residuals = np.sqrt((X_pred - X_arr)**2 + (Y_pred - Y_arr)**2)
    
    # Find worst outlier
    mask = residuals >= dfTolerance
    if np.any(mask):
        worst_index = np.argmax(residuals)
        if residuals[worst_index] < dfTolerance:
            worst_index = -1
    else:
        worst_index = -1
    
    return worst_index

def py_rm_outliers(gcps_lst, dfTolerance, nGCPmin=4):
    """Iteratively removes outliers until residuals are within tolerance."""
    current_gcps = list(gcps_lst)
    
    while len(current_gcps) > nGCPmin:
        # Recompute equations for the current set
        E12, N12, E21, N21, px_m, py_m, X_m, Y_m = solve_lss_from_gcps(current_gcps)
        
        worst = py_worst_outlier(current_gcps, E12, N12, px_m, py_m, dfTolerance)
        if worst == -1:
            break
        del current_gcps[worst]

    # Final solve after all removals
    return solve_lss_from_gcps(current_gcps) + (current_gcps,)

# ============================================================================
# DESTINATION TRANSFORMER (Geotransform Logic)
# ============================================================================

def py_destTransformer(te, tr):
    """
    Standard North-Up Geotransform for the destination image.
    E12/N12: Pixel to Geo
    E21/N21: Geo to Pixel
    """
    xmin, ymin, xmax, ymax = te
    pw, ph = tr

    # Pixel -> Georef (Forward)
    E12 = [xmin, pw, 0]
    N12 = [ymax, 0, -ph]

    # Georef -> Pixel (Inverse)
    E21 = [-xmin / pw, 1.0 / pw, 0]
    N21 = [ymax / ph, 0, -1.0 / ph]

    return E12, N12, E21, N21

# ============================================================================
# GENIMGPROJ TRANSFORMER CLASS
# ============================================================================

class GenImgProjTransformer:
    def __init__(self, gcps_lst, dst_extent, dst_pixel_size, bRefine=False, dfTolerance=0.0, nGCPmin=-1):
        # 1. Source Side: GCP-based
        if bRefine:
            res = py_rm_outliers(gcps_lst, dfTolerance, nGCPmin)
            E12, N12, E21, N21, px_m, py_m, X_m, Y_m, final_gcps = res
            self.gcps_lst = final_gcps
        else:
            E12, N12, E21, N21, px_m, py_m, X_m, Y_m = solve_lss_from_gcps(gcps_lst)
            self.gcps_lst = gcps_lst

        self.E12_src, self.N12_src = E12, N12
        self.E21_src, self.N21_src = E21, N21
        self.px_m_src, self.py_m_src = px_m, py_m
        self.X_m_src, self.Y_m_src = X_m, Y_m

        # 2. Destination Side: Geotransform-based
        self.E12_dst, self.N12_dst, self.E21_dst, self.N21_dst = py_destTransformer(dst_extent, dst_pixel_size)

    def transform(self, x, y, bDstToSrc=True):
        """
        The core pixel-to-pixel transformation (now vectorized).
        Warping usually uses bDstToSrc=True (Dest Pixels -> Source Pixels).
        Automatically handles scalars or arrays.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if bDstToSrc:
            # Step A: Dest Pixel -> Dest Georef (No mean-centering for standard GT)
            geo_x = _transform_simple_x(x, y, self.E12_dst[0], self.E12_dst[1], self.E12_dst[2])
            geo_y = _transform_simple_y(x, y, self.N12_dst[0], self.N12_dst[1], self.N12_dst[2])

            # Step B: Source Georef -> Source Pixel (With mean-centering)
            src_px = _transform_forward_x(geo_x, geo_y, self.E21_src[0], self.E21_src[1], 
                                         self.E21_src[2], self.X_m_src, self.Y_m_src)
            src_py = _transform_forward_y(geo_x, geo_y, self.N21_src[0], self.N21_src[1], 
                                         self.N21_src[2], self.X_m_src, self.Y_m_src)
            return src_px, src_py
        else:
            # Step A: Source Pixel -> Source Georef (With mean-centering)
            geo_x = _transform_forward_x(x, y, self.E12_src[0], self.E12_src[1], 
                                        self.E12_src[2], self.px_m_src, self.py_m_src)
            geo_y = _transform_forward_y(x, y, self.N12_src[0], self.N12_src[1], 
                                        self.N12_src[2], self.px_m_src, self.py_m_src)

            # Step B: Dest Georef -> Dest Pixel (No mean-centering)
            dst_px = _transform_simple_x(geo_x, geo_y, self.E21_dst[0], self.E21_dst[1], self.E21_dst[2])
            dst_py = _transform_simple_y(geo_x, geo_y, self.N21_dst[0], self.N21_dst[1], self.N21_dst[2])
            return dst_px, dst_py

    # Convenience Methods for your debug tests
    def src_pixel_to_src_georef(self, px, py):
        return transform_with_means(px, py, self.E12_src, self.N12_src, self.px_m_src, self.py_m_src)

    def src_georef_to_src_pixel(self, X, Y):
        return transform_with_means(X, Y, self.E21_src, self.N21_src, self.X_m_src, self.Y_m_src)

# ============================================================================
# WARPING / RESAMPLING FUNCTIONS
# ============================================================================

def warp_image_nearest(src_dataset, transformer, dst_extent, dst_pixel_size, 
                      bands=None, nodata_value=None, debug=False):
    """
    Warp/resample a source image to a destination grid using nearest neighbor resampling.
    This follows GDAL's approach: for each destination pixel, transform to source coordinates,
    then sample the nearest source pixel.
    
    Parameters:
    -----------
    src_dataset : gdal.Dataset
        Source GDAL dataset to warp from
    transformer : GenImgProjTransformer
        Initialized transformer with GCPs and destination parameters
    dst_extent : tuple
        Destination extent as (xmin, ymin, xmax, ymax)
    dst_pixel_size : tuple
        Destination pixel size as (pixel_width, pixel_height)
    bands : list or None
        List of band numbers (1-indexed) to warp. If None, warps all bands.
    nodata_value : float or None
        Value to use for pixels that fall outside source image bounds
    debug : bool
        If True, print diagnostic information about coordinate ranges and valid pixels
    
    Returns:
    --------
    dst_array : numpy.ndarray
        Warped output array of shape (nBands, nDstHeight, nDstWidth)
    """
    xmin, ymin, xmax, ymax = dst_extent
    pixel_width, pixel_height = dst_pixel_size
    
    # Calculate destination image dimensions
    nDstWidth = int(np.round((xmax - xmin) / pixel_width))
    nDstHeight = int(np.round((ymax - ymin) / pixel_height))
    
    # Determine which bands to process
    if bands is None:
        bands = list(range(1, src_dataset.RasterCount + 1))
    nBands = len(bands)
    
    # Get source dimensions
    nSrcWidth = src_dataset.RasterXSize
    nSrcHeight = src_dataset.RasterYSize
    
    if debug:
        print(f"Destination grid: {nDstWidth}x{nDstHeight} pixels")
        print(f"  Extent: ({xmin}, {ymin}, {xmax}, {ymax})")
        print(f"  Pixel size: ({pixel_width}, {pixel_height})")
        print(f"Source image: {nSrcWidth}x{nSrcHeight} pixels")
    
    # Read source data for all requested bands
    src_data = []
    for band_idx in bands:
        band = src_dataset.GetRasterBand(band_idx)
        band_array = band.ReadAsArray()
        src_data.append(band_array)
    
    # Initialize output array
    if nodata_value is None:
        nodata_value = 0
    
    dst_array = np.full((nBands, nDstHeight, nDstWidth), nodata_value, dtype=src_data[0].dtype)
    
    # Create meshgrid of destination pixel coordinates
    # GDAL uses pixel center convention: add 0.5 to reference pixel centers
    # This matches gdalwarpkernel.cpp lines 5567-5581
    dst_x_coords = np.arange(nDstWidth, dtype=np.float64) + 0.5
    dst_y_coords = np.arange(nDstHeight, dtype=np.float64) + 0.5
    dst_x_grid, dst_y_grid = np.meshgrid(dst_x_coords, dst_y_coords)
    
    # Transform destination pixel coordinates to source pixel coordinates
    # This uses bDstToSrc=True (the standard warping direction)
    src_x_grid, src_y_grid = transformer.transform(dst_x_grid.ravel(), dst_y_grid.ravel(), bDstToSrc=True)
    src_x_grid = src_x_grid.reshape((nDstHeight, nDstWidth))
    src_y_grid = src_y_grid.reshape((nDstHeight, nDstWidth))
    
    if debug:
        print(f"\nTransformed source coordinates:")
        print(f"  X range: [{np.min(src_x_grid):.2f}, {np.max(src_x_grid):.2f}]")
        print(f"  Y range: [{np.min(src_y_grid):.2f}, {np.max(src_y_grid):.2f}]")
        
        # Show some sample transformations
        print(f"\nSample transformations (dest -> src):")
        test_points = [(0, 0), (nDstWidth//2, nDstHeight//2), (nDstWidth-1, nDstHeight-1)]
        for dx, dy in test_points:
            if dx < nDstWidth and dy < nDstHeight:
                sx, sy = src_x_grid[dy, dx], src_y_grid[dy, dx]
                print(f"  Dest pixel center ({dx}+0.5, {dy}+0.5) -> Src pixel ({sx:.2f}, {sy:.2f})")
    
    # Apply nearest neighbor resampling
    # GDAL uses floor (truncation) for indexing - gdalwarpkernel.cpp lines 5313-5318
    # static_cast<int>(coord + 1.0e-10) is essentially floor for positive numbers
    src_x_int = np.floor(src_x_grid).astype(np.int32)
    src_y_int = np.floor(src_y_grid).astype(np.int32)
    
    # Create validity mask: pixels within source image bounds
    valid_mask = (
        (src_x_int >= 0) & (src_x_int < nSrcWidth) &
        (src_y_int >= 0) & (src_y_int < nSrcHeight)
    )
    
    num_valid_pixels = np.sum(valid_mask)
    if debug:
        print(f"\n  Valid pixels: {num_valid_pixels} / {nDstWidth * nDstHeight} ({100.0*num_valid_pixels/(nDstWidth*nDstHeight):.1f}%)")
        
        if num_valid_pixels > 0:
            # Show where valid pixels are
            valid_y, valid_x = np.where(valid_mask)
            print(f"  Valid pixel range in dest: X=[{np.min(valid_x)}, {np.max(valid_x)}], Y=[{np.min(valid_y)}, {np.max(valid_y)}]")
            
            # Show corresponding source coordinates
            print(f"  Corresponding src coords: X=[{np.min(src_x_int[valid_mask])}, {np.max(src_x_int[valid_mask])}], Y=[{np.min(src_y_int[valid_mask])}, {np.max(src_y_int[valid_mask])}]")
            
            # Check source data values at those coordinates
            sample_indices = np.random.choice(num_valid_pixels, min(5, num_valid_pixels), replace=False)
            print(f"\n  Sample source values at valid coordinates:")
            for idx in sample_indices:
                dst_x, dst_y = valid_x[idx], valid_y[idx]
                src_x, src_y = src_x_int[valid_y[idx], valid_x[idx]], src_y_int[valid_y[idx], valid_x[idx]]
                src_val = src_data[0][src_y, src_x]
                print(f"    Dest({dst_x},{dst_y}) <- Src({src_x},{src_y}) = {src_val}")
        else:
            print(f"  WARNING: No valid pixels found! All destination pixels map outside source bounds.")
    
    # Sample source pixels for each band
    for band_num in range(nBands):
        # Extract valid coordinates
        valid_dst_y, valid_dst_x = np.where(valid_mask)
        valid_src_x = src_x_int[valid_mask]
        valid_src_y = src_y_int[valid_mask]
        
        # Read source values and write to destination
        dst_array[band_num, valid_dst_y, valid_dst_x] = src_data[band_num][valid_src_y, valid_src_x]
    
    if debug and num_valid_pixels > 0:
        print(f"\nOutput array statistics:")
        for band_num in range(nBands):
            non_nodata = dst_array[band_num] != nodata_value
            if np.any(non_nodata):
                print(f"  Band {band_num+1}: min={np.min(dst_array[band_num][non_nodata]):.1f}, max={np.max(dst_array[band_num][non_nodata]):.1f}, non-zero pixels={np.sum(non_nodata)}")
            else:
                print(f"  Band {band_num+1}: All pixels are nodata value ({nodata_value})")
    
    return dst_array


def warp_to_file(src_filename, dst_filename, gcps_list, dst_extent, dst_pixel_size,
                 driver_name='GTiff', bands=None, nodata_value=None, 
                 bRefine=False, dfTolerance=0.0, nGCPmin=-1, debug=False,
                 reopen_output=True):
    """
    Complete warp operation: read source, warp using nearest neighbor, write to destination file.
    
    Parameters:
    -----------
    src_filename : str
        Path to source image file
    dst_filename : str
        Path to output warped image file
    gcps_list : list
        List of GCPs as (X_geo, Y_geo, X_pixel, Y_pixel) tuples
    dst_extent : tuple
        Destination extent as (xmin, ymin, xmax, ymax)
    dst_pixel_size : tuple
        Destination pixel size as (pixel_width, pixel_height)
    driver_name : str
        GDAL driver name for output format (default: 'GTiff')
    bands : list or None
        List of band numbers (1-indexed) to warp. If None, warps all bands.
    nodata_value : float or None
        Value to use for pixels outside source bounds
    bRefine : bool
        Whether to refine GCPs by removing outliers
    dfTolerance : float
        Tolerance for outlier removal (in georeferenced units)
    nGCPmin : int
        Minimum number of GCPs to keep after outlier removal
    debug : bool
        If True, print diagnostic information
    reopen_output : bool
        If True, close and re-open the output file before returning
    
    Returns:
    --------
    dst_dataset : gdal.Dataset
        The created destination dataset
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(dst_filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if debug:
        print(f"Output path: {dst_filename}")
        print(f"Output dir exists: {os.path.isdir(out_dir)}")
        print(f"Output dir writable: {os.access(out_dir, os.W_OK)}")
        try:
            test_path = os.path.join(out_dir, "_write_test.tmp")
            with open(test_path, "wb") as tmp_f:
                tmp_f.write(b"test")
            os.remove(test_path)
            print("Output dir write test: OK")
        except Exception as exc:
            print(f"Output dir write test failed: {exc}")

    # Open source dataset
    src_ds = gdal.Open(src_filename, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Failed to open source file: {src_filename}")
    
    # Create transformer
    transformer = GenImgProjTransformer(
        gcps_list, dst_extent, dst_pixel_size,
        bRefine=bRefine, dfTolerance=dfTolerance, nGCPmin=nGCPmin
    )
    
    # Perform warping
    dst_array = warp_image_nearest(
        src_ds, transformer, dst_extent, dst_pixel_size,
        bands=bands, nodata_value=nodata_value, debug=debug
    )
    
    # Calculate destination dimensions
    xmin, ymin, xmax, ymax = dst_extent
    pixel_width, pixel_height = dst_pixel_size
    nDstWidth = int(np.round((xmax - xmin) / pixel_width))
    nDstHeight = int(np.round((ymax - ymin) / pixel_height))
    
    # Determine output data type
    if bands is None:
        sample_band = src_ds.GetRasterBand(1)
    else:
        sample_band = src_ds.GetRasterBand(bands[0])
    
    # Map NumPy dtype to GDAL type
    dtype_map = {
        np.uint8: gdal.GDT_Byte,
        np.int16: gdal.GDT_Int16,
        np.uint16: gdal.GDT_UInt16,
        np.int32: gdal.GDT_Int32,
        np.uint32: gdal.GDT_UInt32,
        np.float32: gdal.GDT_Float32,
        np.float64: gdal.GDT_Float64,
    }
    gdal_dtype = dtype_map.get(dst_array.dtype.type, gdal.GDT_Float32)
    
    # Create destination dataset
    driver = gdal.GetDriverByName(driver_name)
    dst_ds = driver.Create(dst_filename, nDstWidth, nDstHeight, dst_array.shape[0], gdal_dtype)
    temp_path = None
    
    if dst_ds is None:
        if debug:
            print(f"GDAL Create failed for: {dst_filename}")
            print("Attempting to create in a temp directory and move to destination...")
        temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(dst_filename))
        dst_ds = driver.Create(temp_path, nDstWidth, nDstHeight, dst_array.shape[0], gdal_dtype)
        if dst_ds is None:
            raise RuntimeError(f"Failed to create destination file: {dst_filename}")
    
    # Set geotransform for destination
    # Geotransform: [x_origin, pixel_width, 0, y_origin, 0, -pixel_height]
    geotransform = [xmin, pixel_width, 0, ymax, 0, -pixel_height]
    dst_ds.SetGeoTransform(geotransform)
    
    if debug:
        print(f"\nDestination dataset created:")
        print(f"  Dimensions: {nDstWidth}x{nDstHeight}")
        print(f"  Geotransform: {geotransform}")
    
    # Try to get and set projection from source
    # First try GetProjection()
    src_projection = src_ds.GetProjection()
    if not src_projection:
        # If no projection, try to get it from GCPs
        gcps = src_ds.GetGCPs()
        if gcps:
            src_projection = src_ds.GetGCPProjection()
    
    if src_projection:
        dst_ds.SetProjection(src_projection)
        if debug:
            print(f"  Projection set from source")
    elif debug:
        print(f"  Warning: No projection found in source dataset")
    
    # Write bands
    for band_num in range(dst_array.shape[0]):
        dst_band = dst_ds.GetRasterBand(band_num + 1)
        dst_band.WriteArray(dst_array[band_num])
        if nodata_value is not None:
            dst_band.SetNoDataValue(float(nodata_value))
        dst_band.FlushCache()
    
    # Flush to disk
    dst_ds.FlushCache()

    # Close datasets to ensure data is written to disk
    dst_ds = None
    src_ds = None

    # If we created in temp, move to destination
    if temp_path is not None:
        shutil.move(temp_path, dst_filename)

    if reopen_output:
        return gdal.Open(dst_filename, gdal.GA_ReadOnly)
    return None