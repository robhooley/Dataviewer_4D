import numpy as np

def create_circular_mask(image_height, image_width, mask_center_coordinates=None, mask_radius=None):
    if mask_center_coordinates is None:  # use the middle of the image
        mask_center_coordinates = (int(image_width/2), int(image_height/2))
    if mask_radius is None:  # use the smallest distance between the center and image walls
        mask_radius = min(mask_center_coordinates[0], mask_center_coordinates[1], image_width - mask_center_coordinates[0], image_height - mask_center_coordinates[1])
    Y, X = np.ogrid[:image_height, :image_width]
    dist_from_center = np.sqrt((X - mask_center_coordinates[0])**2 + (Y - mask_center_coordinates[1])**2)
    mask = dist_from_center <= mask_radius
    return mask

def VBF(image_array,radius,center):
    camera_data_shape = image_array[0][0].shape  # shape of first image to get image dimensions
    dataset_shape = image_array.shape[0], image_array.shape[1]  # scanned region shape
    VBF_intensity_list = []  # empty list to take virtual bright field image sigals
    integration_mask = create_circular_mask(camera_data_shape[0], camera_data_shape[1], mask_center_coordinates=center,mask_radius=radius)
    for row in image_array:  # iterates through array rows
        for pixel in row:  # in each row iterates through pixels
            VBF_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
            VBF_intensity_list.append(VBF_intensity)  # adds to the list

    VBF_intensity_array = np.asarray(VBF_intensity_list)  # converts list to array
    VBF_intensity_array = np.reshape(VBF_intensity_array, (
        dataset_shape[0], dataset_shape[1]))  # reshapes array to match image dimensions
    return VBF_intensity_array

def VDF_multi(image_array, radius, centers):
    dataset_shape = image_array.shape[:2]
    cam_shape = image_array[0, 0].shape
    result = np.zeros(dataset_shape, dtype=np.float32)

    # Precompute masks
    masks = [create_circular_mask(cam_shape[0], cam_shape[1],
                                  mask_center_coordinates=c, mask_radius=radius)
             for c in centers]

    for iy, row in enumerate(image_array):
        for ix, pixel in enumerate(row):
            val = 0.0
            for m in masks:
                val += np.sum(pixel[m])
            result[iy, ix] = val
    return result



def VADF(array4d, r_in, r_out, center):
    H, W = array4d.shape[2:]
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cx, cy = center
    rr = np.sqrt((X - cx)**2 + (Y - cy)**2)

    mask_outer = (rr < r_out)
    mask_inner = (rr < r_in)
    annulus_mask = mask_outer ^ mask_inner  # pixels in [r_in, r_out)

    # Collapse DP axes into 1D, mask once, sum in vectorized manner
    return array4d[:, :, annulus_mask].sum(axis=-1)

def cross_correlation_map(
    array4d: np.ndarray,
    *_ignored,                     # <— absorbs (radius, center)
    ddof: int = 0,
    dp_block: int = 16384,
    clip_percentiles=(1.0, 99.0),
    invert: bool = False
) -> np.ndarray:
    # ensure dp_block is an int even if something odd was passed
    try:
        dp_block = int(dp_block)
    except Exception:
        dp_block = 16384

    if array4d.ndim != 4:
        raise ValueError(f"Expected 4-D (Ny, Nx, Dy, Dx), got {array4d.shape}")

    Ny, Nx, Dy, Dx = map(int, array4d.shape)
    P = Dy * Dx
    A = array4d.reshape(Ny, Nx, P).astype(np.float32, copy=False)

    # per-pixel stats
    mean = A.mean(axis=-1)
    std  = A.std(axis=-1, ddof=ddof)
    valid = np.isfinite(std) & (std > 0)

    # accumulators
    num_right = np.zeros((Ny, Nx-1), dtype=np.float64)
    num_down  = np.zeros((Ny-1, Nx), dtype=np.float64)

    for k0 in range(0, P, dp_block):
        k1 = min(P, k0 + dp_block)

        a_r = A[:, :-1, k0:k1]; b_r = A[:,  1:, k0:k1]
        ma  = mean[:, :-1][..., None]; mb = mean[:,  1:][..., None]
        num_right += np.sum((a_r - ma) * (b_r - mb), axis=-1, dtype=np.float64)

        a_d = A[:-1, :, k0:k1]; b_d = A[ 1:, :, k0:k1]
        ma  = mean[:-1, :][..., None]; mb = mean[ 1:, :][..., None]
        num_down  += np.sum((a_d - ma) * (b_d - mb), axis=-1, dtype=np.float64)

    n_eff = (P - ddof) if ddof in (0, 1) else max(1, P - ddof)
    denom_right = (std[:, :-1] * std[:, 1:]) * n_eff
    denom_down  = (std[:-1, :] * std[ 1:, :]) * n_eff

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_right = num_right / denom_right
        corr_down  = num_down  / denom_down

    pair_valid_right = np.isfinite(corr_right) & valid[:, :-1] & valid[:, 1:]
    pair_valid_down  = np.isfinite(corr_down)  & valid[:-1, :] & valid[1:, :]

    corr_map = np.full((Ny, Nx), np.nan, dtype=np.float32)
    sum_corr = np.zeros_like(corr_map)
    count    = np.zeros_like(corr_map)

    sum_corr[:, 1:]  += np.where(pair_valid_right, corr_right, 0.0)
    count[:, 1:]     += pair_valid_right
    sum_corr[:, :-1] += np.where(pair_valid_right, corr_right, 0.0)
    count[:, :-1]    += pair_valid_right

    sum_corr[1:, :]  += np.where(pair_valid_down, corr_down, 0.0)
    count[1:, :]     += pair_valid_down
    sum_corr[:-1, :] += np.where(pair_valid_down, corr_down, 0.0)
    count[:-1, :]    += pair_valid_down

    with np.errstate(invalid="ignore", divide="ignore"):
        corr_map = sum_corr / count
    corr_map[count == 0] = np.nan

    # display: dissimilarity + percentile stretch → 8-bit
    disp = (1.0 - corr_map) if invert else corr_map
    vals = disp[np.isfinite(disp)]
    if vals.size == 0:
        return np.zeros((Ny, Nx), np.uint8)

    lo, hi = np.percentile(vals, clip_percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros((Ny, Nx), np.uint8)

    disp = np.clip(disp, lo, hi)
    disp = (disp - lo) / (hi - lo)
    return (disp * 255.0).astype(np.uint8)

