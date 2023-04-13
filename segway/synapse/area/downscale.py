import skimage
import daisy

def downscale_block(in_array, scale_factor, out_array=None, roi=None):

    if roi is None:
        roi = in_array.roi

    scale_factor = tuple(scale_factor)
    dims = len(scale_factor)
    in_data = in_array.to_ndarray(roi, fill_value=0)

    in_shape = daisy.Coordinate(in_data.shape[-dims:])
    assert in_shape.is_multiple_of(scale_factor)

    if out_array is None:
        data = np.empty([int(i/f) for i, f in zip(in_shape, scale_factor)], dtype=in_array.dtype)
        out_array = daisy.Array(roi=roi, voxel_size=in_array.voxel_size*scale_factor)

    n_channels = len(in_data.shape) - dims
    if n_channels >= 1:
        scale_factor = (1,)*n_channels + scale_factor

    if in_data.dtype == np.uint64:
        slices = tuple(slice(k//2, None, k) for k in scale_factor)
        out_data = in_data[slices]
    else:
        out_data = skimage.measure.block_reduce(in_data, scale_factor, np.mean)

    out_array[roi] = out_data

    return out_array
