import numpy as np

def match_shape(a, b, mode='trunc'): # can be 'trunc'(truncate), or 'pad'
    if a.shape == b.shape:
        return a, b
    elif mode.startswith('trunc'):
        shape = np.minimum(a.shape, b.shape)
        return a[tuple(slice(0, i) for i in shape)], b[tuple(slice(0, i) for i in shape)]
    
    #find the shapes
    shape_a, shape_b = np.array(a.shape), np.array(b.shape)

    # determine the padding lengths for the shorter array
    if shape_a[-1] < shape_b[-1]:
        padding = [(0, 0)] * (a.ndim - 1) + [(0, shape_b[-1] - shape_a[-1])]
        a = np.pad(a, pad_width=padding, mode='constant')
    elif shape_b[-1] < shape_a[-1]:
        padding = [(0, 0)] * (b.ndim - 1) + [(0, shape_a[-1] - shape_b[-1])]
        b = np.pad(b, pad_width=padding, mode='constant')

    return a, b

def mse(a, b, axis=None, mode='pad'):
    a, b = match_shape(a, b, mode)
    return np.mean((a - b) ** 2, axis=axis)

def rmse(a, b, axis=None, mode='pad'):
    a, b = match_shape(a, b, mode)
    return np.sqrt(np.mean((a - b) ** 2, axis=axis))

def mae(a, b, axis=None, mode='pad'):
    a, b = match_shape(a, b, mode)
    return np.mean(np.abs(a - b), axis=axis)

def cosine_distance(a, b, mode='pad'):
    a, b = match_shape(a, b, mode)
    
    # Cosine distance (0 means identical, 1 means completely different)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def psnr(original, target, mean_mlc=False, mode='pad'):

    kwargs={}
    kwargs['axis'] = 1 if original.ndim > 1 else 0
    rmseval = rmse(original, target, mode=mode, **kwargs)
    max_pixel = np.max(original, **kwargs)

    if np.any(rmseval == 0):
        # prevent devision by zero
        return float('inf')
    
    retval = 20 * np.log10(max_pixel / rmseval)

    if mean_mlc:
        retval = np.mean(retval)
    return retval




__all__ = [
    'match_shape',
    'psnr',
]

