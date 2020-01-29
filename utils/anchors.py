import numpy as np
import tensorflow as tf

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    scales = np.array(sizes, dtype=np.float) / stride
    aspect_ratios = np.array(aspect_ratios, dtype=np.float)
    anchor = np.array([1, 1, stride, stride], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors
    
def get_cell_anchors(anchor_sizes, aspect_ratios, anchor_strides, straddle_thresh, octave, scales_per_octave):
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
    
    cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if type(size) is tuple else (size,),
                    aspect_ratios
                ).astype('float32')
                for anchor_stride, size in zip(anchor_strides, tuple(new_anchor_sizes))
            ]
    return cell_anchors

def anchor_generator(features, anchor_strides, cell_anchors):
    grid_sizes = [tf.shape(fea)[1:3][::-1] for fea in features]
    anchors = []
    for size, stride, base_anchors in zip(
        grid_sizes, anchor_strides, cell_anchors):
        grid_height, grid_width = size[0], size[1]
        shifts_x = tf.range(0, tf.cast(grid_width, tf.float32) * stride, stride, dtype=tf.float32)
        shifts_y = tf.range(0, tf.cast(grid_height, tf.float32) * stride, stride, dtype=tf.float32)
        shift_x, shift_y = tf.meshgrid(shifts_y, shifts_x)
        shift_x = tf.reshape(shift_x, shape=(-1,))
        shift_y = tf.reshape(shift_y, shape=(-1,))
        shifts = tf.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
        anchor = tf.reshape(shifts, shape=(-1, 1, 4)) + tf.reshape(base_anchors, shape=(1, -1, 4))
        anchors.append(tf.reshape(anchor, shape=(-1, 4)))
    return anchors
