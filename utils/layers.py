import tensorflow as tf
import numpy as np
from args import configs
from utils.evals import decode

def conv2d(inputs, filters, kernel_size, strides=1, padding=0,
           kernel_initializer=tf.initializers.he_normal(),
#             kernel_initializer=tf.random_normal_initializer(),
           use_bias=True, **args):
    if(padding > 0):
        inputs = tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='CONSTANT')
    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                              kernel_initializer = kernel_initializer,
                              padding='VALID', use_bias=use_bias, **args)
    return inputs

def max_pool2d(inputs, pool_size, strides, padding=0, **args):
    if(padding > 0):
        inputs = tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode='CONSTANT')
    inputs = tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=strides, padding='valid', **args)
    return inputs

def upsample_layer(x, scale=2):
    '''
    Upsample feature maps using nearest neighbour method without setting a specific output shape
    '''
    assert scale > 0 and isinstance(scale, int)             #Please make sure that scale is a int object and is a natural number
    xs = tf.shape(x)[1:3]
    with tf.variable_scope('up_'):
        y = tf.image.resize_nearest_neighbor(x, xs * scale)
    return y    

def stem(inputs, out_channels, is_training):
    inputs = conv2d(inputs, out_channels, kernel_size=7, strides=2, padding=3, use_bias=False)
    inputs = tf.layers.batch_normalization(inputs, training=is_training, momentum=1.0, epsilon=0.0, renorm_momentum=1.0)
    inputs = tf.nn.relu(inputs)
    inputs = max_pool2d(inputs, pool_size=3, strides=2, padding=1)
    return inputs

def bottleneck(
    inputs,
    in_channels,
    bottleneck_channels,
    out_channels,
    stride_in_1x1,
    stride,
    is_training
    ):
    
    residual = inputs
    if in_channels != out_channels:
        residual = conv2d(inputs, out_channels, kernel_size=1, strides=stride, use_bias=False)
        #Set momentum=1.0 and epsilon=0.0 to guarantee result can be reproduced
        residual = tf.layers.batch_normalization(residual, training=is_training, momentum=1.0, epsilon=0.0, renorm_momentum=1.0)
    
    stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
    x = conv2d(
            inputs,
            bottleneck_channels,
            kernel_size=1,
            strides=stride_1x1,
            use_bias=False
        )
    
    x = tf.layers.batch_normalization(x, training=is_training, momentum=1.0, epsilon=0.0, renorm_momentum=1.0)
    x = tf.nn.relu(x)
    x = conv2d(
            x,
            bottleneck_channels,
            kernel_size=3,
            strides=stride_3x3,
            padding=1,
            use_bias=False
        )

    x = tf.layers.batch_normalization(x, training=is_training, momentum=1.0, epsilon=0.0, renorm_momentum=1.0)
    x = tf.nn.relu(x)
    x = conv2d(x, out_channels, kernel_size=1, use_bias=False)
    x = tf.layers.batch_normalization(x, training=is_training, momentum=1.0, epsilon=0.0, renorm_momentum=1.0)
    x = tf.nn.relu(x + residual)
    return x

def cls_head(inputs, channels, repeats, nums_anchors, nums_classes, reuse_name):
    for i in range(repeats):
        layer_name = reuse_name + str(i)
        inputs = conv2d(inputs, channels, 3, strides=1, padding=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=layer_name)
        inputs = tf.nn.relu(inputs)
    layer_name = reuse_name + 'out'
    prior_prob = 0.02                                            #initialize cls bias with prior value
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    cls = conv2d(inputs, nums_anchors * nums_classes, 3, strides=1, padding=1, name=layer_name, bias_initializer=tf.constant_initializer(bias_value))
    return cls

def bbox_head(inputs, channels, repeats, nums_anchors, reuse_name):
    for i in range(repeats):
        layer_name = reuse_name + str(i)
        inputs = conv2d(inputs, channels, 3, strides=1, padding=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), name=layer_name)
        inputs = tf.nn.relu(inputs)
    layer_name = reuse_name + 'out'
    bbox = conv2d(inputs, nums_anchors * 4, 3, strides=1, padding=1, name=layer_name)    
    return bbox

def retinanet_head(features, nums_anchors, nums_classes, in_channels):
    logits = []
    bbox_reg = []
    with tf.variable_scope('retinanet_head', reuse=tf.AUTO_REUSE):
        for i in range(len(features)):
            logits.append(cls_head(features[i], 256, 4, nums_anchors, nums_classes, reuse_name='cls_'))
            bbox_reg.append(bbox_head(features[i], 256, 4, nums_anchors, reuse_name='bbox_'))
    
    return logits, bbox_reg

def compute_single_feature_map(anchors, box_cls, box_regression, img_size, pre_nms_thresh,
                               batch_size, nums_anchors, nums_classes, max_nms_top_n=1000):
    H, W = tf.shape(box_cls)[2], tf.shape(box_cls)[3]
    N, A, C = batch_size, nums_anchors, nums_classes
    img_size = tf.concat([img_size[:,::-1], img_size[:,::-1]], axis=1)
    
    box_cls = tf.reshape(box_cls, (N, H, W, -1, C))
    box_cls = tf.reshape(box_cls, (N, -1, C))
    box_cls = tf.nn.sigmoid(box_cls)
    box_regression = tf.reshape(box_regression, (N, H, W, -1, 4))
    box_regression = tf.reshape(box_regression, (N, -1, 4))
    
    detections, labels, scores = [[]] * N, [[]] * N, [[]] * N
    candidate_inds = tf.cast(box_cls > pre_nms_thresh, tf.float32)
#     if(tf.reduce_sum(candidate_inds) == 0):
#         return [], [], []
    
    pre_nms_top_n = tf.reduce_sum(tf.reshape(candidate_inds, (N, -1)), axis=1)
    pre_nms_top_n = tf.clip_by_value(pre_nms_top_n, float('-inf'), max_nms_top_n)
    
    def cut_top_n(per_box_cls, per_box_loc, per_class, per_pre_nms_top_n):
        cls_top_n = tf.nn.top_k(per_box_cls, tf.cast(per_pre_nms_top_n, tf.int32), sorted=False)
        per_box_cls, top_k_indices = cls_top_n.values, cls_top_n.indices
        per_box_loc = tf.gather(per_box_loc, top_k_indices)
        per_class = tf.gather(per_class, top_k_indices)
        return per_box_cls, per_box_loc, per_class
    
    for i in range(batch_size):
        per_box_cls = box_cls[i]
        per_box_regression = box_regression[i]
        per_pre_nms_top_n = pre_nms_top_n[i]
        per_candidate_inds = tf.where(per_box_cls > pre_nms_thresh)
        
        per_box_cls = tf.gather_nd(per_box_cls, per_candidate_inds)
        per_box_loc = per_candidate_inds[:,0]
        per_class = per_candidate_inds[:, 1]
        
        per_box_cls, per_box_loc, per_class = tf.cond(tf.shape(per_candidate_inds)[0] > tf.cast(per_pre_nms_top_n, tf.int32),
            lambda : cut_top_n(per_box_cls, per_box_loc, per_class, per_pre_nms_top_n),
            lambda : (per_box_cls, per_box_loc, per_class))
        
        box_regression_sel = tf.gather(per_box_regression, per_box_loc)
        anchors_sel = tf.gather(anchors, per_box_loc)
        
        detection = decode(box_regression_sel, anchors_sel)
        detection = tf.clip_by_value(detection, 0, tf.cast(img_size[i], tf.float32))
        detections[i] = detection
        labels[i] = per_class
        scores[i] = per_box_cls
        
    return detections, labels, scores
    
def select_bboxs(boxlists, labellists, scorelists,
                 num_classes, iou_threshold, fpn_post_nms_top_n):
    num_images = len(boxlists)
    results = []
    for i in range(num_images):
        scores = scorelists[i]
        labels = labellists[i]
        boxes = boxlists[i]
        boxes_sel, scores_sel, labels_sel = [], [], []
        for j in range(num_classes):
            inds = tf.reshape(tf.where(tf.equal(labels, j)), [-1])
            scores_j = tf.gather(scores, inds)
            boxes_j = tf.gather(boxes, inds)
            
            nms_indices = tf.image.non_max_suppression(boxes=boxes_j,
                                                       scores=scores_j,
                                                       max_output_size=-1,
                                                       iou_threshold=iou_threshold)
            boxes_sel.append(tf.gather(boxes_j, nms_indices))
            scores_sel.append(tf.gather(scores_j, nms_indices))
            labels_sel.append(tf.ones_like(scores_sel[-1], tf.int32) * j)
        
        boxes_sel = tf.concat(boxes_sel, axis=0)
        scores_sel = tf.concat(scores_sel, axis=0)
        labels_sel = tf.concat(labels_sel,axis=0)
        
        inds = tf.nn.top_k(scores_sel, tf.minimum(fpn_post_nms_top_n, tf.shape(scores_sel)[0]))
        scores_sel = inds.values
        inds = inds.indices
        boxes_sel = tf.gather(boxes_sel, inds)
        labels_sel = tf.gather(labels_sel, inds)
        
        # padding 0 when no box is detected
        boxes_sel, scores_sel, labels_sel = tf.cond(tf.equal(tf.shape(boxes_sel)[0], 0),
            lambda : (tf.zeros((1, 4), tf.float32), tf.ones((1), tf.float32) * 0.01, tf.zeros((1), tf.int32)),
            lambda : (boxes_sel, scores_sel, labels_sel))
        results.append([boxes_sel, scores_sel, labels_sel])
        
    return list(zip(*results))
