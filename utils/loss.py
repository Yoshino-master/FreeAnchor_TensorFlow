import tensorflow as tf
import math
from utils.evals import calc_iou_tf
from utils.evals import decode, encode

class FreeAnchorLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.xywh_weights = (10.0, 10.0, 5.0, 5.0)
        self.bbox_xform_clip = math.log(1000. / 16)
    
    def matched_box_prob(self, indices, labels, object_box_prob_select, len_anchors, nums_classes):
        labels = tf.expand_dims(labels, axis=-1)
        s = tf.shape(object_box_prob_select)
        nonzero_box_prob = tf.where(tf.equal(labels, tf.cast(tf.gather(indices, 0), tf.float32)), object_box_prob_select, tf.zeros(s))
        nonzero_box_prob = tf.reduce_max(nonzero_box_prob, axis=0)
        indices_f = tf.transpose(tf.gather(indices, [1,0]), (1,0))
        image_box_prob = tf.sparse.SparseTensor(indices_f, nonzero_box_prob, dense_shape=((len_anchors, nums_classes)))
        image_box_prob = tf.sparse.to_dense(image_box_prob, validate_indices=False)
        return image_box_prob
    
    def dismatched_box_prob(self, len_anchors, nums_classes):
        return tf.zeros((len_anchors, nums_classes))
    
    def forward(self, anchors, box_cls, box_regression, bboxs, batch_labels, batch_img_size, bboxs_num):
        box_cls_flattened, box_regression_flattened = [], []
        for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
        ):
            cls_shape = tf.shape(box_cls_per_level)
            _, H, W, A = cls_shape[0], cls_shape[1], cls_shape[2], cls_shape[3]
            C = self.cfg.num_classes
            N = self.cfg.batch_size
            box_cls_per_level = tf.reshape(box_cls_per_level, shape=[N, -1, C])
            box_regression_per_level = tf.reshape(box_regression_per_level, shape=[N, -1, 4])    #$$$$$$$$$$$$$$$$$
            box_cls_flattened.append(box_cls_per_level)
            box_regression_flattened.append(box_regression_per_level)
        
        box_cls = tf.concat(box_cls_flattened, axis=1)
        box_regression_cat = tf.concat(box_regression_flattened, axis=1)
        anchors = tf.concat(anchors, axis=0)
        cls_prob = tf.nn.sigmoid(box_cls)
        anchor_shape = tf.shape(anchors)
        box_prob, positive_losses = [], []
        
        for i in range(N):
            box = tf.gather(bboxs[i], tf.range(0, bboxs_num[i], 1))
            labels = tf.gather(batch_labels[i], tf.range(0, bboxs_num[i], 1))
            cls_prob_ = cls_prob[i]
            
            box_localization = decode(box_regression_cat[i], anchors, self.xywh_weights, self.bbox_xform_clip)
            ious = calc_iou_tf(box, box_localization)
            
            t1 = self.cfg.bbox_threshold
            t2 = tf.clip_by_value(tf.expand_dims(tf.reduce_max(ious, axis=[1]), axis=-1), t1+1e-12, float('inf'))
            object_box_prob = tf.clip_by_value((ious - t1) / (t2 - t1), 0, 1)
            
            oh_labels = tf.one_hot(tf.cast(labels, tf.int64), tf.cast(tf.reduce_max(labels, 0) + 1, dtype=tf.int32))
            oh_labels = tf.transpose(oh_labels, perm=(1,0))
            object_cls_box_prob = tf.expand_dims(tf.transpose(object_box_prob, perm=(1,0)), axis=1) * oh_labels
            object_cls_box_prob = tf.transpose(object_cls_box_prob, perm=(2,1,0))
             
            indices = tf.reduce_sum(object_cls_box_prob, axis=0)
            indices = tf.transpose(tf.where(indices > 0), (1,0))
             
            object_box_prob_select = tf.gather(object_box_prob, indices[1], axis=1)
            image_box_prob = tf.cond(tf.equal(tf.size(indices), 0), 
                               lambda : self.dismatched_box_prob(anchor_shape[0], self.cfg.num_classes),
                               lambda : self.matched_box_prob(indices, labels, object_box_prob_select,
                                        anchor_shape[0], self.cfg.num_classes))
            box_prob.append(image_box_prob)
             
            match_quality_matrix = calc_iou_tf(box, anchors)
            matched = tf.nn.top_k(match_quality_matrix, self.cfg.pre_anchor_topk, sorted=False).indices
            
            index_ = tf.range(0, tf.shape(labels)[0], 1)
            label_index = tf.transpose(tf.concat([[index_, tf.cast(labels, tf.int32)]], axis=0), (1,0))
            cls_prob_tmp = tf.gather(cls_prob_, indices=matched, axis=0)
            cls_prob_tmp = tf.transpose(cls_prob_tmp, (0,2,1))
            matched_cls_prob = tf.gather_nd(cls_prob_tmp, indices = label_index)                             #checked
             
            matched_object_targets = encode(tf.expand_dims(box, axis=1), tf.gather(anchors, indices=matched, axis=0), self.xywh_weights)
            retinanet_regression_loss = smooth_l1_loss(tf.gather(box_regression_cat[i], matched, axis=0),
                                                       matched_object_targets,
                                                       self.cfg.bbox_reg_weight, self.cfg.bbox_reg_beta)
            matched_box_prob = tf.exp(-retinanet_regression_loss)
            positive_losses.append(positive_bag_loss(matched_cls_prob * matched_box_prob, dims=1))
            
        positive_numels = tf.reduce_sum(bboxs_num)
        positive_loss = tf.reduce_sum(tf.concat(positive_losses, axis=0)) / tf.cast(tf.maximum(1, tf.cast(positive_numels, tf.int32)), tf.float32)
        box_prob = tf.stack(box_prob)
        negative_loss = focal_loss(cls_prob * (1 - box_prob), self.cfg.focal_loss_gamma)  \
             / tf.cast(tf.maximum(1, tf.cast(positive_numels * self.cfg.pre_anchor_topk, tf.int32)), tf.float32)
            
        return positive_loss * self.cfg.focal_loss_alpha + negative_loss * (1 - self.cfg.focal_loss_alpha)
        
def tensor2sparse(tensor):
    arr_idx = tf.where(tf.not_equal(tensor, 0))
    arr_sparse = tf.SparseTensor(arr_idx, tf.gather_nd(tensor, arr_idx), tensor.get_shape())
    return arr_sparse

def smooth_l1_loss(pred, target, weight, beta):
    val = target - pred
    abs_val = tf.abs(val)
    return weight * tf.reduce_sum(tf.where(abs_val < beta, 0.5 / beta * tf.pow(val, 2), (abs_val - 0.5 * beta)), axis=-1)

def positive_bag_loss(logits, dims):
    weight = 1.0 / tf.clip_by_value(1 - logits, 1e-12, float('inf'))
    weight_div = tf.reduce_sum(weight, axis=dims)
    weight = tf.transpose(tf.transpose(weight, (1,0)) / weight_div, (1,0))
    bag_prob = tf.reduce_sum((weight * logits), axis=dims)
    return tf.keras.backend.binary_crossentropy(tf.ones_like(bag_prob), bag_prob)
    
def focal_loss(logits, gamma):
    #count focal loss for negative_loss
    logits_ = tf.pow(logits, gamma)
    bce_loss = tf.keras.backend.binary_crossentropy(tf.zeros_like(logits), logits)
    return tf.reduce_sum(bce_loss * logits_)

