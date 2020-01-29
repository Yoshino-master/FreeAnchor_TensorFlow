import tensorflow as tf
import math
import numpy as np
from utils.data_aug import resize_bboxs
from utils.misc import AvarageMeter


def calc_iou_tf(boxes1, boxes2):
    '''
    Params :
        boxes1 : [N, 4], (xmin, ymin, xmax, ymax)
        boxes2 : [M, 4], (xmin, ymin, xmax, ymax)
    '''
    TO_REMOVE = 1
    
    #left_top, right_bottom values, shape : [N, M, 2]
    intersect_lt = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    intersect_rb = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    
    #width_height values of boxes1, boxes2, intersected_box
    boxes1_wh = tf.clip_by_value(boxes1[:, 2:] - boxes1[:, :2] + TO_REMOVE, 0, float('inf'))         #shape : [N, 2]
    boxes2_wh = tf.clip_by_value(boxes2[:, 2:] - boxes2[:, :2] + TO_REMOVE, 0, float('inf'))         #shape : [M, 2]
    intersect_wh = tf.clip_by_value(intersect_rb - intersect_lt + TO_REMOVE, 0, float('inf'))        #shape : [N, M, 2]
    
    #area of boxes
    boxes1_area = boxes1_wh[:, 0] * boxes1_wh[:, 1]                   #shape : [N]
    boxes2_area = boxes2_wh[:, 0] * boxes2_wh[:, 1]                   #shape : [M]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]      #shape : [N, M]
    
    boxes1_area = tf.expand_dims(boxes1_area, axis=1)
    iou = intersect_area / (boxes1_area + boxes2_area - intersect_area)     #shape : [N, M]
    return iou

def voc_ap(rec, prec, use_11_points=False):
    '''Compute ap on rec([N]) and prec([N])'''
    if use_11_points:
        ap = 0.0
        for t in range(0., 1.1, 0.1):
            if np.sum(rec > t) == 0:
                p = 0.
            else:
                p = np.max(prec[ rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
def decode(box_regression, anchors, xywh_weights = (10.0, 10.0, 5.0, 5.0),
           bbox_xform_clip = math.log(1000. / 16)):
    ''' reorganize the box_regression to the real coordination with mode xyxy '''
    # Transfer anchors from xyxy mode to cx,cy,w,h format
    TO_REMOVE = 1.0
    x1, y1, x2, y2 = tf.split(anchors, 4, axis=1)
    widths  = tf.add(TO_REMOVE, tf.subtract(x2, x1))            #width = x2 - x1 + 1
    heights = tf.add(TO_REMOVE, tf.subtract(y2, y1))            #heights = y2 - y1 + 1
    ctr_x = tf.add(x1, tf.multiply(0.5, widths))                #ctr_x = x1 + 0.5 * widths
    ctr_y = tf.add(y1, tf.multiply(0.5, heights))               #ctr_y = y1 + 0.5 * heights
    
    #predicted divide weights
    wx, wy, ww, wh = xywh_weights
    box_regression = tf.reshape(box_regression, shape=(-1, 4))
    dx, dy, dw, dh = tf.split(box_regression, 4, axis=1)
    dx = tf.divide(dx, wx)
    dy = tf.divide(dy, wy)
    dw = tf.divide(dw, ww)
    dh = tf.divide(dh, wh)
    
    # Prevent too large value
    dw = tf.clip_by_value(dw, float('-inf'), bbox_xform_clip)
    dh = tf.clip_by_value(dh, float('-inf'), bbox_xform_clip)
    
    #real predicted bbox
    pred_ctr_x = tf.add(ctr_x, tf.multiply(dx, widths))         #pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = tf.add(ctr_y, tf.multiply(dy, heights))        #pred_ctr_y = dy * heights + ctr_y
    pred_w = tf.multiply(widths, tf.exp(dw))                    #pred_w = widths * exp(dw)
    pred_h = tf.multiply(heights, tf.exp(dh))                   #pred_h = heights * exp(dh)
    
    #transfer to xyxy mode
    pred_x1 = tf.subtract(pred_ctr_x, tf.multiply(pred_w, 0.5))                 #x1 = ctr_x - w * 0.5
    pred_y1 = tf.subtract(pred_ctr_y, tf.multiply(pred_h, 0.5))                 #y1 = ctr_y - h * 0.5
    pred_x2 = tf.subtract(tf.add(pred_ctr_x, tf.multiply(pred_w, 0.5)), 1)      #x2 = ctr_x + w * 0.5 + 1
    pred_y2 = tf.subtract(tf.add(pred_ctr_y, tf.multiply(pred_h, 0.5)), 1)      #y2 = ctr_y + h * 0.5 + 1
    pred_boxes = tf.concat([pred_x1, pred_y1, pred_x2, pred_y2], axis=1)
    return pred_boxes

def encode(box_target, box_proposal, xywh_weights = (10.0, 10.0, 5.0, 5.0)):
    ''' Encode boxes with real coordination (xmin, ymin, xmax, ymax) to target (dx, dy, dw, dh) '''
    TO_REMOVE = 1  # TODO remove
    ex_widths = box_proposal[..., 2] - box_proposal[..., 0] + TO_REMOVE
    ex_heights = box_proposal[..., 3] - box_proposal[..., 1] + TO_REMOVE
    ex_ctr_x = box_proposal[..., 0] + 0.5 * ex_widths
    ex_ctr_y = box_proposal[..., 1] + 0.5 * ex_heights
    
    gt_widths = box_target[..., 2] - box_target[..., 0] + TO_REMOVE
    gt_heights = box_target[..., 3] - box_target[..., 1] + TO_REMOVE
    gt_ctr_x = box_target[..., 0] + 0.5 * gt_widths
    gt_ctr_y = box_target[..., 1] + 0.5 * gt_heights
    
    wx, wy, ww, wh = xywh_weights
    targets_dx = tf.expand_dims(wx * (gt_ctr_x - ex_ctr_x) / ex_widths, axis=-1)
    targets_dy = tf.expand_dims(wy * (gt_ctr_y - ex_ctr_y) / ex_heights, axis=-1)
    targets_dw = tf.expand_dims(ww * tf.log(gt_widths / ex_widths), axis=-1)
    targets_dh = tf.expand_dims(wh * tf.log(gt_heights / ex_heights), axis=-1)
    
    targets = tf.concat((targets_dx, targets_dy, targets_dw, targets_dh), axis=-1)
    return targets

def evaluate_box_proposals(pred_boxes, pred_scores, pred_labels,
                           gt_boxes, gt_labels, img_size_t, img_size,
                           num_classes, iou_threshold, show=False):
    '''
    Evaluate mAP, precision, recall on given results.
    pred_boxes : [N], list, each item in pred_boxes shape : [M, 4]
    pred_scores : [N], list, each item in pred_scores shape : [M]
    pred_labels : [N], list, each item in pred_labels shape : [M]
    gt_boxes : [N], list, each item in gt_boxes shape : [M, 4]
    gt_labels : [N], list, each item in gt_labels shape : [M, 4]
    img_size_t : [N], list, each item in img_size_t shape : [M, 2], means the transformed size of the image.
    img_size : [N], list, each item in img_size shape : [M, 2], means the original size of each image
    '''
    img_ids = []
    for i in range(len(gt_boxes)):
        ind = gt_labels[i] != -1
        gt_boxes[i] = gt_boxes[i][ind]
        gt_labels[i] = gt_labels[i][ind] 
    for i in range(len(pred_boxes)):
        img_ids.extend([i] * pred_scores[i].shape[0])
    
    img_ids = np.array(img_ids)
    num_img = len(pred_scores)
    
    pred_boxes = np.concatenate(pred_boxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    
    aps, recs, precs = AvarageMeter(), AvarageMeter(), AvarageMeter()
    #Evaluate for evary class
    for c in range(num_classes):
        # get prediction for class c
        index_c = pred_labels == c
        pred_boxes_c = pred_boxes[index_c]
        pred_scores_c = pred_scores[index_c]
        pred_labels_c = pred_labels[index_c]
        img_ids_c = img_ids[index_c]
        if(len(pred_boxes_c) == 0):
#             ap.update(0.0)
            continue
        
        # get gt information for class c
        npos = 0
        gt_c = {}
        for i in range(num_img):
            index_c = gt_labels[i] == c
            gt_boxes_c = gt_boxes[i][index_c]
            gt_labels_c = gt_labels[i][index_c]
            npos += len(gt_labels_c)
            det = [False] * len(gt_labels_c)
            gt_c[i] = (gt_boxes_c, gt_labels_c, det)
        
        sorted_ind = np.argsort(-pred_scores_c)
        pred_boxes_c = pred_boxes_c[sorted_ind, :]
        pred_scores_c = pred_scores_c[sorted_ind]
        pred_labels_c = pred_labels_c[sorted_ind]
        ids_sorted = [img_ids_c[x] for x in sorted_ind]
        
        nd = len(ids_sorted)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        
        for d in range(nd):
            gt_boxes_c, gt_labels_c, det = gt_c[ids_sorted[d]]
            bb = pred_boxes_c[d]
            ovmax = -np.inf
            
            #Compute iou
            if(gt_boxes_c.size > 0):
                ixmin = np.maximum(gt_boxes_c[:, 0], bb[0])
                iymin = np.maximum(gt_boxes_c[:, 1], bb[1])
                ixmax = np.minimum(gt_boxes_c[:, 2], bb[2])
                iymax = np.minimum(gt_boxes_c[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1, 0)
                ih = np.maximum(iymax - iymin + 1, 0)
                inters = iw * ih
                area_pred = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                area_gt = (gt_boxes_c[:, 2] - gt_boxes_c[:, 0] + 1) * (gt_boxes_c[:,3] - gt_boxes_c[:, 1])
                iou = inters / (area_pred + area_gt - inters)
                ovmax = np.max(iou)
                jmax = np.argmax(iou)
            
            if(ovmax > iou_threshold):
                if not det[jmax]:
                    tp[d] = 1.
                    det[jmax] = True
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / (float(npos) + np.spacing(1))
        prec = tp / np.maximum(tp + fp, np.spacing(1))
        ap = voc_ap(rec, prec, use_11_points=False)
        aps.update(ap)
        recs.update(rec[-1])
        precs.update(prec[-1])
        if show is True:
            print('Class: {} Recall: {:4f}, Precision: {:.4f}, AP:{:.4f}'.format(c, rec[-1], prec[-1], ap))
    
    return recs.get_avg(), precs.get_avg(), aps.get_avg()

