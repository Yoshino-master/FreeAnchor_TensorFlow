#-*- coding:utf-8 -*-
import os, sys
sys.path.append(os.getcwd())
import tensorflow as tf
from tqdm import trange
from args import configs
from model import FreeAnchor
from utils.data import get_batch_data
from utils.evals import evaluate_box_proposals
from utils.misc import AvarageMeter

def _test():
    # tf.data pipeline for evaluating
    testset = tf.data.TextLineDataset(configs.test_file)
    testset = testset.shuffle(configs.testset_num, seed=configs.random_seed)
    testset = testset.batch(configs.batch_size)
    testset = testset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, configs.bbox_mode, 'test'],
                             Tout=[tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64]),
        num_parallel_calls=configs.num_threads)
    testset.prefetch(configs.prefetch_buffer)
    
    iterator = testset.make_one_shot_iterator()
    images, bboxs, labels, batch_img_size, img_size, bboxs_num, img_idxs = iterator.get_next()
    
    images.set_shape([None, None, None, 3])
    bboxs.set_shape([None, None, 4])
    labels.set_shape([None, None])
    batch_img_size.set_shape([None, 2])
    img_size.set_shape([None, 2])
    bboxs_num.set_shape([None])
    img_idxs.set_shape([None])
    
    # Define model
    free_anchor_model = FreeAnchor(configs)
    with tf.variable_scope('free_anchor'):
        box_cls, box_regression, anchors = free_anchor_model.forward(images, bboxs, labels, batch_img_size, bboxs_num, is_training=False)
        boxes_sel, scores_sel, labels_sel = free_anchor_model.get_prediction(anchors, box_cls, box_regression, batch_img_size)
        loss = free_anchor_model.compute_loss(anchors, box_cls, box_regression, bboxs, labels, batch_img_size, bboxs_num)
    
    #get prediction
    val_loss = AvarageMeter()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        model_file = tf.train.latest_checkpoint(configs.model_path)                #restore from checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, \
        img_size_ts, img_size_os = [], [], [], [], [], [], []
        
        print('\n-------------start to get prediction-------------\n')
        for i in trange(configs.testset_num // configs.batch_size):
            try:
                [pred_box, pred_score, pred_label, l, gt_box, gt_label, img_size_t, img_size_o] \
                    = sess.run([boxes_sel, scores_sel, labels_sel, loss, bboxs, labels, batch_img_size, img_size])
                
                pred_boxes.extend(pred_box)
                pred_scores.extend(pred_score)
                pred_labels.extend(pred_label)
                gt_boxes.extend(gt_box)
                gt_labels.extend(gt_label)
                img_size_ts.extend(img_size_t)
                img_size_os.extend(img_size_o)
                val_loss.update(l, configs.batch_size)
            
            except:
                print('error accur or finish on step_' + str(i))
                pass
        
        print('\n-------------start to evaluating results-------------\n')
        rec, prec, ap = evaluate_box_proposals(pred_boxes, pred_scores, pred_labels, gt_boxes, 
            gt_labels, img_size_ts, img_size_os, configs.num_classes, configs.iou_threshold)
        print('mAP: {:.4f}'.format(ap))
        print('overall rec: {:.4f}'.format(rec))
        print('overall prec: {:.4f}'.format(prec))
        print('overall loss: {:.4f}'.format(val_loss.get_avg()))
        print('finish')

if __name__ == '__main__':
    _test()
