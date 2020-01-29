import os, sys, logging
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from args import configs
from model import FreeAnchor
from utils.data import get_batch_data
from utils.misc import config_optimizer, config_learning_rate, AvarageMeter, make_summary
from utils.evals import evaluate_box_proposals

def _train():
    #Set loggers
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=configs.progress_log_path, filemode='w')
    
    #Build up pipelines
    trainset = tf.data.TextLineDataset(configs.train_file)
    trainset = trainset.shuffle(configs.trainset_num, seed=configs.random_seed)
    trainset = trainset.batch(configs.batch_size)
    trainset = trainset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, configs.bbox_mode, 'train'],
                             Tout=[tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64]),
        num_parallel_calls=configs.num_threads)
    trainset = trainset.prefetch(configs.prefetch_buffer)
    
    valset = tf.data.TextLineDataset(configs.val_file)
    valset = valset.batch(configs.batch_size)
    valset = valset.map(
        lambda x: tf.py_func(get_batch_data,
                        inp=[x, configs.bbox_mode, 'test'],
                        Tout=[tf.float32, tf.float32, tf.float32, tf.int64, tf.int64, tf.int64, tf.int64]),
        num_parallel_calls=configs.num_threads)
    valset = valset.prefetch(configs.prefetch_buffer)
    
    iterator = tf.data.Iterator.from_structure(trainset.output_types, trainset.output_shapes)
    train_init_op = iterator.make_initializer(trainset)
    val_init_op = iterator.make_initializer(valset)
    images, bboxs, labels, batch_img_size, img_size, bboxs_num, img_idxs = iterator.get_next()
    
    images.set_shape([None, None, None, 3])
    bboxs.set_shape([None, None, 4])
    labels.set_shape([None, None])
    batch_img_size.set_shape([None, 2])
    img_size.set_shape([None, 2])
    bboxs_num.set_shape([None])
    img_idxs.set_shape([None])
    
    #Build model
    free_anchor_model = FreeAnchor(configs)
    with tf.variable_scope('free_anchor'):
        box_cls, box_regression, anchors = free_anchor_model.forward(images, bboxs, labels, img_size, bboxs_num, is_training=True)
        boxes_sel, scores_sel, labels_sel = free_anchor_model.get_prediction(anchors, box_cls, box_regression, batch_img_size)
        loss = free_anchor_model.compute_loss(anchors, box_cls, box_regression, bboxs, labels, img_size, bboxs_num)
    
    #Build optimizer
    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    lr = config_learning_rate(configs, global_step)
    optimizer = config_optimizer(configs.optimizer_name, lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    #Cut gradient to aviod gradient explosion
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(loss)
        clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)
    
    #Set summary and savers
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('train/learning_rate', lr)
    saver = tf.train.Saver()
    saver_best = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if configs.model_path is not None:
            model_file = tf.train.latest_checkpoint(configs.model_path)
            saver.restore(sess, model_file)                                #restore from checkpoint
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(configs.log_path, sess.graph)
        
        #Start to train
        best_map = -np.inf
        print('**************start to train**************')
        for epoch in range(configs.epochs):
            sess.run(train_init_op)
            loss_rec = AvarageMeter()
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, \
                img_size_ts, img_size_os = [], [], [], [], [], [], []
            
            for i in range(configs.testset_num // configs.batch_size):
                [_, summary, l, global_step_, pred_box, pred_score, pred_label, gt_box, gt_label, img_size_t, img_size_o] = \
                    sess.run([train_op, merged, loss, global_step, boxes_sel, \
                    scores_sel, labels_sel, bboxs, labels, batch_img_size, img_size])
                
                writer.add_summary(summary, global_step=global_step_)
                loss_rec.update(l, n=configs.batch_size)
                pred_boxes.extend(pred_box)
                pred_scores.extend(pred_score)
                pred_labels.extend(pred_label)
                gt_boxes.extend(gt_box)
                gt_labels.extend(gt_label)
                img_size_ts.extend(img_size_t)
                img_size_os.extend(img_size_o)
                 
                #Show information for every train_evaluate_step
                if global_step_ % configs.train_evaluate_step == 0 and global_step_ > 0:
                    rec, prec, mAP = evaluate_box_proposals(pred_boxes, pred_scores, pred_labels, gt_boxes, 
                        gt_labels, img_size_ts, img_size_os, configs.num_classes, configs.iou_threshold, show=False)
                    info = 'Epoch: {}, global_step: {}, | loss_total: {:.3f}, loss_last: {:.3f}, rec: {:.3f}, prec: {:.3f}, map: {:.3f}'.format( \
                        epoch, global_step_, loss_rec.get_avg(), loss_rec.val, rec, prec, mAP)
                    print(info)
                    logging.info(info)
                    
                    writer.add_summary(make_summary('evaluation/train_batch_recall', rec), global_step=global_step_)
                    writer.add_summary(make_summary('evaluation/train_batch_precision', prec), global_step=global_step_)
                    
                    if np.isnan(loss_rec.get_avg()):
                        print('****************split line****************')
                        raise ArithmeticError('Gradient exploded! Please change some hyper-parameters or initialition mode of your model')
                     
                    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, \
                        img_size_ts, img_size_os = [], [], [], [], [], [], []
                
            #Save per save_epoch
            if (epoch + 1) % configs.save_epoch == 0:
                saver.save(sess, configs.save_path + 'epoch_{}-step_{}-loss{:.4f}-map_{:.4f}'.format(epoch, int(global_step_), loss_rec.get_avg(), mAP))
            
            #Evaluate on valset per val_evaluation_epoch
            if (epoch + 1) % configs.val_evaluation_epoch == 0:
                sess.run(val_init_op)
                val_loss = AvarageMeter()
                pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, \
                    img_size_ts, img_size_os = [], [], [], [], [], [], []
                for i in range(configs.testset_num // configs.batch_size):
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
                    
                rec, prec, mAP = evaluate_box_proposals(pred_boxes, pred_scores, pred_labels, gt_boxes, 
                    gt_labels, img_size_ts, img_size_os, configs.num_classes, configs.iou_threshold, show=True)
                info = 'Val epoch: {}, | loss_total: {:.3f}, rec: {:.3f}, prec: {:.3f}, map: {:.3f}'.format( \
                    epoch, val_loss.get_avg(), rec, prec, mAP)
                print(info)
                logging.info(info)
                
                #Save the best model
                if mAP > best_map:
                    best_map = mAP
                    saver_best.save(sess, configs.save_path + 'best_model-epoch_{}-loss_{:.4f}-map_{:.4f}'.format( \
                        epoch, val_loss.get_avg(), mAP))
                
                writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_recall', rec), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', prec), global_step=epoch)
                writer.add_summary(make_summary('evaluation/total_loss', val_loss.get_avg()), global_step=epoch)
           
    print('finish')


if __name__ == '__main__':
    _train()
