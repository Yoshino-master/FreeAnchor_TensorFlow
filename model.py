import tensorflow as tf
from utils.layers import conv2d, stem
from utils.layers import bottleneck, upsample_layer
from utils.layers import retinanet_head, compute_single_feature_map
from utils.layers import select_bboxs
from utils.anchors import get_cell_anchors, anchor_generator
from utils.loss import FreeAnchorLoss

class Resnet50FPN(object):
    def __init__(self, fpn_outchannels):
        self.fpn_outchannels = fpn_outchannels
    
    def forward(self, inputs, is_training=True):
        with tf.variable_scope('body'):
            with tf.variable_scope('stem'):
                stem_out = stem(inputs, 64, is_training)
            stages = self.resnet50(stem_out, is_training)
        with tf.variable_scope('fpn'):
            features = self.fpn(stages, self.fpn_outchannels)
        return features
    
    def resnet50(self, inputs, is_training):
        stage_countlist = [3, 4, 6, 3]                   #nums of bottleneck for each stage of resnet50 body
        
        with tf.variable_scope('stage1'):
            stage1_out = inputs
            in_channel, bottleneck_channels, out_channel, stride = 64, 64, 256, 1
            for _ in range(stage_countlist[0]):
                stage1_out = bottleneck(stage1_out, in_channel, bottleneck_channels, out_channel, stride_in_1x1=True, stride=stride, is_training=is_training)
                in_channel = out_channel
                stride = 1
        
        with tf.variable_scope('stage2'):
            stage2_out = stage1_out
            in_channel, bottleneck_channels, out_channel, stride = 256, 128, 512, 2
            for _ in range(stage_countlist[1]):
                stage2_out = bottleneck(stage2_out, in_channel, bottleneck_channels, out_channel, stride_in_1x1=True, stride=stride, is_training=is_training)
                in_channel = out_channel
                stride = 1
        
        with tf.variable_scope('stage3'):
            stage3_out = stage2_out
            in_channel, bottleneck_channels, out_channel, stride = 512, 256, 1024, 2
            for _ in range(stage_countlist[2]):
                stage3_out = bottleneck(stage3_out, in_channel, bottleneck_channels, out_channel, stride_in_1x1=True, stride=stride, is_training=is_training)
                in_channel = out_channel
                stride = 1
        
        with tf.variable_scope('stage4'):
            stage4_out = stage3_out
            in_channel, bottleneck_channels, out_channel, stride = 1024, 512, 2048, 2
            for _ in range(stage_countlist[3]):
                stage4_out = bottleneck(stage4_out, in_channel, bottleneck_channels, out_channel, stride_in_1x1=True, stride=stride, is_training=is_training)
                in_channel = out_channel
                stride = 1
            
        return stage1_out, stage2_out, stage3_out, stage4_out
    
    def fpn(self, features, out_channel):
        last_inner = conv2d(features[-1], out_channel, 1)                             #Set up the last (except the top layer) fpn output. In additional, the first element of the resnet outputs will not be used in fpn
        results = []
        results.append(conv2d(last_inner, out_channel, 3, 1, padding=1))
        assert len(features)-2 > 0           #Please make sure that at least 3 features are feed in the fpn
        for i in range(len(features) - 2):
            inner_top_down = upsample_layer(last_inner, 2)
            inner_lateral = conv2d(features[-2-i], out_channel, 1)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, conv2d(last_inner, out_channel, 3, strides=1, padding=1))
        with tf.variable_scope('top_blocks'):
            extra_feas = self.fpn_extra_tops(results[-1], out_channel)
        results.extend(extra_feas)
        return results
        
    def fpn_extra_tops(self, inputs, out_channels):                                  #Add extra 2 feature maps of fpn
        f1 = conv2d(inputs, out_channels, 3, strides=2, padding=1)
        f2 = conv2d(tf.nn.relu(f1), out_channels, 3, strides=2, padding=1)
        return [f1, f2]

class AnchorGenerator(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cell_anchors = get_cell_anchors(
            self.cfg.anchor_sizes, self.cfg.aspect_ratios, self.cfg.anchor_strides, self.cfg.straddle_thresh, self.cfg.octave, self.cfg.scales_per_octave)
        self.cell_anchors = [tf.constant(cell.tolist(), dtype=tf.float32) for cell in self.cell_anchors]
    
    def forward(self, features):
        anchors = anchor_generator(features, self.cfg.anchor_strides, self.cell_anchors)
        return anchors

class FreeAnchor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.body = Resnet50FPN(self.cfg.fpn_outchannels)
        self.head = retinanet_head
        self.anchor_generator = AnchorGenerator(self.cfg)
        self.loss = FreeAnchorLoss(self.cfg)

    def forward(self, images, bboxs, batch_labels, batch_img_size, bboxs_num, is_training=True):
        with tf.variable_scope('backbone'):
            features = self.body.forward(images, is_training)
        with tf.variable_scope('rpn'):
            box_cls, box_regression = self.head(features, self.cfg.nums_anchors, self.cfg.num_classes, self.cfg.fpn_outchannels)
        anchors = self.anchor_generator.forward(features)
        # box_cls, box_regression and anchors consist of 5 blocks, sacle=[ 1,2,4,8,16 ]
        # box_cls : [ N, feature_h / scale, feature_w / scale, nums_anchors * nums_classes ]
        # box_regression : [ N, feature_h / scale, feature_w / scale, nums_anchors * 4 ]
        # anchors : [ N, feature_h / scale, feature_w / scale, nums_anchors * 4 ]
        return box_cls, box_regression, anchors
    
    def compute_loss(self, anchors, box_cls, box_regression, bboxs, batch_labels, batch_img_size, bboxs_num):
        loss= self.loss.forward(anchors, box_cls, box_regression, bboxs, batch_labels, batch_img_size, bboxs_num)
        return loss
    
    def get_prediction(self, anchors, box_cls, box_regression, batch_img_size):
        #decoder box_regression to bbox
        sampled_boxes, labellists, scorelists = [], [], []
        for i, (cls, reg) in enumerate(zip(box_cls, box_regression)):
            detections, labels, scores = compute_single_feature_map(anchors[i], cls, reg, batch_img_size, self.cfg.pre_nms_thresh,
                self.cfg.batch_size, self.cfg.nums_anchors, self.cfg.num_classes)
            
            sampled_boxes.append(detections)
            labellists.append(labels)
            scorelists.append(scores)
        
        boxlists = list(zip(*sampled_boxes))
        labellists = list(zip(*labellists))
        scorelists = list(zip(*scorelists))
        boxlists = [tf.concat(box, axis=0) for box in boxlists]
        labellists = [tf.concat(label, axis=0) for label in labellists]
        scorelists = [tf.concat(score, axis=0) for score in scorelists]
        
        #implement nms over boxes
        boxes_sel, scores_sel, labels_sel = select_bboxs(boxlists, labellists, scorelists, self.cfg.num_classes,
            self.cfg.nms_iou_threshold, self.cfg.fpn_post_nms_top_n)
        
        return boxes_sel, scores_sel, labels_sel

