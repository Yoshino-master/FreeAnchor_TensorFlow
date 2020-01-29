#-*- coding:utf-8 -*-
import argparse
import os, random
import numpy as np
import tensorflow as tf
from utils import data_aug

def get_configs():
    parser = argparse.ArgumentParser()
    
    #Some paths
#     parser.add_argument('--model_path', default=r'./weights/your_model_name/')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--save_path', default=r'./weights/train_model/model')
    parser.add_argument('--train_file', type=str, default=r'your/dataset/path/annotations\instances_train.txt')
    parser.add_argument('--val_file', type=str, default=r'your/dataset/path/annotations\instances_val.txt')
    parser.add_argument('--test_file', type=str, default=r'your/dataset/path/annotations\instances_test.txt')
    parser.add_argument('--log_path', type=str, default=r'./logs/')
    parser.add_argument('--progress_log_path', type=str, default=r'./logs/progress.log')
    
    #Basic params
    parser.add_argument('--mode', type=str, choices=['trian', 'test'], default='test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--val_evaluation_epoch', type=int, default=1)
    parser.add_argument('--lr_init', type=float, default=0.01)
    parser.add_argument('--lr_type', type=str, default='cosine_decay_restart')
    parser.add_argument('--train_evaluate_step', type=int, default=6)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=999)
    parser.add_argument('--optimizer_name', type=str, default='sgd')
    
    #Params for detection task
    parser.add_argument('--bbox_mode', default=r'xywh', choices=[r'xyxy', r'xywh'])
    parser.add_argument('--num_classes', type=int, default=80)                     #Background class is not counted in num_classes
    parser.add_argument('--transform_list', type=list, default=['MultiScaleResizes', 'RandomHorizontalFlip', 'Normalization', 'Assemble'])
    parser.add_argument('--transform_params', type= list, default=[None, None, [[102.9801, 115.9465, 122.7717],[1.0, 1.0, 1.0]], [32]])
    parser.add_argument('--val_transform_list', type=list, default=['MultiScaleResizes', 'Normalization', 'Assemble'])
    parser.add_argument('--val_transform_params', type=list, default=[None, [[102.9801, 115.9465, 122.7717],[1.0, 1.0, 1.0]], [32]])
    parser.add_argument('--nms_iou_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--pre_nms_thresh', type=float, default=0.05)
    parser.add_argument('--bbox_threshold', type=float, default=0.6)
    parser.add_argument('--pre_anchor_topk', type=int, default=100)
    args = parser.parse_args()
    return args

class init_config():
    def __init__(self):
        self.configs = get_configs()      #get base args
           
        # try to get nums of data in trainset, valset and testset
        self.configs.trainset_num = len(open(self.configs.train_file, 'r', encoding='utf-8').readlines()) if getattr(self.configs, 'train_file', None) is not None else 0
        self.configs.valset_num =  len(open(self.configs.val_file, 'r', encoding='utf-8').readlines()) if getattr(self.configs, 'val_file', None) is not None else 0
        self.configs.testset_num = len(open(self.configs.test_file, 'r', encoding='utf-8').readlines()) if getattr(self.configs, 'test_file', None) is not None else 0
        
        try:    # set random seeds of tensorflow, random, numpy and os
            self.init_random_seed(self.configs.random_seed)
        except:
            pass
        self.init_other_params()
        self.configs.train_transforms = self.init_transforms(self.configs.transform_list, self.configs.transform_params)
        self.configs.val_transforms = self.init_transforms(self.configs.val_transform_list, self.configs.val_transform_params)
        
    def init_random_seed(self, seed):
        ''' Initialize random seed to ensure the result reproductable '''
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONASHSEED'] = str(seed)
        tf.set_random_seed(seed)
    
    def init_other_params(self):
        ''' Initialize params that may not so interest for training or test '''
        self.configs.fpn_outchannels = 256
        self.configs.num_fpnout = 5
        self.configs.anchor_sizes = (32, 64, 128, 256, 512)
        self.configs.aspect_ratios = (0.5, 1.0, 2.0)
        self.configs.scales_per_octave = 3
        self.configs.anchor_strides = (8, 16, 32, 64, 128)
        self.configs.straddle_thresh = -1
        self.configs.octave = 2.0
        self.configs.scales_per_octave = 3
        self.configs.nums_anchors = len(self.configs.aspect_ratios) * self.configs.scales_per_octave
        self.configs.bbox_reg_weight, self.configs.bbox_reg_beta = 0.75, 0.11
        self.configs.focal_loss_alpha = 0.5
        self.configs.focal_loss_gamma = 2.0
        self.configs.prefetch_buffer = 2
        self.configs.pre_nms_top_n = 1000
        self.configs.fpn_post_nms_top_n = 100
    
    def get_configs(self):
        return self.configs
    
    def init_transforms(self, transform_list, transform_params):
        '''
        Initialize transforms of data. 4 transformer are used, they are:'MultiScaleResizes', 'RandomHorizontalFlip', 'Normalization', 'Assemble', respectively.
        '''
        transforms = []
        for transform_name, transform_param in zip(transform_list, transform_params):
            if transform_param is not None:
                transforms.append(getattr(data_aug, transform_name)(*transform_param))
            else:
                transforms.append(getattr(data_aug, transform_name)())
        return transforms

inited_config = init_config()
configs = inited_config.get_configs()                 #initialize the configs
train_transforms = configs.train_transforms
val_transforms = configs.val_transforms
