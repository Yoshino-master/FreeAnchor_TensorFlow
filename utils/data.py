import os, pickle, copy
# import cv2
import numpy as np
from PIL import Image
from utils.data_aug import clamp_bboxs, convert_bboxs
from args import train_transforms, val_transforms

def parse_lines(lines):
    '''
    Given lines from train/val annotation file, return img_paths, bboxs and labels of every img.
    line format: img_path, xmin, ymin, xmax, ymax, label, ... (bboxs in the format of [xmin, ymin, xmax, ymax, label])
    return :
        img_paths : string, path of each img
        bboxs : list, elements in the list are in the shape of [N, 4], N is bbox nums of each img, 
            4 stands for [xmin, ymin, xmax, ymax] respectively
        labels : list, elements in the the are in the shape of [N]
        bboxs_num : Numpy.array object, bboxs num of each image
    '''
    idxs, img_paths, bboxs, labels, bboxs_num = [], [], [], [], []
    for line in lines:
        if isinstance(line, str) is False:
            try:
                line = line.decode('utf-8')
            except:
                raise Exception('Error accur when parse the annotation. There may be some special characters that are not included in utf-8 in your annotation file')
        s = line.strip().split(' ')
        idxs.append(int(s.pop(0)))
        img_paths.append(s.pop(0))
        if( len(s) == 0):
            idxs.pop(-1)
            img_paths.pop(-1)
            continue
        assert len(s) % 5 == 0,  'Annotation Error. Please check your annotation files. Make sure the completeness of bboxs and ensure the img paths and labels in the file'
        num_bbox = len(s) // 5
        bbox, label = [], []
        for i in range(num_bbox):
            bbox.append([float(s[ i * 5 ]), float(s[ i * 5 + 1 ]), float(s[ i * 5 + 2 ]), float(s[ i * 5 + 3 ])])      #xmin, ymin, xmax, ymax, respectively
            label.append(int(s[ i * 5 + 4 ]))
        if(len(bbox) > 35):
            bbox = bbox[:35]
            label = label[:35]
        bboxs.append(np.array(bbox, np.float32))
        labels.append(np.array(label, np.int64))
        bboxs_num.append(len(bbox))
            
    return np.array(idxs), img_paths, bboxs, labels, np.array(bboxs_num)

def imgs_loader(img_paths, root=None):
    '''
    Given a list of img_paths (or a single img path), return the list of imgs (PIL.Image object) and their sizes
    Note : If your model need images with BGR format, please uncomment the line : imgs[-1] = Image.fromarray(cv2.cvtColor(np.asarray(imgs[-1]),cv2.COLOR_RGB2BGR))
    return :
        imgs : list, PIL.Image object
        img_size : list, img sizes
    '''
    root = '' if root is None else root
    if isinstance(img_paths, list) is False:
        img_paths = [img_paths]
    imgs, img_size = [], []
    for img_path in img_paths:
        imgs.append(Image.open(os.path.join(root, img_path)).convert('RGB'))
#         imgs[-1] = Image.fromarray(cv2.cvtColor(np.asarray(imgs[-1]),cv2.COLOR_RGB2BGR))
        img_size.append(list(imgs[-1].size))
    return imgs, img_size

#return images, bboxs, labels, img_size, img_idxs
def get_batch_data(lines, bbox_mode, mode='train'):
    '''
    Function for getting batch data, receive annotation lines and return images, boxes, labels, labels and idxs
    '''
    bbox_mode = bbox_mode.decode()
    
    idxs, img_paths, bboxs, labels, bboxs_num = parse_lines(lines)
    imgs, img_size = imgs_loader(img_paths)
    if bbox_mode == 'xywh':                                          #if the bboxs is xywh format, then change it to xyxy format
        bboxs = [convert_bboxs(box, 'xyxy') for box in bboxs]
    bboxs = clamp_bboxs(bboxs, img_size, remove_empty=True)
    
    transform = train_transforms if mode == 'train' else val_transforms
    for transform in train_transforms:
        imgs, bboxs = transform(imgs, bboxs)
    imgs, batch_img_size = imgs
    bboxs, bboxs_num = bboxs
    batch_labels = np.ones((bboxs.shape[:2]), dtype=np.float32) * -1
    for batch_label, label in zip(batch_labels, labels):
        batch_label[:len(label)] = label
    return imgs, bboxs, batch_labels, batch_img_size.astype(np.int64), \
        np.array(img_size, dtype=np.int64), bboxs_num.astype(np.int64), idxs.astype(np.int64)
    
