import os
import cv2
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

def change_coco_dataset(train_anno, val_anno, test_anno,
                         train_anno_c, val_anno_c, test_anno_c,
                         train_img_root, val_img_root, test_img_root):
    '''
    Change cocodataset annotation files  for object detection (including train,val,test) to the general format of object detection
    '''
    source_anns = [val_anno, test_anno, train_anno]
    target_anns = [val_anno_c, test_anno_c, train_anno_c]
    imgroots = [val_img_root, test_img_root, train_img_root]
    
    for i in range(len(source_anns)):
        source_ann = source_anns[i]
        f = open(target_anns[i], 'w')
        coco = COCO(source_ann)
        idxs = coco.getImgIds()
        
        json_category_id_to_contiguous_id = {
            v: j for j, v in enumerate(coco.getCatIds())}
        
        count = 0
        for idx in idxs:
            imgdir = os.path.join(imgroots[i], coco.imgs[idx]['file_name'])
            anno_idxs = coco.getAnnIds(idx)
            anns = coco.loadAnns(anno_idxs)
        
            line = str(count) + ' ' + \
                imgdir + ' ' + \
                ' '.join([' '.join([str(box) for box in ann['bbox']]) + ' ' + \
                str(json_category_id_to_contiguous_id[ann['category_id']]) \
                for ann in anns if ann['iscrowd']==0]) + '\n'
            if(len(line.strip().split(' ')) <= 2):                          #remove imgs without box
                continue
            f.write(line)
            count += 1
        f.close()

if __name__ == '__main__':
    train_anno = r'your/path/to/instances_train2017.json'
    val_anno = r'your/path/to/instances_val2017.json'
    test_anno = r'your/path/to/image_info_test2017.json'
    train_anno_c = r'your/path/to/target.txt'
    val_anno_c = r'your/path/to/target.txt'
    test_anno_c = r'your/path/to/target.txt'
    train_img_root = r'your/path/to/train_image_data'
    val_img_root = r'your/path/to/val_image_data'
    test_img_root = r'your/path/to/test_image_data'
    change_coco_dataset(train_anno, val_anno, test_anno,
                         train_anno_c, val_anno_c, test_anno_c,
                         train_img_root, val_img_root, test_img_root)

    
