import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
try:
    import accimage
except ImportError:
    accimage = None
try:
    import cv2
except ImportError:
    cv2 = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def convert_bboxs(bboxs, to_mode):
    '''
    Change the boxes mode from xyxy to xywh, or from xywh to xyxy
    '''
    TO_REMOVE = 1
    for i in range(len(bboxs)):
        if to_mode == 'xyxy':
            xmin, ymin, w, h = bboxs[i]
            bboxs[i] = [xmin, ymin, xmin + w - TO_REMOVE, ymin + h - TO_REMOVE]
        if to_mode == 'xywh':
            xmin, ymin, xmax, ymax = bboxs[i]
            bboxs[i] = [xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE]
    return bboxs

def clamp_bboxs(bboxs, img_size, remove_empty=True):
    '''
    Clamp boxes to make sure that coordinates won't out of range
    '''
    TO_REMOVE = 1
    for i in range(len(bboxs)):
        bbox = bboxs[i]
        bbox[:, 0].clip(min=0, max=img_size[i][0] - TO_REMOVE)
        bbox[:, 1].clip(min=0, max=img_size[i][1] - TO_REMOVE)
        bbox[:, 2].clip(min=0, max=img_size[i][0] - TO_REMOVE)
        bbox[:, 3].clip(min=0, max=img_size[i][1] - TO_REMOVE)
        if remove_empty:
            keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
            bboxs[i] = bbox[keep]
        else:
            bboxs[i] = bbox
    return bboxs

def resize_bboxs(bboxs, size_o, size_t):
    '''
    Resize Boxes according to the change of image size
    params:
        bboxs : numpy.array object, shape:[N * 4]
        size_o : origin size of the image, except tuple (or list) object
        size_t : transformed size of the image
    '''
    assert len(size_o) == 2 and len(size_t) == 2          #Check the size of image and make sure that sizes contain only two elements (x, y).
    ratio = tuple( float(s_t) / float(s_o) for s_o, s_t in zip(size_o, size_t) )
    xmin, ymin, xmax, ymax = np.split(bboxs, 4, axis=1)
    bboxs = np.concatenate([xmin * ratio[0], ymin * ratio[1], xmax * ratio[0], ymax * ratio[1]], axis=1)
    return bboxs


class Resize(object):
    '''
    Resize images and bboxs according to min_size and max_size
    '''
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, bboxs=None):
        size_o = image.size
        size = self.get_size(size_o)
        image = F.resize(image, size)
        if bboxs is not None:
            bboxs = resize_bboxs(bboxs, size_o, image.size)
        return image, bboxs

class MultiScaleResize(object):
    def __init__(self, min_sizes=(800,), max_size=1333):
        self.resizers = []
        for min_size in min_sizes:
            self.resizers.append(Resize(min_size, max_size))

    def __call__(self, image, bboxs=None):
        resizer = random.choice(self.resizers)
        image, bboxs = resizer(image, bboxs)
        return image, bboxs

class MultiScaleResizes(object):
    def __init__(self, min_sizes=(800,), max_size=1333):
        self.multi_resizer = MultiScaleResize(min_sizes, max_size)
    
    def __call__(self, images, bboxs=None):
        if bboxs is None:
            bboxs = [] * len(images)
        for i, (image, bbox) in enumerate(zip(images, bboxs)):
            images[i], bboxs[i] = self.multi_resizer(image, bbox)
        return images, bboxs
        
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def random_horizonal_flip(self, image, bboxs=None):
        TO_REMOVE = 1
        width, height = image.size
        if random.random() < self.prob:
            image = F.hflip(image)
            if bboxs is None:
                return image, bboxs
            xmin, ymin, xmax, ymax = np.split(bboxs, 4, axis=1)
            xmin_t = width - TO_REMOVE - xmax
            xmax_t = width - TO_REMOVE - xmin
            ymin_t = ymin
            ymax_t = ymax
            bboxs = np.concatenate([xmin_t, ymin_t, xmax_t, ymax_t], axis=1)
        return image, bboxs
            
    def __call__(self, images, bboxs=None):
        if bboxs is None:
            bboxs = [] * len(images)
        for i, (image, bbox) in enumerate(zip(images, bboxs)):
            images[i], bboxs[i] = self.random_horizonal_flip(image, bbox)
        return images, bboxs

class Normalization(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std  = std
        self.to_bgr255 = to_bgr255
    
    def __call__(self, images, bboxs=None):
        if _is_pil_image(images[0]) is True:
            for i in range(len(images)):
                images[i] = np.array(images[i])
        assert isinstance(images[0], np.ndarray)              #Please make sure that images are in the format of PIL.Image or numpy.ndarry
        
        for i in range(len(images)):
            if self.to_bgr255 is True:
                if cv2 is not None:
                    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                else:
                    images[i] = np.expand_dims(images[i], axis=-1)
                    images[i] = np.concatenate([images[i][:,:,2], images[i][:,:,1], images[i][:,:,0]], axis=2)
            
            images[i] = np.subtract(images[i], self.mean)
            images[i] = np.divide(images[i], self.std)

        return images, bboxs

class Assemble(object):
    '''
    Assemble transformed data and padding batch data
    '''
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible
    
    def __call__(self, images, bboxs=None):
        max_size  = tuple(max(s) for s in zip(*[img.shape for img in images]))
        max_bboxs = max([len(box) for box in bboxs])
        if self.size_divisible > 0:
            stride = self.size_divisible
            max_size = list(max_size)
            max_size[0] = int(np.ceil(max_size[0] / stride) * stride)
            max_size[1] = int(np.ceil(max_size[1] / stride) * stride)
            max_size = tuple(max_size)
        
        batch_shape = (len(images),) + max_size
        assembled_images = np.zeros(batch_shape, dtype=np.float32)
        image_sizes = np.array([img.shape[:2][::-1] for img in images])
        for img, pad_img in zip(images, assembled_images):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        
        boxes_shape = (len(bboxs), max_bboxs, 4)
        assembled_bboxs  = np.zeros(boxes_shape, dtype=np.float32)
        for box, pad_box in zip(bboxs, assembled_bboxs):
            pad_box[:len(box)] = box
        boxes_num = np.array([len(box) for box in bboxs])
        
        return (assembled_images, image_sizes), (assembled_bboxs, boxes_num)
        
