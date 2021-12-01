#Preprocessing helpersi
import os
import ast
import random
import shutil
import pandas as pd
import numpy as np
import cv2

def count_n_annotations(labels_df: pd.DataFrame)->pd.Series:
    """Count n annotations per label

    Args:
        labels_df (pd.DataFrame): dataframe of labels

    Returns:
        pd.Series: n annotations
    """
    df = labels_df.copy()
    n_annotations=df['annotations'].apply(lambda x: str.count(x, 'x'))
    return n_annotations

def subsample_images(images_src: str, images_dest: str, labels_df: pd.DataFrame, n_images: int, perc_annotated:float=0.3) -> None:
    """Subsample images from video_0 folder and copy randomly selected to new images folder

    Args:
        images_src (str): path to raw image folder
        images_dest (str): path to destination folder
        labels_df (pd.DataFrame): df of annotated images
        n_images (int): number of total images to copy
        perc_annotated (float, optional): Percentage of n_images which are annotated. Defaults to 0.3.
    """
    no_annotations = labels_df.loc[(labels_df.video_id == int(images_src[-1])) & (labels_df.n_annotations == 0)]
    annotations = labels_df.loc[(labels_df.video_id == int(images_src[-1])) & (labels_df.n_annotations > 0)]
    
    n_annotated = int(np.round(n_images * perc_annotated, 0))
    n_not_annotated = n_images - n_annotated
    
    no_annotations_frames = [str(row) + '.jpg' for row in no_annotations.video_frame]
    annotations_frames = [str(row) + '.jpg' for row in annotations.video_frame]

    # Random sample from non-annotated 
    for image in random.sample(no_annotations_frames, n_not_annotated):
        source = os.path.join(images_src, image)
        destination = os.path.join(images_dest, image)
        shutil.copy(source, destination)
        
    # Random sample from annotated
    for image in random.sample(annotations_frames, n_annotated):
        source = os.path.join(images_src, image)
        destination = os.path.join(images_dest, image)
        shutil.copy(source, destination)
        
def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes
        
def yolo_formatting(labels_df: pd.DataFrame, labels_dest: str, train_images_src: str) -> None:
    train_subset = [int(image.replace('.jpg','')) for image in os.listdir(train_images_src)]
    
    train_labels = labels_df.loc[(labels_df.video_frame in train_subset),]   
    train_labels['annotations_lists'] = ast.literal_eval(train_labels['annotations'])
    
    train_labels.DataFrame(train_labels['annotations'].to_list(), columns = ['x', 'y', 'c3'])

        
    
    return


def voc2yolo(image_height, image_width, bboxes):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    
    return bboxes

def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes

def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normolizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes

def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes


def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, classes, class_ids, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
            
    elif bbox_format == 'coco':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))

                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)

    elif bbox_format == 'voc_pascal':
        
        for idx in range(len(bboxes)):  
            
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def get_path(row, ROOT_DIR, IMAGE_DIR, LABEL_DIR):
    row['old_image_path'] = f'{ROOT_DIR}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    row['image_path'] = f'{IMAGE_DIR}/video_{row.video_id}_{row.video_frame}.jpg'
    row['label_path'] = f'{LABEL_DIR}/video_{row.video_id}_{row.video_frame}.txt'
    return row