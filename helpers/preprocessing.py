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
        
    # Rename images to include video origin
    for image in os.listdir(images_dest):
        os.rename(os.path.join(images_dest, image), os.path.join(images_dest, 'video_0_'+image))
    print('COPY AND RENAME SUCCESSFUL')
        
def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_path(row, ROOT_DIR, IMAGE_DIR, LABEL_DIR):
    row['old_image_path'] = f'{ROOT_DIR}/train_images/video_{row.video_id}/{row.video_frame}.jpg'
    row['image_path'] = f'{IMAGE_DIR}/video_{row.video_id}_{row.video_frame}.jpg'
    row['label_path'] = f'{LABEL_DIR}/video_{row.video_id}_{row.video_frame}.txt'
    return row
        
def yolo_formatting(labels_df: pd.DataFrame, image_width: int=1280, image_height: int=720) -> pd.DataFrame:  
    """Convert df of bbox annotations from competition data to yolo format

    Args:
        labels_df (pd.DataFrame): annotated bbox with one image per row. Annotations column need to be preprocessed in [[x, y, width, heigh]] format
        image_width (int, optional): image px width. Defaults to 1280.
        image_height (int, optional): image px height. Defaults to 720.

    Returns:
        pd.DataFrame: DF with: path to label, x, y, width, and height in separate columns
    """
    # One row per bbox
    df = labels_df.copy()
    idx = []
    bboxes = []

    for i,row in enumerate(df.bboxes):
        if len(row) == 0:
            idx.append(i)
            bboxes.append([0,0,0,0])
        else:
            for list in row:
                idx.append(i)
                bboxes.append(list)
    
    # Convert to dataframe
    yolo_df = df.iloc[idx,:]
    yolo_df['bbox'] = bboxes
    
    tmp = pd.DataFrame(yolo_df['bbox'].to_list(), columns = ['x', 'y', 'width','height']).reset_index()
    final = pd.concat([yolo_df.reset_index(), tmp], axis = 1)
    
    # Convert x and y to x_center and y_center
    final['x'] = np.round((final['x'] + final['width']/2)/image_width,5)
    final['y'] = np.round((final['y'] + final['height']/2)/image_height,5)
    final['width'] = np.round(final['width']/image_width,5)
    final['height'] = np.round(final['height']/image_height,5)
    
    return final.loc[:,['label_path','x','y','width', 'height']]