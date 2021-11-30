#Preprocessing helpersi
import os
import random
import shutil
import pandas as pd
import numpy as np

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