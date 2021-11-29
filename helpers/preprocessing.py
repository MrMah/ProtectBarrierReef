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

def sample_images(images_src, images_dest, labels_df, n_images, perc_annotated=0.3):
    no_annotations = labels_df.loc[(labels_df.video_id == int(images_src[-1])) & (labels_df.n_annotations == 0)]
    annotations = labels_df.loc[(labels_df.video_id == int(images_src[-1])) & (labels_df.n_annotations > 0)]
    
    n_annotated = np.round(n_images * perc_annotated, 0)
    n_not_annotated = n_images - n_annotated

    # Random sample from non-annotated 
    for image in random.sample(os.listdir(images_src), n_annotated):
        print(image)
        source = os.path.join(images_src, image)
        destination = os.path.join(images_dest, image)
        shutil.copy(source, destination)
        
    # Random sample from annotated