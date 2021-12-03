import pandas as pd
import ast
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os

#Plot sample image
def draw_annotation_coco(dir: str, video_id: int, video_frame: int):
    """Draw annotation box on top of image with annotations in original coco format (xmin, ymin, width, height)
    Args:
        dir (str): basedir of files
        video_id (int): Id of videofile
        video_frame (int): Id of image
    """
    # Get annotations
    train = pd.read_csv(dir+'train.csv')
    annotation_str = train[(train.video_id == video_id) & (train.video_frame == video_frame)].annotations.values[0]
    annotations = ast.literal_eval(annotation_str)
    
    # Show image
    im = mpimg.imread(dir + 'train_images/video_' + str(video_id) +'/' + str(video_frame) + '.jpg')
    if len(annotations) == 0:
        plt.imshow(im)
    else:
        fig, ax = plt.subplots()
        for i,annotation in enumerate(annotations):
            rect = Rectangle((annotation['x'], annotation['y']), annotation['width'], annotation['height'], linewidth=1,
                                     edgecolor='r', facecolor="none")
            ax.add_patch(rect)
        plt.imshow(im)
    plt.show()
    
def draw_annotation_yolo(image_name: str, dir_images: str, dir_labels: str, image_width:int = 1280, image_height:int = 720) -> None:
    """Draw annotation box on top of image with annotations in normalized yolo format (xcenter, ycenter, width, height)

    Args:
        image_name (str): image name (e.g. 'video_0_15')
        dir_images (str): image folder
        dir_labels (str): label folder
        image_width (int, optional): image width. Defaults to 1280.
        image_height (int, optional): image height. Defaults to 720.
    """
    # Get labels
    label_src = os.path.join(dir_labels, image_name + '.txt')
    label = pd.read_csv(label_src, sep = " ", names = ["label","xcenter","ycenter","width","height"])
    label["xcenter"] = label["xcenter"] * image_width
    label["ycenter"] = label["ycenter"] * image_height
    label["width"] = label["width"]*image_width
    label["height"] = label["height"]*image_height
    label["x"] = label["xcenter"] - label["width"]/2
    label["y"] = label["ycenter"] - label["height"]/2
    
    # Draw image
    image_src = os.path.join(dir_images, image_name + '.jpg')
    im = mpimg.imread(image_src)
    
    if label.iloc[0].width == 0:
        plt.imshow(im)
    else:
        fig, ax = plt.subplots()
        for i,row in label.iterrows():
            rect = Rectangle((row['x'], row['y']), row['width'], row['height'], linewidth=1,
                                     edgecolor='r', facecolor="none")
            ax.add_patch(rect)
        plt.imshow(im)
    plt.show()
    