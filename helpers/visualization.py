import pandas as pd
import ast
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

#Plot sample image
def draw_annotation(dir: str, video_id: int, video_frame: int):
    """Draw annotation box on top of image
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
    elif len(annotations) > 0:
        fig, ax = plt.subplots(len(annotations))
        for i,annotation in enumerate(annotations):
            rect = Rectangle((annotation['x'], annotation['y']), annotation['width'], annotation['height'], linewidth=1,
                                     edgecolor='r', facecolor="none")
            if len(annotations)>1:
                ax[i].imshow(im)
                ax[i].add_patch(rect)
            else:
                ax.imshow(im)
                ax.add_patch(rect)  
    plt.show()