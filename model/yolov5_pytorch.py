# Preprocessing helpers
import os

def yolo_formatting(train_image_src, label_src, n):
    for video_folder in os.listdir(train_image_src):
        for image in os.listdir(train_image_src + '/' + video_folder):
            print(image)
    
    return

if __name__ == '__main__':
    yolo_formatting('datasets/images','t')