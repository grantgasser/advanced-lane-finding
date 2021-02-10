import os
import glob

import cv2
from moviepy.editor import ImageSequenceClip

def write_images(images, video_folder):
    """Write images from list into folder for later use (to convert to video, etc.)"""
    video_images_files = []
    # if using glob, need to sort
    for image_idx, image in enumerate(images):
        # format and write image to folder
        file_str = str(image_idx).zfill(0)
        path = os.path.join(video_folder, file_str + '.png')

        try:
            cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # append to file list
            video_images_files.append(path)
        except:
            print('Done writing images..')
            return video_images_files

    return video_images_files


def make_video(video_images_files, name, fps=30):
    """Given list of image files, create video"""
    # create video
    print('\nCreating video...')
    clip = ImageSequenceClip(video_images_files, fps)
    
    # write video
    clip.write_videofile(name)
    return clip

def write_gif(clip, path, sub_start, sub_end):
    """Write subclip of clip as gif."""
    subclip = clip.subclip(sub_start, sub_end)
    subclip.write_gif(path)
    return subclip