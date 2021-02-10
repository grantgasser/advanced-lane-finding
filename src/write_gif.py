import glob

from moviepy.editor import VideoFileClip

import postprocessing as post

clip = VideoFileClip('../output_videos/project_video.mp4')
post.write_gif(
    clip=clip, 
    path='../output_videos/project_video.gif',
    sub_start=8,
    sub_end=10
)