#! /bin/bash

# Using info from
# https://stackoverflow.com/questions/35350607/ffmpeg-add-text-frames-to-the-start-of-video
# https://stackoverflow.com/questions/22710099/ffmpeg-create-blank-screen-with-text-video
# https://stackoverflow.com/questions/8213865/ffmpeg-drawtext-over-multiple-lines

# FIRST USE ../render_SuppVid_SfM.py and ../render_SuppVid_SfM_nested.py to create the video files.
# THEN THIS FILE

# Generate intros 
ffmpeg -y -f lavfi -i color=size=1050x1050:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Structure-from-motion
One cylinder'" intro_0.mp4

ffmpeg -y -f lavfi -i color=size=1050x1050:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Structure-from-motion
Nested cylinders
Same speed'" intro_1.mp4

ffmpeg -y -f lavfi -i color=size=1050x1050:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Structure-from-motion
Nested cylinders
Fast inner cylinder'" intro_2.mp4

ffmpeg -y -f lavfi -i color=size=1050x1050:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Structure-from-motion
Nested cylinders
Fast outer cylinder'" intro_3.mp4

ffmpeg -y -f concat -i SfM_filelist.txt -c copy -fflags +genpts video_S4_Structure-from-motion.mp4
