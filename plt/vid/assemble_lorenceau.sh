#! /bin/bash

# Using info from
# https://stackoverflow.com/questions/35350607/ffmpeg-add-text-frames-to-the-start-of-video
# https://stackoverflow.com/questions/22710099/ffmpeg-create-blank-screen-with-text-video
# https://stackoverflow.com/questions/8213865/ffmpeg-drawtext-over-multiple-lines

# FIRST RUN ../render_SuppVid_Lorenceau.py
# THEN THIS FILE

# Generate intros 
ffmpeg -y -f lavfi -i color=size=600x450:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Lorenceau illusion
No noise'" intro_0.mp4

ffmpeg -y -f lavfi -i color=size=600x450:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Lorenceau illusion
Low noise'" intro_1.mp4

ffmpeg -y -f lavfi -i color=size=600x450:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Lorenceau illusion
Medium noise'" intro_2.mp4

ffmpeg -y -f lavfi -i color=size=600x450:duration=5:rate=60:color=white -vf "drawtext=fontfile=/path/to/font.ttf:fontsize=30:fontcolor=black:x=(w-text_w)/2:y=(h-text_h)/2:text='Lorenceau illusion
High noise'" intro_3.mp4

ffmpeg -y -f concat -i lorenceau_filelist.txt -c copy -fflags +genpts video_S3_Lorenceau_combined.mp4
