from subprocess import call
import os

# Makes a subprocess call to ffmpeg command line tool, which converts a video
# file into a set of image files at the given frames per second
def video_to_images(videopath, output_dir, fps=15, output_h=480, output_w=640):
    # ffmpeg -i video.MP4 -qscale:v 2 -vf scale=640:480 -r 15 ca56_1/f%05d.png
    print 'Calling ffmpeg to convert video {}'.format(videopath)
    call(['ffmpeg -i {} -qscale:v 2 -vf scale={}:{} -r {} {}/f%05d.png'.format(
            videopath, output_w, output_h, fps, output_dir)], shell=True)

def list_contents(dir_path, keyword='.png'):
    output_list = []
    for filename in os.listdir(dir_path):
        if keyword in filename:
            output_list.append(filename)

    # write the list.txt file
    with open(dir_path + '/' + 'list.txt', 'w') as f:
        for line in output_list:
            f.write(line + '\n')

if __name__ == '__main__':

    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    raw_video_dir = os.path.join(package_dir, 'validation/raw')
    output_dir = os.path.join(package_dir, 'validation')

    output_width = 640
    output_height = 360
    output_fps = 15

    for video_fname in os.listdir(raw_video_dir):
        if '.mp4' in video_fname.lower():
            print 'Beginning to process ' + video_fname
            save_dir = os.path.join(output_dir, video_fname.lower().split('.mp4')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            video_to_images(os.path.join(raw_video_dir, video_fname), save_dir, 
                            output_fps, output_height, output_width)
            list_contents(save_dir)
            print 'Done processing ' + video_fname

