import cv2
import os
import argparse

def video_to_images(input_video, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    vidcap = cv2.VideoCapture(input_video)
    success,image = vidcap.read()
    cv2.imwrite(save_dir + '/IMG_1.jpg', image)
    count = 2
    success = True
    while success:
      success,image = vidcap.read()
      if image is None:
          break
      path = save_dir + '/IMG_' + str(count) + ".jpg"
      print(path)
      cv2.imwrite(path, image)     # save frame as JPEG file
      if cv2.waitKey(10) == 27:                     
          break
      count += 1  




def images_to_video(args,frame_dir,output_video):
    fps = 30  # 帧率
    img_array = []
    img_width = args.frame_width # 根据帧图像的大小进行修改
    img_height = args.frame_height # 根据帧图像的大小进行修改
    imdir = sorted(os.listdir(frame_dir))
    imdir.sort(key=lambda x: int(x.split('.')[0][4:]))
    print(len(imdir))
    for idx in range(len(imdir)):
        imgname = frame_dir + '/' + imdir[idx]
        #print(imgname)
        img = cv2.imread(imgname)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        img_array.append(img)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(fourcc)
    out = cv2.VideoWriter(output_video, fourcc, fps,(img_width,img_height))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()
 
 
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--frame_dir", default=None, type=str, 
                        help="The dir which saves frames")
    parser.add_argument("--input_video", default=None, type=str, 
                        help="The path of the input video")
    parser.add_argument("--output_video", default=None, type=str,
                        help="The path of the output video")
    parser.add_argument("--frame_width", type=int, default=720,
                        help="The width of frames")
    parser.add_argument("--frame_height", type=int, default=576,
                        help="The height of frames")
    parser.add_argument("--v2i", default=False, type=bool, 
                        help="video_to_images:True")
    parser.add_argument("--i2v", default=False, type=bool, 
                        help="images_to_video:True")

    args = parser.parse_args()
    if args.v2i:
        video_to_images(input_video=args.input_video, save_dir=args.frame_dir)
    if args.i2v:
        images_to_video(args, frame_dir=args.frame_dir, output_video=args.output_video)
 
 
if __name__ == "__main__":
    main()
    # python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/frames --input_video F:/SJTU_VideoAnalysis/Input/demo.avi --v2i True
    # python SplitCombine.py --frame_dir F:/SJTU_VideoAnalysis/frames --output_video F:/SJTU_VideoAnalysis/demo_output.avi --i2v True
