from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import warnings
import os
import tensorflow as tf
import time
from imutils.face_utils.helpers import FACIAL_LANDMARKS_68_IDXS
from imutils.face_utils.helpers import FACIAL_LANDMARKS_5_IDXS
from imutils.face_utils.helpers import shape_to_np
import numpy as np
import cv2
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def sub(frames_dir_path, intermediate_dir_path, output_dir_path, frame_rate):

    try: 
        # creating a folder named data 
        if not os.path.exists(output_dir_path): 
            os.makedirs(output_dir_path)
        
        if not os.path.exists(intermediate_dir_path): 
            os.makedirs(intermediate_dir_path) 

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of sub_dataset') 

    # jump = int(frame_rate) #the amount with which we have to jump for subtracting next set of images
    # print(jump)
    # n = int(count) #total number of frames from a video 

    # #copying images
    # for i in range(0, n*jump, jump):
    #     cv2.imwrite('./result/img'+str(i)+'.jpg', cv2.imread('./rotated_images/frame'+str(i)+'.jpg',0))

    #to count number of frames
    count=0
    for files in os.listdir(frames_dir_path):
        count = count+1

    jump = int(frame_rate) #the amount with which we have to jump for subtracting next set of images
    n = int(count) #total number of frames from a video
    # print(n)
    # print(jump)
    #print("folder: "+folder)

    #subtracting images in following pattern
    #   * - *   * - *   * - *   * - *....
    #     *   -   *       *   -   * ....
    #         *       -       * ....
    #                 *
        
    while n>1:
        #print("n: ",n)
        j = 0
        for i in range (0, (n*jump)-jump, jump):
            re_img0 = np.int32(cv2.imread(frames_dir_path + '/frame' + str(i) + '.jpg',1))
            print("i: "+str(i)+" aux: "+str(i+jump))
            #print("d: ",i+j)
            aux = int(i + jump)
            #print('aux: ',aux )
            #issue: with adding i & j; code works when either i or j are there instead of aux(sum of i+j)
            re_img1 = np.int32(cv2.imread(frames_dir_path + '/frame' + str(aux) + '.jpg',1))
            sub = re_img0 - re_img1
            sub = np.absolute(sub)

            cv2.imwrite(intermediate_dir_path + '/frame' + str(j)+'.jpg', sub)
            j = j + jump
            print("inner")
        print("outer")
        n=n//2
    
    sub_image_path = output_dir_path + '/' + os.path.basename(frames_dir_path).split('_')[1] + '.jpg'
    cv2.imwrite(sub_image_path, sub)


def preprocess_video(input_video_path, output_dir_path, frame_rate):
    #read video from specified path
    print("path of videos: " + input_video_path)
    cam = cv2.VideoCapture(input_video_path)          
    fname = os.path.basename(input_video_path).split('.')[0]
    frames_dir_path = output_dir_path + '/' + 'frames' + '/' + 'data_' + fname 
    rotated_frames_dir_path = output_dir_path + '/' + 'rotated_frames' + '/' + 'rotated_' + fname
    intermediate_dir_path = output_dir_path + '/' + 'sub_intermediate' + '/' + 'intermediate_' + fname
    sub_dir_path = output_dir_path + '/' + 'subtracted_frames'
    try:
        # creating a folder per video 
        if not os.path.exists(frames_dir_path): 
            os.makedirs(frames_dir_path) 
        
        if not os.path.exists(rotated_frames_dir_path): 
            os.makedirs(rotated_frames_dir_path) 
        
        if not os.path.exists(intermediate_dir_path): 
            os.makedirs(intermediate_dir_path)

        if not os.path.exists(sub_dir_path): 
            os.makedirs(sub_dir_path)
               
            # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data')
                
    frame_rate, count = extract_frames(cam, frames_dir_path, frame_rate)
                
                # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()

    #rotate frames
    rotate_image(frames_dir_path, rotated_frames_dir_path)
    sub(rotated_frames_dir_path, intermediate_dir_path, sub_dir_path, frame_rate)










