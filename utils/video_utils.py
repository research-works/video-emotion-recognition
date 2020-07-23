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
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import utils.local_config as local_config

def extract_frames(cam, output_dir, frame_rate):
    currentframe = 0
    count = 0

    while(True): 
        # reading from frame 
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            if (currentframe % frame_rate) == 0:
                name = output_dir+ '/frame' + str(currentframe) + '.jpg'
                #print ('Creating...' + name) 
                
                # writing the extracted images 
                print(frame.shape)
                cv2.imwrite(name, frame)
                count = count +1 

                # increasing counter so that it will 
                # show how many frames are created 
            currentframe += 1
        else:
            print("oof") 
            break
    return frame_rate, count


class FaceAligner:
	def __init__(self, predictor, desiredLeftEye=(0.20, 0.30),
		desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth

	def align(self, image, gray, rect):
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = self.predictor(gray, rect)
		shape = shape_to_np(shape)
		
		#simple hack ;)
		if (len(shape)==68):
			# extract the left and right eye (x, y)-coordinates
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		else:
			(lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
			
		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		# return the aligned face
		return output

def rotate_image(frames_dir_path, output_dir_path):
 
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(local_config.BASE_DIR, 'resources/shape_predictor_5_face_landmarks.dat'))
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    print(frames_dir_path)
    for filename in os.listdir(frames_dir_path):
        
        # load the input image, resize it, and convert it to grayscale
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = cv2.imread(frames_dir_path + '/' + filename)
            #test_image = image.load_img("./data/"+filename)
            image = imutils.resize(image, width=800)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # show the original input image and detect faces in the grayscale
            # image
            #cv2_imshow(image)
            # print("Input Image")
            # #plt.gray()
            # plt.imshow(image)
            # plt.show()

            rects = detector(gray, 2)
            # loop over the face detections
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
                faceAligned = fa.align(image, gray, rect)

                # print("Original Image: " + filename)
                # cv2_imshow(faceOrig)
                    
                # print("Aligned Image")
                # cv2_imshow(faceAligned)

                name = output_dir_path + '/' + filename
                print(name)
                print("size: "+str(faceAligned.size))
                cv2.imwrite(name, faceAligned)

                
                #break



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










