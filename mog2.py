from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from skimage import data
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import  Image
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(1000,12, True)
else:
    backSub = cv.createBackgroundSubtractorKNN(800,150, True)
    
capture = cv.VideoCapture(2)
capture.set(cv.CAP_PROP_FPS, 60)
capture.set(cv.CAP_PROP_EXPOSURE,-7)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
    
pImage = np.zeros((480, 640, 3), dtype = "uint8")
while True:
    ret, frame = capture.read()

    
    if frame is None:
        break
    
    lab = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    # b = lab - pImage 
    # pImage = lab
    cv.imshow('lab', lab)
    
    fgMask = backSub.apply(lab)
    
    shadows = np.zeros(fgMask.shape, dtype=np.uint8) 
    not_shadows = (fgMask == 0) | (fgMask == 255) 
    shadows[~not_shadows] = 255
    # shadow = backSub.getDetectShadows()
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (1 5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('FG Mask 1', shadows)
    
    
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LUV)
    cv.imshow('lab', lab[:,:,2])
        
    # rgb = np.asarray(frame)

    # image_B = np.copy(rgb[:, :, 0])
    # image_G = np.copy(rgb[:, :, 1])
    # image_R = np.copy(rgb[:, :, 2]) 
    # s=np.shape(rgb)

    # #Converting RGB to LAB color space
    # lab = color.rgb2lab(rgb)
    # cv.imshow(lab)
    # image_b = np.copy(lab[:, :, 0])
    # image_a = np.copy(lab[:, :, 1])
    # image_l = np.copy(lab[:, :, 2])

    # lm=np.mean(lab[:,:,0], axis=(0, 1))
    # am=np.mean(lab[:,:,1], axis=(0, 1))
    # bm=np.mean(lab[:,:,2], axis=(0, 1))

    # #Creating empty mask for masking shadow
    # mas = np.empty([rgb.shape[0], rgb.shape[1]], dtype = bool)
    # lb=lab[:,:,0]+lab[:,:,2]

    # #Hand crafted thresholds: Dataset specific
    # if (am+bm)<=15:
    #     mas[(image_l <=(lm-(np.std(image_l))/15))] = False

    # else:
    #     mas[(image_l+image_b)<=50] = False
    #     B_masked = np.ma.masked_array(image_b, mask = mas)
    #     G_masked = np.ma.masked_array(image_G, mask = mas)
    #     R_masked = np.ma.masked_array(image_R, mask = mas) 
    #     mam = np.dstack([rgb, (~mas).astype(np.uint8)*255])

        # plt.subplot(1,2,2)
        # plt.imshow(mam)
        # plt.title('Shadow detected Image')
        # plt.show()
    
    keyboard = cv.waitKey(10)
    if keyboard == 'q' or keyboard == 27:
        break