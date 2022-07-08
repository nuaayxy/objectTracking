#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob
import random
import sys


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# Note: Pattern generated using the following link
# https://calib.io/pages/camera-calibration-pattern-generator
board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.012, aruco_dict)


def read_chessboards(frames):
    """
    Charuco base pose estimation.
    """
    all_corners = []
    all_ids = []
    cv2.imshow("board", board)

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        cv2.imshow("board", board)
        if len(corners) > 0:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            # ret is the number of detected corners
            if ret > 0:
                all_corners.append(c_corners)
                all_ids.append(c_ids)
            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
            cv2.imshow('frame',gray)
        else:
            print('Failed!')

    imsize = gray.shape
    return all_corners, all_ids, imsize


def capture_camera(dev_num=0, num=1, mirror=False, size=None):
    frames = []

    cap = cv2.VideoCapture(dev_num)

    while True:
        ret, frame = cap.read()

        if mirror is True:
            frame = cv2.flip(frame, 1)

        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # My config applies floating layout for windows named 'Java'
        cv2.imshow('Java', frame)

        k = cv2.waitKey(1)
        if k == 27:  # Esc
            break
        elif k == 10 or k == 32:  # Enter or Space
            frames.append(frame)
            print('Frame captured!')
            if len(frames) == num:
                break

    return frames


def draw_axis(frame, camera_matrix, dist_coeff, board, verbose, aruco_setup):
    corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, aruco_setup)
    # print(corners, ids)
    if corners is None or ids is None:
        return None
    if len(corners) != len(ids) or len(corners) == 0:
        print("error number")
        return None

    # try:
    ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
                                                                ids,
                                                                frame,
                                                                board)
    if c_corners is not None and c_ids is not None:
        # cv2.aruco.drawDetectedMarkers(frame,c_corners,c_ids)
        ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
                                                                c_ids,
                                                                board,
                                                                camera_matrix,
                                                                dist_coeff)
        if p_rvec is None or p_tvec is None:
            print("p t is none")
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            print(" is nan ")
            
        cv2.aruco.drawAxis(frame,
                        camera_matrix,
                        dist_coeff,
                        p_rvec,
                        p_tvec,
                        10.1)
        # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
        # except cv2.error:
        #     return None

        if verbose:
            print('Translation : {0}'.format(p_tvec))
            print('Rotation    : {0}'.format(p_rvec))
            print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec)))

        return frame


def main():
    video_dev = int(1)
    # frames = capture_camera(video_dev, 50)
    # if len(frames) == 0:
    #     print('No frame captured')
    #     sys.exit(1)
    # all_corners, all_ids, imsize = read_chessboards(frames)
    # all_corners = [x for x in all_corners if len(x) >= 4]
    # all_ids = [x for x in all_ids if len(x) >= 4]
    # ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(
    #     all_corners, all_ids, board, imsize, None, None
    # )

    # print('> Camera matrix')
    # print(camera_matrix)
    # print('> Distortion coefficients')
    # print(dist_coeff)
    allCorners = []
    allIds = []
    decimator = 0
    # Real-time axis drawing
    cap = cv2.VideoCapture(video_dev)

    for i in range(200):
        imboard = board.draw((600, 600))
        cv2.imshow("test", imboard)
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.aruco.detectMarkers(gray,aruco_dict)
        print(res)
        if len(res[0])>0:
            res2 = cv2.aruco.interpolateCornersCharuco(res[0],res[1],gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%3==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])
            cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
        cv2.imshow('frame', gray)
        k = cv2.waitKey(1)
        decimator+=1
        if k == 27:  # Esc
            break

    imsize = gray.shape

    #Calibration fails for lots of reasons. Release the video if we do
    # try:
    ret, camera_matrix, dist_coeff, rvec, tvec = cv2.aruco.calibrateCameraCharuco(allCorners,allIds,board,imsize,None,None)
    print(camera_matrix)
    print(dist_coeff)
    # except:
    #     cap.release()

    while True:
        ret, frame = cap.read()
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = draw_axis(gray, camera_matrix, dist_coeff, board, True, aruco_dict)
            # if axis_frame is not None:
        cv2.imshow('Java', gray)
        k = cv2.waitKey(10)
        decimator+=1
        if k == 27:  # Esc
            break

        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()