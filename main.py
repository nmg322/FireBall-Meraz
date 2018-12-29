import sys
import os
import signal
import copy

import numpy as np
import cv2
import pygame

import time
import argparse

import utility
import pong


class SignalHandler:
    """
    The object that will handle signals and stop the worker threads.
    """

    def __init__(self, grabber):
        self.grabber = grabber

    def __call__(self, signum, frame):
        """
        This will be called by the python signal module
        https://docs.python.org/3/library/signal.html#signal.signal
        """

        cv2.destroyAllWindows()
        self.grabber.stop()
        self.grabber.join()
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        type=int,
        default=0,
        help="Display the detected images using OpenCV. This reduces FPS")
    parser.add_argument(
        "-ip",
        "--server_ip",
        dest="host",
        type=str,
        default="127.0.0.1",
        help="IP of host server for camera")
    parser.add_argument(
        "-port",
        "--server_port",
        dest="port",
        type=int,
        default=1080,
        help="Port on which host server for camera is running")
    args = parser.parse_args()

    grabber = utility.WebcamVideoStream((args.host, args.port))
    if args.debug: print("Camera Resolution:", grabber.cam_size)
    grabber.start()

    # Create our signal handler and connect it
    handler = SignalHandler(grabber)
    signal.signal(signal.SIGINT, handler)

    game = pong.Pong()

    # Kernal's for image processing
    kernel_morp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_erod = np.ones((3, 3), np.uint8)

    # Parameters for image processing
    threshold = 60  #  BINARY threshold
    blurValue = 51  # GaussianBlur parameter
    bgSubThreshold = 100
    historyL, historyR = 0, 0
    learningRate = 0

    # Camera crop parameters
    cropLY, cropLX = (0, grabber.cam_size[0]), (0, 100)
    cropRY, cropRX = (0, grabber.cam_size[0]), (540, grabber.cam_size[1])

    # Variables
    isBgCaptured = False   # Bool, whether the background captured
    fgmaskL = np.zeros((cropLY[1]-cropLY[0], cropLX[1]-cropLX[0], 3))
    fgmaskR = np.zeros((cropRY[1]-cropRY[0], cropRX[1]-cropRX[0], 3))
    yL, hL, yR, hR = 0, 0, 0, 0

    if args.debug:
        cv2.namedWindow("trackbar")
        cv2.createTrackbar("trh1", "trackbar", threshold, 200, lambda x: print("!Threshold is changed to", x))

    while True:
        # Keyboard input
        k = cv2.waitKey(1) & 0xFF
        keys = pygame.key.get_pressed()

        if keys[pygame.K_c]:  # press 'c' to capture the background
            bgModelL = cv2.createBackgroundSubtractorMOG2(historyL, bgSubThreshold, detectShadows=False)
            bgModelR = cv2.createBackgroundSubtractorMOG2(historyR, bgSubThreshold, detectShadows=False)
            isBgCaptured = True
            print("!Background Captured")
        elif keys[pygame.K_b]:  # press 'b' to reset the background
            bgModel = None
            isBgCaptured = False
            print ("!BackGround Reset")
        elif keys[pygame.K_r] and (game.state==pong.PLAYER1_WINS or game.state==pong.PLAYER2_WINS):
            game.game_init()
        elif keys[pygame.K_q]:
            break

        if args.debug:
            threshold = cv2.getTrackbarPos("trh1", "trackbar")

        frame = grabber.read()
        if frame is not None :
            # Croping players out of the frame of camera
            #frameL, frameR = frame[cropYL:cropYH, :320, :], frame[cropYL:cropYH, 320:, :]
            #playerLL, playerLR = frame[:, :100, :], frame[:, 540:, :]
            #playerRL, playerRR = frameR[:, :93, :], frameR[:, 222:, :]
            playerL = frame[cropLY[0]:cropLY[1], cropLX[0]:cropLX[1], :]
            playerR = frame[cropRY[0]:cropRY[1], cropRX[0]:cropRX[1], :]
            cv2.rectangle(frame, (cropLX[0], cropLY[0]), (cropLX[1], cropLY[1]), (0, 255, 0), 1)
            cv2.rectangle(frame, (cropRX[0], cropRY[0]), (cropRX[1], cropRY[1]), (0, 0, 255), 1)

            if isBgCaptured:
                # For playerL
                fgmaskL = bgModelL.apply(playerL, fgmaskL, learningRate=learningRate)
                imgL = cv2.morphologyEx(fgmaskL, cv2.MORPH_OPEN, kernel_morp)
                imgL = cv2.erode(imgL, kernel_erod, iterations=1)
                imgL = cv2.bitwise_and(playerL, playerL, mask=imgL)
                if args.debug: cv2.imshow("ForegroundL", imgL)

                # Convert the image into binary image
                imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

                blurL = cv2.GaussianBlur(imgL, (blurValue, blurValue), 0)
                #blurL = cv2.bilateralFilter(blurL, 5, 75, 75)
                if args.debug: cv2.imshow("BlurL", blurL)

                ret, threshL = cv2.threshold(blurL, threshold, 255, cv2.THRESH_BINARY)
                if args.debug: cv2.imshow("ThresholdL", threshL)

                # Get the coutours
                contours, hierarchy = cv2.findContours(threshL.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>0:
                    resL = max(contours, key=lambda x: cv2.contourArea(x))
                    xL, yL, wL, hL = cv2.boundingRect(resL)
                    cv2.rectangle(frame, (cropLX[0]+xL, cropLY[0]+yL),
                     (cropLX[0]+xL+wL, cropLY[0]+yL+hL), (255, 0, 0), 2)

                # For playerR
                fgmaskR = bgModelR.apply(playerR, fgmaskR, learningRate=learningRate)
                imgR = cv2.morphologyEx(fgmaskR, cv2.MORPH_OPEN, kernel_morp)
                imgR = cv2.erode(imgR, kernel_erod, iterations=1)
                imgR = cv2.bitwise_and(playerR, playerR, mask=imgR)
                if args.debug: cv2.imshow("ForegroundR", imgR)

                # Convert the image into binary image
                imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

                blurR = cv2.GaussianBlur(imgR, (blurValue, blurValue), 0)
                #blurR = cv2.bilateralFilter(blurR, 5, 75, 75)
                if args.debug: cv2.imshow("BlurR", blurR)

                ret, threshR = cv2.threshold(blurR, threshold, 255, cv2.THRESH_BINARY)
                if args.debug: cv2.imshow("ThresholdR", threshR)

                # Get the coutours
                contours, hierarchy = cv2.findContours(threshR.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>0:
                    resR = max(contours, key=lambda x: cv2.contourArea(x))
                    xR, yR, wR, hR = cv2.boundingRect(resR)
                    cv2.rectangle(frame, (cropRX[0]+xR, cropRY[0]+yR),
                     (cropRX[0]+xR+wR, cropRY[0]+yR+hR), (255, 0, 0), 2)

            if args.debug: cv2.imshow("frame", frame)

            # Interpolating the position of user from range of camera's height to game screen height
            ymL, ymR = yL+hL//2, yR+hR//2
            yoL = np.interp(ymL, [cropLY[0], cropLY[1]], [-pong.PADDLE1_HEIGHT/2, pong.SCREEN_SIZE[1]+pong.PADDLE1_HEIGHT/2])
            yoR = np.interp(ymR, [cropRY[0], cropRY[1]], [-pong.PADDLE2_HEIGHT/2, pong.SCREEN_SIZE[1]+pong.PADDLE2_HEIGHT/2])
            if args.debug: print("ymL:", ymL, "ymR:", ymR, "yoL:", yoL, "yoR:", yoR)

            # Updating paddle's velocity according to player's change in position
            game.paddle1_vel = int(yoL - game.paddle1.top)
            game.paddle1.top += game.paddle1_vel
            if game.paddle1.top < 0:
                game.paddle1.top = 0
                game.paddle1_vel = 0
            if game.paddle1.top > pong.MAX_PADDLE1_Y:
                    game.paddle1.top = pong.MAX_PADDLE1_Y
                    game.paddle1_vel = 0

            game.paddle2_vel = int(yoR - game.paddle2.top)
            game.paddle2.top += game.paddle2_vel
            if game.paddle2.top < 0:
                game.paddle2.top = 0
                game.paddle2_vel = 0
            if game.paddle2.top > pong.MAX_PADDLE2_Y:
                    game.paddle2.top = pong.MAX_PADDLE2_Y
                    game.paddle2_vel = 0
        game.run()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break

    cv2.destroyAllWindows()
    grabber.stop()
    grabber.join()
    pygame.quit()
