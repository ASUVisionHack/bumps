import numpy as np
import cv2
import os
import sys
import time
import imutils
from matplotlib import pyplot as plt



class Annotate:
    def __init__(self, filename_ext):
        self.filename_ext = filename_ext
        self.filename = filename_ext[:len(self.filename_ext) - 4]  # remove .avi
        self.frame = None
        self.count = 1

    def process_video(self):
        video = cv2.VideoCapture(self.filename_ext)
        print('loaded video {}'.format(self.filename))

        cv2.namedWindow('video {}'.format(self.filename))

        found = False
        #SIFT stuff
        bumpSignPic = cv2.imread('/home/chowder/Desktop/bumpSign.png',0)
        #cv2.imshow('bumpSignPic', bumpSignPic)
        #key = cv2.waitKey(0) & 0xFF
        #bumpSignPic = cv2.cvtColor(bumpSignPic ,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        Samplekp, Sampledes = sift.detectAndCompute(bumpSignPic,None)

        frame_number =   0
        while video.isOpened():
            ret, self.frame = video.read()

            if not ret:
                break



            # how to Test
                    #show the blurred image
                    #show the Binary image
                    #show the image with all the contours
                    #get permiter of sign to filter
                    #show image with all 4 sided contours
                    #show the four sided contours
                    #show the ORB comparison

            #hella reduce the resolution
            lowRez = cv2.resize(self.frame, (0,0), fx=0.3, fy=0.3)

            cv2.imshow('lowRez', lowRez)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            #convert to grayscale
            gray = cv2.cvtColor(lowRez, cv2.COLOR_BGR2GRAY)

            #apply gaussian blur
            #blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            #cv2.imshow('blurred', blurred)
            #key = cv2.waitKey(0) & 0xFF
            #cv2.destroyAllWindows()

            #automatically convert to binary
            (thresh, bandw) = cv2.threshold(gray.copy(), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            cv2.imshow('bandw', bandw)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            #find contours
            cnts = cv2.findContours(bandw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            possSigns = []
            print("number of Contours", len(cnts))

            #print("Average Length of Contours", np.average(cnts))
            cv2.imshow('rawCont', cv2.drawContours(gray.copy(),cnts, -1, (200,255,0), 1))
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            #iterate through approvimations with 4 sides
            forSides = 0
            gr8rThan10 = 0
            for c in cnts:
                #apply approximate
                peri = cv2.arcLength(c, True)
                if (peri > 50):#if it's big enough then check the sides
                    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                    gr8rThan10 += 1
                    if len(approx) == 4:
                        possSigns.append(c)
                        forSides += 1
            print("number of Contours with primiter greater than 10", gr8rThan10)
            print("number of Contours with 4 sides", forSides)
            cv2.imshow('rawCont', cv2.drawContours(gray.copy(),possSigns, -1, (200,255,255), 1))
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()


            #iterate through approvimations with 4 sides
            for psc in possSigns:
                cv2.imshow('rawCont', cv2.drawContours(gray.copy(),psc, -1, (200,255,255), 1))
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                #get the approximating rectangle
                x,y,w,h = cv2.boundingRect(psc)
                print(x,y,w,h)
                #crop
                crop = gray.copy()[y:y+h, x:x+w]
                cv2.imshow('rawCont',crop)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
"""                #use ORB to compare
                kp2, des2 = sift.detectAndCompute(crop,None)
                # Using FLANN based Matcher for matching the descriptors
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks = 500)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1,des2,k=2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                # Draw first 10 matches.
                img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

                plt.imshow(img3),plt.show()

                if len(matches) > 5:
                    print ("found!")
                    found = True
            #iterate through polygons using orb to tell if they're the shit



            lastFrame = self.frame
            subframe = diff[200:201, 0:1920]

            thresh = 127
            v_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

            print(np.average(subframe))
            cv2.imshow('window', diff)
            cv2.imshow('video {}'.format(self.filename), subframe)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('h'):
                 plt.hist(subframe.ravel(), 256, [0, 256])
                 plt.show()

            elif key == ord('q'):
                 exit()

            frame_number += 1

        cv2.destroyAllWindows()
"""

if __name__ == '__main__':
    a = Annotate(sys.argv[1])
    a.process_video()
