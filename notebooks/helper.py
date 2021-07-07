import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import imageio
import os
import logging
import boto3
from botocore.exceptions import ClientError
from urllib.request import urlretrieve
from multiprocessing.dummy import Pool
from itertools import repeat
import pandas as pd
from subprocess import run
import subprocess

def displayImage(img, title=""):
    """
    Takes an image and displays it inline.
    Automatically detects color vs. grayscale.
    img: image to display
    title: optional title for image
    """
    if len(np.shape(img)) > 2: # see if this is a color image
        img = img[:,:,::-1] # shuffle color from BGR to RGB for matplotlib display
    else:
        plt.gray()
    plt.imshow(img)
    plt.axis('off')
    if title != "":
        plt.title(title)
    plt.show()
    
def histo(img, channel=[0]):
    """
    Plots a histogram of channels in the given image.
    img: image to analyze.
    channel: list containing relevant channels in img.
    """
    histo = cv2.calcHist(img, 
        channels=channel, 
        mask=None, 
        histSize=[256], 
        ranges=[0,256],
    )
    plt.plot(histo)
    plt.show()
    
def subsectionAroundCircle(img, circle, sigma=1.5): # assumes grayscale
    """
    Pulls out a subsection of an image around the selected centerpoint.
    img: grayscale image to subsection
    circle: a tuple of form x,y,r
    x: x location of center of subsection
    y: y location of center of subsection
    r: radius of image to look at.
    sigma: scaling factor of how much "extra" to capture. sigma=1 means subsection stops at radius
    returns: subsection of image.
    """
    x,y,r = circle
    h,w = img.shape
    lowx = np.clip(int(x-r*sigma), 0, w)
    hix = np.clip(int(x+r*sigma), 0, w)
    lowy = np.clip(int(y-r*sigma), 0, h)
    hiy = np.clip(int(y+r*sigma), 0, h)
    return img[lowy:hiy,lowx:hix]

def inRangeOfCenter(img, blur=7, sigma=12): # assumes grayscale
    """
    Makes binary image of pixels within specified intensity range of
    pixels at center of image.
    img: grayscale image to threshold
    blur: strength of gaussian blur to apply. 0 means no blurring.
    sigma: range of values to accept as in range.
    returns: thresholded image.
    """
    y = int(np.shape(img)[0]/2)
    x = int(np.shape(img)[1]/2)
    color = img[y][x]
    lower = np.array(color-sigma,dtype="uint8")
    upper = np.array(color+sigma,dtype="uint8")
    blurred = img
    if blur != 0:
        blurred = cv2.GaussianBlur(img,(blur,blur),0)
    mask = cv2.inRange(blurred, lower, upper)
    masked = cv2.bitwise_and(img,img,mask=mask)
    return cv2.threshold(masked,60,255,cv2.THRESH_BINARY)[1]