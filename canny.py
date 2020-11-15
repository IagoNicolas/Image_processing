#  ___                    _   _   
# |_ _|__ _  __ _  ___   | \ | |      Iago Nicolas (Necronzero)
#  | |/ _` |/ _` |/ _ \  |  \| |      https://github.com/IagoNicolas
#  | | (_| | (_| | (_) | | |\  |_     
# |___\__,_|\__, |\___/  |_| \_(_)    Ran with python 3.8.6 64-bit
#           |___/                     @ Thinkpad T480 on Manjaro 20.1 Micah.

from scipy.signal import bilinear
from scipy.signal import lfilter
from scipy.signal import freqz
from scipy.signal import medfilt2d
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import feature
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import time
import cv2
import sys
import pdb

def image_load(image_file):
    img_color = cv2.imread(image_file, 1)
    # by: Iago N.
    # img_color_PIL = Image.fromarray(img_color, 'RGB')
    # img_color_PIL.show()
    img_gs = cv2.imread(image_file, 0)
    # by: Iago N.
    # img_gs_PIL = Image.fromarray(img_gs , 'L')
    # img_gs_PIL.show()
    image_check(img_color, img_gs)
    # by: Iago N.
    # Use once, cv2 format is bs to work with
    img_color = cv2.cvtColor(cv2.imread(image_file, 1), cv2.COLOR_BGR2RGB)
    return (img_gs, img_color)

def image_check(img_color, img_gs):
    if img_color is None:
        print("\nError opening color image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        # by: Iago N.
        # print("\nColor image loaded!")
        time.sleep(0)

    if img_gs is None:
        print("Error opening grayscale image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        # by: Iago N.
        # print("Grayscale image loaded!")
        time.sleep(0)
    return None

def edge_gs_detect(image, std_dev, k_size):
    k_size = int(k_size) // 2
    x, y = np.mgrid[-k_size : k_size + 1, -k_size : k_size + 1]
    normal = 1 / (2.0 * np.pi * pow(std_dev, 2))
    gauss = np.exp(-((pow(x, 2) + pow(y, 2)) / (2.0 * pow(std_dev, 2)))) * normal
    img_conv = convolve(img_gs, gauss)
    # by: Iago N.
    # Wtf is this convolution missing...
    # pdb.set_trace()
    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img_conv, dx)
    Iy = ndimage.filters.convolve(img_conv, dy)
    edge = np.hypot(Ix, Iy)
    edge = edge / edge.max() * 255
    arctg = np.arctan2(Iy, Ix)
    
    edge = no_blur(edge, arctg)
    edge = finding_white(edge, high = .15, low = .05)
    # by: Iago N.
    # Set params
    # pdb.set_trace()
    edge = finding_black(edge, white = 100, gray = 75)
    # by: Iago N.
    # Set params, the return
    # pdb.set_trace()
    return edge

def no_blur(image, arctg):
    val = np.zeros((len(image), len(image)), dtype=np.int32)
    side = arctg * 180. / np.pi
    side[side < 0] += 180

    for i in range(1,len(image)-1):
        for j in range(1,len(image)-1):
            q = 255
            r = 255
            # by: Iago N.
            # 0°
            if (0 <= side[i,j] < 22.5) or (157.5 <= side[i,j] <= 180):
                q = image[i, j+1]
                r = image[i, j-1]
            # by: Iago N.
            # 45°
            elif (22.5 <= side[i,j] < 67.5):
                q = image[i+1, j-1]
                r = image[i-1, j+1]
            # by: Iago N.
            # 90°
            elif (67.5 <= side[i,j] < 112.5):
                q = image[i+1, j]
                r = image[i-1, j]
            # by: Iago N.
            # 135°
            elif (112.5 <= side[i,j] < 157.5):
                q = image[i-1, j-1]
                r = image[i+1, j+1]
            if (image[i,j] >= q) and (image[i,j] >= r):
                val[i,j] = image[i,j]
            else:
                val[i,j] = 0
    # by: Iago N.
    # Trace? Not even sure what is wrong...
    # pdb.set_trace()
    return val

def finding_white(image, high, low):
    high = image.max() * high
    low = high * low
    bright = np.zeros((len(image), len(image)), dtype=np.int32)
    # by: Iago N.
    # int32, why not.
    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= high)
    z_i, z_j = np.where(image < low)
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    # by: Iago N.
    # Garbage variables, have to fix
    # pdb.set_trace()
    bright[strong_i, strong_j] = strong
    bright[weak_i, weak_j] = weak

    return (bright)
# by: Iago N.
# Recheck this, strange behaviour.
def finding_black(image, white, gray):
    for i in range(1, len(image)-1):
        for j in range(1, len(image)-1):
            if (image[i,j] == gray):
                if ((image[i+1, j-1] == white) or (image[i+1, j] == white) or (image[i+1, j+1] == white)
                    or (image[i, j-1] == white) or (image[i, j+1] == white)
                    or (image[i-1, j-1] == white) or (image[i-1, j] == white) or (image[i-1, j+1] == white)):
                    image[i, j] = white
                    # pdb.set_trace()
                else:
                    image[i, j] = 0
    return image

img_gs, img_color = image_load("Lenna.tif")

edge_gs = edge_gs_detect(img_gs, 1, 3)

# by: Iago N.
# Using uint8, else it goes analog ¯\_('.')_/¯
edge = Image.fromarray(np.uint8(edge_gs), "L")
# by: Iago N.
# pdb.set_trace()
orig = Image.fromarray(np.uint8(img_gs), "L")
# by: Iago N.
# pdb.set_trace()

orig.show()
edge.show()