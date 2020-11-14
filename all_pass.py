from scipy.signal import bilinear
from scipy.signal import lfilter
from scipy.signal import freqz
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import feature
from tqdm import tqdm
import scipy.ndimage
import pandas as pd
import numpy as np
import time
import cv2
import sys


def image_load(image_file):
    img_color = cv2.imread(image_file, 1)
    # img_color_PIL = Image.fromarray(img_color, 'RGB')
    # img_color_PIL.show()
    img_gs = cv2.imread(image_file, 0)
    # img_gs_PIL = Image.fromarray(img_gs , 'L')
    # img_gs_PIL.show()
    image_check(img_color, img_gs)
    img_color = cv2.cvtColor(cv2.imread(image_file, 1), cv2.COLOR_BGR2RGB)
    return (img_gs, img_color)


def image_check(img_color, img_gs):
    if img_color is None:
        print("\nError opening color image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        # print("\nColor image loaded!")
        time.sleep(0)

    if img_gs is None:
        print("Error opening grayscale image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        # print("Grayscale image loaded!")
        time.sleep(0)
    return None

img_gs, img_color = image_load("Lenna.tif")

b = [-1/2, 1]
a = [1, 1/2]

w, h = freqz(b, a, worN = 3584)

#y = lfilter(x = img_color[:,:,0], b = w, a = h)
y = lfilter(x = img_gs, b = w, a = h)

img = Image.fromarray(np.uint8(y), "L")

img.show()
