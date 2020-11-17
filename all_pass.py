#  ___                    _   _   
# |_ _|__ _  __ _  ___   | \ | |      Iago Nicolas (Necronzero)
#  | |/ _` |/ _` |/ _ \  |  \| |      https://github.com/IagoNicolas
#  | | (_| | (_| | (_) | | |\  |_     
# |___\__,_|\__, |\___/  |_| \_(_)    Works with python 3.8.6 64-bit
#           |___/                     @ Thinkpad T480 on Manjaro 20.1 Micah.

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
    # by: Iago N.
    # img_color_PIL = Image.fromarray(img_color, 'RGB')
    # img_color_PIL.show()
    img_gs = cv2.imread(image_file, 0)
    # by: Iago N.
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

img_gs, img_color = image_load("Lenna.tif")

b = [-1/2, 1]
a = [1, 1/2]
# by: Iago N.
# Using this N° of instances while no artifacts happen,
# higher tanks performance, lower tanks quality.
# Is this even working?
#
# YES, but worN has to be 2*len(image).
#
# DON'T TRY ON 4K IMAGES AGAIN. Since this program is 
# single threaded, no ones pc will run this fast.
w, h = freqz(b, a, worN = 1024)

k = lfilter(x = img_color[:,:,:], b = w, a = h)
# by: Iago N.
# k = lfilter(x = img_gs, b = w, a = h)

out2 = np.fft.fft2(k)

img = Image.fromarray(np.uint8(out2), "RGB")

img.show()
