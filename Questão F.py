from scipy.ndimage import gaussian_filter
from warnings import filterwarnings
from scipy.signal import bilinear
from scipy.signal import convolve
from scipy.signal import lfilter
from PIL import Image, ImageOps
from scipy.ndimage import sobel
from scipy.signal import freqz
from cv2 import COLOR_BGR2RGB
from skimage import feature
from scipy import ndimage
from cv2 import cvtColor
from cv2 import imread
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import sys

filterwarnings("ignore")


def image_load(image_file):
    img_color = imread(image_file, 1)
    img_gs = imread(image_file, 0)
    image_check(img_color, img_gs)
    img_color = cvtColor(imread(image_file, 1), COLOR_BGR2RGB)
    return (img_gs, img_color)


def image_check(img_color, img_gs):
    if img_color is None:
        print("\nError opening color image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        time.sleep(0)

    if img_gs is None:
        print("Error opening grayscale image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        time.sleep(0)
    return None


def grayscale_filter(img_gs, bil_1, bil_2):
    img_gs_df = pd.DataFrame(img_gs)
    img_gs_sv_row = np.zeros(shape=[len(img_gs_df), len(img_gs_df)])
    img_gs_sv_col = np.zeros(shape=[len(img_gs_df), len(img_gs_df)])
    for i in range(0, len(img_gs_df)):
        img_gs_sv_row[i] = lfilter(x=img_gs_df.loc[i, :], b=bil_1, a=bil_2)
    for j in range(0, len(img_gs_df)):
        img_gs_sv_col[j] = lfilter(
            x=pd.DataFrame(img_gs_sv_row).loc[:, j], b=bil_1, a=bil_2
        )
    return (img_gs_sv_row, img_gs_sv_col)


def rgb_filter(img_color, bil_1, bil_2):
    img_color_r_df = pd.DataFrame(img_color[:, :, 0])
    img_color_g_df = pd.DataFrame(img_color[:, :, 1])
    img_color_b_df = pd.DataFrame(img_color[:, :, 2])
    img_r_sv_row = np.zeros(shape=[len(img_color_r_df), len(img_color_r_df)])
    img_r_sv_col = np.zeros(shape=[len(img_color_r_df), len(img_color_r_df)])
    img_g_sv_row = np.zeros(shape=[len(img_color_g_df), len(img_color_g_df)])
    img_g_sv_col = np.zeros(shape=[len(img_color_g_df), len(img_color_g_df)])
    img_b_sv_row = np.zeros(shape=[len(img_color_b_df), len(img_color_b_df)])
    img_b_sv_col = np.zeros(shape=[len(img_color_b_df), len(img_color_b_df)])
    for i in range(0, len(img_color_r_df)):
        img_r_sv_row[i] = lfilter(x=img_color_r_df.loc[i, :], b=bil_1, a=bil_2)
    for j in range(0, len(img_color_r_df)):
        img_r_sv_col[j] = lfilter(
            x=pd.DataFrame(img_r_sv_row).loc[:, j], b=bil_1, a=bil_2
        )

    for i in range(0, len(img_color_g_df)):
        img_g_sv_row[i] = lfilter(x=img_color_g_df.loc[i, :], b=bil_1, a=bil_2)
    for j in range(0, len(img_color_g_df)):
        img_g_sv_col[j] = lfilter(
            x=pd.DataFrame(img_g_sv_row).loc[:, j], b=bil_1, a=bil_2
        )
    for i in range(0, len(img_color_b_df)):
        img_b_sv_row[i] = lfilter(x=img_color_b_df.loc[i, :], b=bil_1, a=bil_2)
    for j in range(0, len(img_color_b_df)):
        img_b_sv_col[j] = lfilter(
            x=pd.DataFrame(img_b_sv_row).loc[:, j], b=bil_1, a=bil_2
        )

    img_rgb_sv_row = np.zeros([512, 512, 3])
    img_rgb_sv_col = np.zeros([512, 512, 3])

    img_rgb_sv_row[:, :, 0] = img_r_sv_row
    img_rgb_sv_row[:, :, 1] = img_g_sv_row
    img_rgb_sv_row[:, :, 2] = img_b_sv_row
    img_rgb_sv_col[:, :, 0] = img_r_sv_col
    img_rgb_sv_col[:, :, 1] = img_g_sv_col
    img_rgb_sv_col[:, :, 2] = img_b_sv_col
    return (img_rgb_sv_row, img_rgb_sv_col)


def image_save(name, image, mode, angle, side):
    img = Image.fromarray(np.uint8(image), mode)
    if side == 0:
        time.sleep(0)
    if side == 1:
        img = ImageOps.flip(img)
    img.rotate(angle).save("output/" + name)
    return None


def all_pass(d1_array, pole, freq_number):
    w, h = freqz([-1 * pole, 1], [1, pole], freq_number)
    y = lfilter(x=d1_array, b=w, a=h)
    return y


def grayscale_allpass(img_gs, pole, freq_number):
    img_gs_df = pd.DataFrame(img_gs)
    img_gs_ap_row = np.zeros(shape=[len(img_gs_df), len(img_gs_df)])
    for i in range(0, len(img_gs_df)):
        img_gs_ap_row[i] = all_pass(img_gs_df.loc[i, :], pole, freq_number)
    return img_gs_ap_row


def rgb_allpass(pole, freq_number):
    img_color_r_df = pd.DataFrame(img_color[:, :, 0])
    img_color_g_df = pd.DataFrame(img_color[:, :, 1])
    img_color_b_df = pd.DataFrame(img_color[:, :, 2])
    img_r_ap_row = np.zeros(shape=[len(img_color_r_df), len(img_color_r_df)])
    img_g_ap_row = np.zeros(shape=[len(img_color_g_df), len(img_color_g_df)])
    img_b_ap_row = np.zeros(shape=[len(img_color_b_df), len(img_color_b_df)])
    for i in range(0, len(img_color_r_df)):
        img_r_ap_row[i] = all_pass(img_color_r_df.loc[i, :], pole, freq_number)

    for i in range(0, len(img_color_g_df)):
        img_g_ap_row[i] = all_pass(img_color_g_df.loc[i, :], pole, freq_number)

    for i in range(0, len(img_color_b_df)):
        img_b_ap_row[i] = all_pass(img_color_b_df.loc[i, :], pole, freq_number)

    img_rgb_ap_row = np.zeros([512, 512, 3])

    img_rgb_ap_row[:, :, 0] = img_r_ap_row
    img_rgb_ap_row[:, :, 1] = img_g_ap_row
    img_rgb_ap_row[:, :, 2] = img_b_ap_row
    return img_rgb_ap_row


def derivative(img):
    img_der_hor = np.diff(img, n=1, axis=0)  # horizontal derivative
    img_der_ver = np.diff(img_der_hor, n=1, axis=1)  # vertical derivative
    return img_der_ver


def no_blur(image, arctg):
    val = np.zeros((len(image), len(image)), dtype=np.int32)
    side = arctg * 180. / np.pi
    side[side < 0] += 180

    for i in range(1,len(image)-1):
        for j in range(1,len(image)-1):
            q = 255
            r = 255
            #angle 0
            if (0 <= side[i,j] < 22.5) or (157.5 <= side[i,j] <= 180):
                q = image[i, j+1]
                r = image[i, j-1]
            #angle 45
            elif (22.5 <= side[i,j] < 67.5):
                q = image[i+1, j-1]
                r = image[i-1, j+1]
            #angle 90
            elif (67.5 <= side[i,j] < 112.5):
                q = image[i+1, j]
                r = image[i-1, j]
            #angle 135
            elif (112.5 <= side[i,j] < 157.5):
                q = image[i-1, j-1]
                r = image[i+1, j+1]
            if (image[i,j] >= q) and (image[i,j] >= r):
                val[i,j] = image[i,j]
            else:
                val[i,j] = 0
    return val


def finding_white(image, high, low):
    high = image.max() * high
    low = high * low
    bright = np.zeros((len(image), len(image)), dtype=np.int32)
    weak = np.int32(75)
    strong = np.int32(255)
    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image <= high) & (image >= low))
    bright[strong_i, strong_j] = strong
    bright[weak_i, weak_j] = weak

    return (bright)


def finding_black(image, white, gray):
    for i in range(1, len(image)-1):
        for j in range(1, len(image)-1):
            if (image[i,j] == gray):
                if ((image[i+1, j-1] == white) or (image[i+1, j] == white) or (image[i+1, j+1] == white)
                    or (image[i, j-1] == white) or (image[i, j+1] == white)
                    or (image[i-1, j-1] == white) or (image[i-1, j] == white) or (image[i-1, j+1] == white)):
                    image[i, j] = white
                else:
                    image[i, j] = 0
    return image


def edge_gs_detect(image, std_dev, k_size):
    k_size = int(k_size) // 2
    x, y = np.mgrid[-k_size : k_size + 1, -k_size : k_size + 1]
    normal = 1 / (2.0 * np.pi * pow(std_dev, 2))
    gauss = np.exp(-((pow(x, 2) + pow(y, 2)) / (2.0 * pow(std_dev, 2)))) * normal
    img_conv = convolve(img_gs, gauss)

    dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = ndimage.filters.convolve(img_conv, dx)
    Iy = ndimage.filters.convolve(img_conv, dy)
    edge = np.hypot(Ix, Iy)
    edge = edge / edge.max() * 255
    arctg = np.arctan2(Iy, Ix)
    
    edge = no_blur(edge, arctg)
    edge = finding_white(edge, high = .15, low = .05)
    edge = finding_black(edge, white = 100, gray = 75)
    return edge


def edge_rgb_detect(image, std_dev, k_size):
    img_color_r_df = pd.DataFrame(image[:, :, 0])
    img_color_g_df = pd.DataFrame(image[:, :, 1])
    img_color_b_df = pd.DataFrame(image[:, :, 2])
    img_color_r_ed = edge_gs_detect(img_color_r_df, std_dev, k_size)
    img_color_g_ed = edge_gs_detect(img_color_g_df, std_dev, k_size)
    img_color_b_ed = edge_gs_detect(img_color_b_df, std_dev, k_size)

    img_rgb_edge = np.zeros([len(img_color_r_ed), len(img_color_r_ed), 3])
    img_rgb_edge[:, :, 0] = img_color_r_ed
    img_rgb_edge[:, :, 1] = img_color_g_ed
    img_rgb_edge[:, :, 2] = img_color_b_ed
    
    return img_rgb_edge


with tqdm(total=5, file=sys.stdout) as pbar:
    # Questão 1
    img_gs, img_color = image_load("Lenna.tif")
    pbar.update(1)
    # Questão 2
    bil_1, bil_2 = bilinear(1, 4)
    img_gs_row, img_gs_col = grayscale_filter(img_gs, bil_1, bil_2)
    image_save("Lenna_gs_1_row.tif", img_gs_row, "L", 0, 0)
    image_save("Lenna_gs_2_col.tif", img_gs_col, "L", 270, 1)
    img_rgb_row, img_rgb_col = rgb_filter(img_color, bil_1, bil_2)
    image_save("Lenna_rgb_1_row.tif", img_rgb_row, "RGB", 0, 0)
    image_save("Lenna_rgb_2_col.tif", img_rgb_col, "RGB", 270, 1)
    pbar.update(1)
    # Questão 3
    img_gs_derivative = derivative(img_gs_col)
    image_save("Lenna_gs_3_der.tif", img_gs_derivative, "L", 270, 1)
    img_rgb_derivative = derivative(img_rgb_col)
    image_save("Lenna_rgb_3_der.tif", img_rgb_derivative, "RGB", 270, 1)
    pbar.update(1)
    # Questão 4
    img_gs_ap = grayscale_allpass(img_gs, 1 / 2, 3584)
    image_save("Lenna_gs_4_ap.tif", img_gs_ap, "L", 0, 0)
    img_rgb_ap = rgb_allpass(1 / 2, 3584)
    image_save("Lenna_rgb_4_ap.tif", img_rgb_ap, "RGB", 0, 0)
    pbar.update(1)
    # Questão 5
    edge_gs = edge_gs_detect(img_gs, 1, 3)
    image_save("Lenna_gs_5_ed.tif", edge_gs, "L", 0, 0)
    edge_rgb = edge_rgb_detect(img_color, 1, 3)
    image_save("Lenna_rgb_5_ed.tif", edge_rgb, "RGB", 0, 0)
    pbar.update(1)
