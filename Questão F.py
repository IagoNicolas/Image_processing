from scipy.signal import bilinear
from scipy.signal import lfilter
from scipy.signal import freqz
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.signal import convolve
from scipy import ndimage
from PIL import Image, ImageOps
from skimage import feature
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings
import time
import cv2
import sys

warnings.filterwarnings("ignore")


def image_load(image_file):
    img_color = cv2.imread(image_file, 1)
    img_gs = cv2.imread(image_file, 0)
    image_check(img_color, img_gs)
    img_color = cv2.cvtColor(cv2.imread(image_file, 1), cv2.COLOR_BGR2RGB)
    pbar.update(1)
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
    pbar.update(1)
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
    pbar.update(1)
    return (img_rgb_sv_row, img_rgb_sv_col)


def grayscale_save(row, col):
    img = Image.fromarray(np.uint8(row), "L")
    img.save("output/Lenna_gs_1_row.tif")
    img = Image.fromarray(np.uint8(col), "L")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_gs_2_col.tif")
    pbar.update(1)
    return None


def grayscale_der_save(der):
    img = Image.fromarray(np.uint8(der), "L")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_gs_der.tif")
    pbar.update(1)
    return None


def rgb_der_save(der):
    img = Image.fromarray(np.uint8(der), "RGB")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_rgb_der.tif")
    pbar.update(1)
    return None


def rgb_save(row, col):
    row = np.clip(row, 0, 255)
    img = Image.fromarray(np.uint8(row), "RGB")
    img.save("output/Lenna_rgb_1_row.tif")
    col = np.clip(col, 0, 255)
    img = Image.fromarray(np.uint8(col), "RGB")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_rgb_2_col.tif")
    pbar.update(1)
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
    pbar.update(1)
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
    pbar.update(1)
    return img_rgb_ap_row


def grayscale_allpass_save(img_gs_ap):
    img = Image.fromarray(np.uint8(img_gs_ap), "L")
    img.save("output/Lenna_gs_1_ap.tif")
    pbar.update(1)
    return None


def rgb_allpass_save(img_rgb_ap):
    row = np.clip(img_rgb_ap, 0, 255)
    img = Image.fromarray(np.uint8(img_rgb_ap), "RGB")
    img.save("output/Lenna_rgb_1_ap.tif")
    pbar.update(1)
    return None


def derivative(img):
    img_der_hor = np.diff(img, n=1, axis=0)  # horizontal derivative
    img_der_ver = np.diff(img_der_hor, n=1, axis=1)  # vertical derivative
    pbar.update(1)
    return img_der_ver


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
    pbar.update(1)
    return edge


def edge_gs_save(image):
    img = Image.fromarray(np.uint8(image), "L")
    img.save("output/Lenna_gs_edge.tif")
    pbar.update(1)
    return None


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


def edge_rgb_save(image):
    img = Image.fromarray(np.uint8(image), "RGB")
    img.save("output/Lenna_rgb_edge.tif")
    pbar.update(1)
    return None


with tqdm(total=19, file=sys.stdout) as pbar:
    # Questão 1
    img_gs, img_color = image_load("Lenna.tif")
    # Questão 2
    bil_1, bil_2 = bilinear(1, 4)
    img_gs_row, img_gs_col = grayscale_filter(img_gs, bil_1, bil_2)
    grayscale_save(img_gs_row, img_gs_col)
    img_rgb_row, img_rgb_col = rgb_filter(img_color, bil_1, bil_2)
    rgb_save(img_rgb_row, img_rgb_col)
    # Questão 3
    img_gs_derivative = derivative(img_gs_col)
    grayscale_der_save(img_gs_derivative)
    img_rgb_derivative = derivative(img_rgb_col)
    rgb_der_save(img_rgb_derivative)
    # Questão 4
    img_gs_ap = grayscale_allpass(img_gs, 1 / 2, 3584)
    grayscale_allpass_save(img_gs_ap)
    img_rgb_ap = rgb_allpass(1 / 2, 3584)
    rgb_allpass_save(img_rgb_ap)
    # Questão 5
    edge_gs = edge_gs_detect(img_gs, 1, 3)
    edge_gs_save(edge_gs)
    edge_rgb = edge_rgb_detect(img_color, 1, 3)
    edge_rgb_save(edge_rgb)
