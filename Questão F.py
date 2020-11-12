from scipy.signal import symiirorder1
from scipy.signal import savgol_filter
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
    pbar.update(10)
    return None


def grayscale_filter(img_gs, window, poly):
    img_gs_df = pd.DataFrame(img_gs)
    pbar.update(5)
    img_gs_sv_row = np.zeros(shape=[len(img_gs_df), len(img_gs_df)])
    img_gs_sv_col = np.zeros(shape=[len(img_gs_df), len(img_gs_df)])
    pbar.update(5)
    for i in range(0, len(img_gs_df)):
        img_gs_sv_row[i] = symiirorder1(img_gs_df.loc[i, :], window, poly)
    pbar.update(10)
    for j in range(0, len(img_gs_df)):
        img_gs_sv_col[j] = symiirorder1(
            pd.DataFrame(img_gs_sv_row).loc[:, j], window, poly
        )
    pbar.update(10)
    return (img_gs_sv_row, img_gs_sv_col)


def rgb_filter(img_color, window, poly):
    img_color_r_df = pd.DataFrame(img_color[:, :, 0])
    pbar.update(2)
    img_color_g_df = pd.DataFrame(img_color[:, :, 1])
    pbar.update(2)
    img_color_b_df = pd.DataFrame(img_color[:, :, 2])
    pbar.update(2)
    img_r_sv_row = np.zeros(shape=[len(img_color_r_df), len(img_color_r_df)])
    img_r_sv_col = np.zeros(shape=[len(img_color_r_df), len(img_color_r_df)])
    img_g_sv_row = np.zeros(shape=[len(img_color_g_df), len(img_color_g_df)])
    img_g_sv_col = np.zeros(shape=[len(img_color_g_df), len(img_color_g_df)])
    img_b_sv_row = np.zeros(shape=[len(img_color_b_df), len(img_color_b_df)])
    img_b_sv_col = np.zeros(shape=[len(img_color_b_df), len(img_color_b_df)])
    pbar.update(4)
    for i in range(0, len(img_color_r_df)):
        img_r_sv_row[i] = symiirorder1(img_color_r_df.loc[i, :], window, poly)
    pbar.update(3)
    for j in range(0, len(img_color_r_df)):
        img_r_sv_col[j] = symiirorder1(
            pd.DataFrame(img_r_sv_row).loc[:, j], window, poly
        )
    pbar.update(3)

    for i in range(0, len(img_color_g_df)):
        img_g_sv_row[i] = symiirorder1(img_color_g_df.loc[i, :], window, poly)
    pbar.update(3)
    for j in range(0, len(img_color_g_df)):
        img_g_sv_col[j] = symiirorder1(
            pd.DataFrame(img_g_sv_row).loc[:, j], window, poly
        )
    pbar.update(3)
    for i in range(0, len(img_color_b_df)):
        img_b_sv_row[i] = symiirorder1(img_color_b_df.loc[i, :], window, poly)
    pbar.update(3)
    for j in range(0, len(img_color_b_df)):
        img_b_sv_col[j] = symiirorder1(
            pd.DataFrame(img_b_sv_row).loc[:, j], window, poly
        )
    pbar.update(3)

    img_rgb_sv_row = np.zeros([512, 512, 3])
    img_rgb_sv_col = np.zeros([512, 512, 3])
    pbar.update(1)

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
    # img.show()
    img = Image.fromarray(np.uint8(col), "L")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_gs_2_col.tif")
    # img.rotate(270).show()
    pbar.update(5)
    return None


def grayscale_der_save(der):
    img = Image.fromarray(np.uint8(der), "L")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_gs_der.tif")
    return None


def rgb_der_save(der):
    img = Image.fromarray(np.uint8(der), "RGB")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_rgb_der.tif")
    return None


def rgb_save(row, col):
    row = np.clip(row, 0, 255)
    img = Image.fromarray(np.uint8(row), "RGB")
    img.save("output/Lenna_rgb_1_row.tif")
    # img.show()
    col = np.clip(col, 0, 255)
    img = Image.fromarray(np.uint8(col), "RGB")
    img = ImageOps.flip(img)
    img.rotate(270).save("output/Lenna_rgb_2_col.tif")
    # img.rotate(270).show()
    pbar.update(5)
    return None


def edge_detect(img, sigma):
    edges = feature.canny(img, sigma=sigma)

    fig, (ax1) = plt.subplots(
        nrows=1, ncols=1, figsize=(8, 8)
    )
    ax1.imshow(edges, cmap=plt.cm.gray)
    ax1.axis("off")
    fig.tight_layout()
    plt.savefig("output/Lenna_canny_gs.tif")
    # plt.show()
    return None


def all_pass(z, z_0):
    h_z = (pow(z, -2) - 2 * z_0 * pow(z, -1) + pow(abs(z), 2)) / (
        1 - 2 * z_0 * pow(z, -1) + pow(abs(z_0), 2) * pow(z, -2)
    )
    return h_z


def derivative(img):
    img_der_hor = np.diff(img, n=1, axis = 0)  # horizontal derivative
    img_der_ver = np.diff(img_der_hor, n=1, axis = 1)  # vertical derivative
    return img_der_ver


with tqdm(total=100, file=sys.stdout) as pbar:
    # Questão 1
    img_gs, img_color = image_load("Lenna.tif")
    # Questão 2
    img_gs_row, img_gs_col = grayscale_filter(img_gs, .3, .5)
    grayscale_save(img_gs_row, img_gs_col)
    img_rgb_row, img_rgb_col = rgb_filter(img_color, .3, .5)
    rgb_save(img_rgb_row, img_rgb_col)
    # Questão 3
    img_gs_derivative = derivative(img_gs_col)
    grayscale_der_save(img_gs_derivative)
    img_rgb_derivative = derivative(img_rgb_col)
    rgb_der_save(img_rgb_derivative)
    img_rgb_derivative = derivative(img_rgb_col)
    # Questão 5
    edge_detect(img_gs, 1)
    pbar.update(20)
