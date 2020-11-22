#  ___                    _   _
# |_ _|__ _  __ _  ___   | \ | |      Iago Nicolas
#  | |/ _` |/ _` |/ _ \  |  \| |      https://github.com/IagoNicolas
#  | | (_| | (_| | (_) | | |\  |_
# |___\__,_|\__, |\___/  |_| \_(_)    Works with python 3.8.6 64-bit
#           |___/                     @ Thinkpad T480 on 5.9.3-1-MANJARO, 20.1 Mikah.

from warnings import filterwarnings
from scipy.signal import bilinear
from scipy.signal import convolve
from scipy.signal import lfilter
from PIL import Image, ImageOps
from scipy.signal import freqz
from cv2 import COLOR_BGR2RGB
from scipy import ndimage
from cv2 import cvtColor
from cv2 import imread
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import sys
import cv2

filterwarnings("ignore")

gauss = cv2.getGaussianKernel(ksize=9, sigma=0)
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
averg = np.ones((1, 20), dtype="float32") / 20


def image_load(image_file):
    img_color = imread(image_file, 1)
    img_gs = imread(image_file, 0)
    image_check(img_color, img_gs)
    img_color = cvtColor(imread(image_file, 1), COLOR_BGR2RGB)
    return (img_gs, img_color)


def image_check(img_color, img_gs):
    if img_color is None:
        print("\nError opening image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        time.sleep(0)

    if img_gs is None:
        print("Error opening image!")
        print("\nProgram will now exit")
        sys.exit()
    else:
        time.sleep(0)
    return None


def image_save(name, image, mode, angle, side):
    img = Image.fromarray(np.uint8(image), mode)
    if side == 0:
        time.sleep(0)
    if side == 1:
        img = ImageOps.flip(img)
    img.rotate(angle).save("output/" + name)
    return None


def filtro_separavel(kernel, img):
    try:
        img_row = cv2.filter2D(img, -1, kernel, borderType=0)
        kernel.transpose()
        img_col = cv2.filter2D(img_row, -1, kernel, borderType=0)
    except:
        img_row = cv2.filter2D(img, -1, kernel, borderType=0)
        kernel = np.array([kernel])
        kernel.transpose()
        img_col = cv2.filter2D(
            img_row, -1, kernel, borderType=0
        )  # aplicando o mesmo filtro, transposto

    return (img_row, img_col)


def fir_derivativo():
    fir = np.zeros(16)
    for i in range(-8, 9):
        if i == 0:
            fir[8] = 0
        else:
            if i == 8:
                fir[15] = 1 / i
            else:
                fir[i + 8] = 1 / i
    return fir


def all_pass(d1_array, pole, freq_number):
    w, h = freqz([-1 * pole, 1], [1, -1 * pole], freq_number)
    y = lfilter(x=d1_array, b=w, a=h)
    return y


def grayscale_allpass(pole):
    freq_number = 2 * len(img_gs)
    img_gs_df = img_gs
    img_gs_ap_row = all_pass(img_gs_df, pole, freq_number)
    img_gs_ap_row = np.transpose(img_gs_ap_row)
    img_gs_ap_col = all_pass(img_gs_ap_row, pole, freq_number)
    img_gs_ap_row = np.real(img_gs_ap_row)
    img_gs_ap_col = np.real(img_gs_ap_col)

    img_gs_ap_row = np.clip(img_gs_ap_row, 0, 255)
    img_gs_ap_col = np.clip(img_gs_ap_col, 0, 255)

    return img_gs_ap_row, img_gs_ap_col


def rgb_allpass(pole):
    freq_number = 2 * len(img_color)
    img_color_r_df = pd.DataFrame(img_color[:, :, 0])
    img_color_g_df = pd.DataFrame(img_color[:, :, 1])
    img_color_b_df = pd.DataFrame(img_color[:, :, 2])
    # by: Iago N.
    # No reason to do it inside a for, the function already does the row by
    # row reading/processing what was happening was, that i  added overhead
    # to an already unoptimized non threaded/non multiprocessed program.
    img_r_ap_row = all_pass(img_color_r_df, pole, freq_number)
    img_r_ap_row = np.transpose(img_r_ap_row)
    img_r_ap_col = all_pass(img_r_ap_row, pole, freq_number)
    img_g_ap_row = all_pass(img_color_g_df, pole, freq_number)
    img_g_ap_row = np.transpose(img_g_ap_row)
    img_g_ap_col = all_pass(img_g_ap_row, pole, freq_number)
    img_b_ap_row = all_pass(img_color_b_df, pole, freq_number)
    img_b_ap_row = np.transpose(img_b_ap_row)
    img_b_ap_col = all_pass(img_b_ap_row, pole, freq_number)

    img_rgb_ap_row = np.zeros([len(img_color_r_df), len(img_color_r_df), 3])
    img_rgb_ap_col = np.zeros([len(img_color_r_df), len(img_color_r_df), 3])

    img_rgb_ap_row[:, :, 0] = np.real(img_r_ap_row)
    img_rgb_ap_col[:, :, 0] = np.real(img_r_ap_col)
    img_rgb_ap_row[:, :, 1] = np.real(img_g_ap_row)
    img_rgb_ap_col[:, :, 1] = np.real(img_g_ap_col)
    img_rgb_ap_row[:, :, 2] = np.real(img_b_ap_row)
    img_rgb_ap_col[:, :, 2] = np.real(img_b_ap_col)

    img_rgb_ap_row = np.clip(img_rgb_ap_row, 0, 255)
    img_rgb_ap_col = np.clip(img_rgb_ap_col, 0, 255)

    return img_rgb_ap_row, img_rgb_ap_col


def all_pass2():
    polos = np.linspace(-0.75, 0.75, 7)
    for i, wf in enumerate(polos):
        w, h = freqz([-wf, 1.0], [1.0, -wf])
        angles = np.unwrap(np.angle(h))
    return w, h


def inversa_tftd_gs(img):
    w, h = all_pass2()
    inversa = np.fft.ifft(np.abs(h))
    result = np.zeros((512, 1023))

    for i in range(512):
        result[i] = np.convolve(inversa, img[i])

    linha = np.array(result)
    conv = linha.transpose()
    col = np.zeros((1023, 1023))

    for i in range(1023):
        col[i] = np.convolve(inversa, conv[i])

    final = np.array(col).transpose()
    final = np.clip(final, 0, 255)
    return final


def inversa_tftd_rgb(img):
    img_color_r_df = pd.DataFrame(img[:, :, 0])
    img_color_g_df = pd.DataFrame(img[:, :, 1])
    img_color_b_df = pd.DataFrame(img[:, :, 2])
    img_color_r_tftd = inversa_tftd_gs(img_color_r_df)
    img_color_g_tftd = inversa_tftd_gs(img_color_g_df)
    img_color_b_tftd = inversa_tftd_gs(img_color_b_df)

    img_rgb_tftd = np.zeros([len(img_color_r_tftd), len(img_color_r_tftd), 3])
    img_rgb_tftd[:, :, 0] = img_color_r_tftd
    img_rgb_tftd[:, :, 1] = img_color_g_tftd
    img_rgb_tftd[:, :, 2] = img_color_b_tftd

    return img_rgb_tftd


def inversa_z_gs(img):
    w, h = all_pass2()
    l = len(h)
    x_axis = np.arange(l)
    impulse = np.zeros(l)
    impulse[0] = 1.0
    z = lfilter(x=impulse, b=[-0.5, 1.0], a=[1.0, -0.5])
    result = np.zeros((512, 1023))

    for i in range(512):
        result[i] = np.convolve(z, img[i])

    linha = np.array(result)
    conv = linha.transpose()
    col = np.zeros((1023, 1023))
    for i in range(1023):
        col[i] = np.convolve(z, conv[i])

    final = np.array(col).transpose()

    linha = np.clip(linha, 0, 255)
    conv = np.clip(conv, 0, 255)
    final = np.clip(final, 0, 255)

    return final


def inversa_z_rgb(img):
    img_color_r_df = pd.DataFrame(img[:, :, 0])
    img_color_g_df = pd.DataFrame(img[:, :, 1])
    img_color_b_df = pd.DataFrame(img[:, :, 2])
    img_color_r_iz = inversa_z_gs(img_color_r_df)
    img_color_g_iz = inversa_z_gs(img_color_g_df)
    img_color_b_iz = inversa_z_gs(img_color_b_df)

    img_rgb_iz = np.zeros([len(img_color_r_iz), len(img_color_r_iz), 3])
    img_rgb_iz[:, :, 0] = img_color_r_iz
    img_rgb_iz[:, :, 1] = img_color_g_iz
    img_rgb_iz[:, :, 2] = img_color_b_iz

    return img_rgb_iz


def efeito_fantasma_gs(img):
    w, h = all_pass2()
    inversa = np.abs(np.fft.ifft(h))
    result = np.zeros((512, 1023))

    for i in range(512):
        result[i] = np.convolve(inversa, img[i])

    linha = np.array(result)
    conv = linha.transpose()
    col = np.zeros((1023, 1023))

    for i in range(1023):
        col[i] = np.convolve(inversa, conv[i])

    final = np.clip(np.array(col).transpose() * .1, 0, 255)

    return final


def efeito_fantasma_rgb(img):
    img_color_r_df = pd.DataFrame(img[:, :, 0])
    img_color_g_df = pd.DataFrame(img[:, :, 1])
    img_color_b_df = pd.DataFrame(img[:, :, 2])
    img_color_r_ef = efeito_fantasma_gs(img_color_r_df)
    img_color_g_ef = efeito_fantasma_gs(img_color_g_df)
    img_color_b_ef = efeito_fantasma_gs(img_color_b_df)

    img_rgb_ef = np.zeros([len(img_color_r_ef), len(img_color_r_ef), 3])

    img_rgb_ef[:, :, 0] = img_color_r_ef
    img_rgb_ef[:, :, 1] = img_color_g_ef
    img_rgb_ef[:, :, 2] = img_color_b_ef

    return img_rgb_ef


def no_blur(image, arctg):
    val = np.zeros((len(image), len(image)), dtype=np.int32)
    side = arctg * 180.0 / np.pi
    side[side < 0] += 180

    for i in range(1, len(image) - 1):
        for j in range(1, len(image) - 1):
            q = 255
            r = 255
            # 0°    Deg
            if (0 <= side[i, j] < 22.5) or (157.5 <= side[i, j] <= 180):
                q = image[i, j + 1]
                r = image[i, j - 1]
            # 45°   Deg
            elif 22.5 <= side[i, j] < 67.5:
                q = image[i + 1, j - 1]
                r = image[i - 1, j + 1]
            # 90°   Deg
            elif 67.5 <= side[i, j] < 112.5:
                q = image[i + 1, j]
                r = image[i - 1, j]
            # 135°  Deg
            elif 112.5 <= side[i, j] < 157.5:
                q = image[i - 1, j - 1]
                r = image[i + 1, j + 1]
            if (image[i, j] >= q) and (image[i, j] >= r):
                val[i, j] = image[i, j]
            else:
                val[i, j] = 0
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

    return bright


def finding_black(image, white, gray):
    for i in range(1, len(image) - 1):
        for j in range(1, len(image) - 1):
            if image[i, j] == gray:
                if (
                    (image[i + 1, j - 1] == white)
                    or (image[i + 1, j] == white)
                    or (image[i + 1, j + 1] == white)
                    or (image[i, j - 1] == white)
                    or (image[i, j + 1] == white)
                    or (image[i - 1, j - 1] == white)
                    or (image[i - 1, j] == white)
                    or (image[i - 1, j + 1] == white)
                ):
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
    edge = finding_white(edge, high=0.15, low=0.05)
    edge = finding_black(edge, white=100, gray=75)
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


with tqdm(total=5.0, file=sys.stdout) as pbar:
    # Questão 1
    image = "Lenna"
    img_gs, img_color = image_load(image + ".tif")
    pbar.update(1.0)
    # Questão 2
    img_gs_row, img_gs_col = filtro_separavel(gauss, img_gs)
    image_save(image + "_gs_1_row.tif", img_gs_row, "L", 0, 0)
    image_save(image + "_gs_2_col.tif", img_gs_col, "L", 0, 0)
    pbar.update(0.5)
    img_rgb_row, img_rgb_col = filtro_separavel(gauss, img_color)
    image_save(image + "_rgb_1_row.tif", img_rgb_row, "RGB", 0, 0)
    image_save(image + "_rgb_2_col.tif", img_rgb_col, "RGB", 0, 0)
    pbar.update(0.5)
    # Questão 3
    img_gs_der_row, img_gs_der_col = filtro_separavel(fir_derivativo(), img_gs_col)
    image_save(image + "_gs_3_der_row.tif", img_gs_der_row, "L", 0, 0)
    image_save(image + "_gs_4_der_col.tif", img_gs_der_col, "L", 0, 0)
    pbar.update(0.5)
    img_rgb_der_row, img_rgb_der_col = filtro_separavel(fir_derivativo(), img_rgb_col)
    image_save(image + "_rgb_3_der_row.tif", img_rgb_der_row, "RGB", 0, 0)
    image_save(image + "_rgb_4_der_col.tif", img_rgb_der_col, "RGB", 0, 0)
    pbar.update(0.5)
    # Questão 4
    ## Proposição 1
    img_gs_ap_row, img_gs_ap_col = grayscale_allpass(1 / 2)
    image_save(image + "_gs_5_ap_row.tif", img_gs_ap_row, "L", 270, 1)
    image_save(image + "_gs_6_ap_col.tif", img_gs_ap_col, "L", 270, 1)
    pbar.update(0.2)
    img_rgb_ap_row, img_rgb_ap_col = rgb_allpass(1 / 2)
    image_save(image + "_rgb_5_ap_row.tif", img_rgb_ap_row, "RGB", 270, 1)
    image_save(image + "_rgb_6_ap_col.tif", img_rgb_ap_col, "RGB", 270, 1)
    pbar.update(0.5)
    ## Proposição 2
    img_gs_tftd_rc = inversa_tftd_gs(img_gs)
    image_save(image + "_gs_7_tftd_rc.tif", img_gs_tftd_rc, "L", 0, 0)
    img_rgb_tftd_rc = inversa_tftd_rgb(img_color)
    image_save(image + "_rgb_7_tftd_rc.tif", img_rgb_tftd_rc, "RGB", 270, 1)
    pbar.update(0.1)
    img_gs_iz_rc = inversa_z_gs(img_gs)
    image_save(image + "_gs_8_iz_rc.tif", img_gs_iz_rc, "L", 0, 0)
    img_rgb_iz_rc = inversa_z_rgb(img_color)
    image_save(image + "_rgb_8_iz_rc.tif", img_rgb_iz_rc, "RGB", 270, 1)
    pbar.update(0.1)
    img_gs_ef_rc = efeito_fantasma_gs(img_gs)
    image_save(image + "_gs_9_ef_rc.tif", img_gs_ef_rc, "L", 0, 0)
    img_rgb_ef_rc = efeito_fantasma_rgb(img_color)
    image_save(image + "_rgb_9_ef_rc.tif", img_rgb_ef_rc, "RGB", 270, 1)
    pbar.update(0.1)
    # Questão 5
    edge_gs = edge_gs_detect(img_gs, 1, 3)
    image_save(image + "_gs_10_ed.tif", edge_gs, "L", 0, 0)
    pbar.update(0.5)
    edge_rgb = edge_rgb_detect(img_color, 1, 3)
    image_save(image + "_rgb_10_ed.tif", edge_rgb, "RGB", 0, 0)
    pbar.update(0.5)
