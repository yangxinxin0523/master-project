import glob

import imageio
from skimage import io
import cv2
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import openslide
import os


def get_tissue(img, contour_area_threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphology = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(morphology.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(morphology.copy())

    print("Number of Contours found = " + str(len(contours)))

    tissue_cnts = []
    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > contour_area_threshold:
            # omit the small area contour
            tissue_cnts.append(np.squeeze(np.asarray(cnt)))

    print("Number of cleaned Contours = " + str(len(tissue_cnts)))

    # initialize mask to zero
    mask = np.zeros((img.shape[0], img.shape[1])).astype(img.dtype)
    color = [1]
    mask = cv2.fillPoly(mask, tissue_cnts, color)

    return mask, tissue_cnts


def cut(filename, mask, level, patch_size, save_dir):
    slide = openslide.open_slide(filename)  # 载入全扫描图
    slide.level_dimensions[0]
    w_count = int(slide.level_dimensions[0][0] // patch_size)
    h_count = int(slide.level_dimensions[0][1] // patch_size)

    get_cut = 0
    for x in range(w_count):
        for y in range(h_count):
            top = y * patch_size
            left = x * patch_size
            area = 0
            for i in range(top, top + patch_size + 1):
                for j in range(left, left + patch_size + 1):
                    area += mask[i][j]
            if area > patch_size * patch_size * 0.85:
                print(area)
                region = np.array(
                    slide.read_region((top, left), 0, (patch_size,
                                                       patch_size)))[:,
                         :, :3]
                imageio.imwrite(
                    os.path.join(save_dir, f"{filename}-{top}_{left}_.png"),
                    region)
                get_cut += 1


def cut_patch(filename, mask, level, patch_size, save_dir):
    slide = openslide.open_slide(filename)  # 载入全扫描图
    # slide.level_dimensions[0]
    downsamples = slide.level_downsamples

    w = int(slide.level_dimensions[0][0] * (downsamples[0] / downsamples[
        level]))
    h = int(
        slide.level_dimensions[0][1] * (downsamples[0] / downsamples[level]))

    ratio = int(downsamples[level] // downsamples[0])
    get_cut = 0



    # w_count = int(w // patch_size)
    # h_count = int(h // patch_size)
    #
    # for x in range(w_count):
    #     for y in range(h_count):
    #         top = y * patch_size
    #         left = x * patch_size

    w_count = int(w // 150)
    h_count = int(h // 150)
    for x in range(w_count - 2):
        for y in range(h_count - 2):
            top = y * 150
            left = x * 150

            area = 0
            for i in range(top, top + patch_size + 1):
                for j in range(left, left + patch_size + 1):
                    area += mask[i][j]
            if area > patch_size * patch_size * 0.85:
                print(area)
                region = np.array(
                    slide.read_region((ratio * top, ratio * left), level,
                                      (patch_size,
                                       patch_size)))[:,
                         :, :3]
                imageio.imwrite(
                    os.path.join(save_dir,
                                 f""
                                 f""
                                 f""
                                 f""
                                 f""
                                 f""
                                 f""
                                 f""
                                 f""
                                 f"{os.path.basename(filename).split('.')[0]}-{get_cut}.png"),
                    region)
                get_cut += 1


if __name__ == '__main__':

    INPUT_IMAGE_DIR = '/Users/xinxinyang/data/normal/'
    save_dir = '/Users/xinxinyang/data/normal_cut_1'
    files = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.svs"))
    files = sorted(files)

    for file in files:
        filename = file
        svs_img = openslide.open_slide(filename)

        patch_size = 256
        level = 2

        slide = np.array(svs_img.get_thumbnail(svs_img.level_dimensions[level]))[:,
                :, :3]
        slide_name = f"{os.path.basename(filename).split('.')[0]}.png"
        #imageio.imwrite(slide_name, slide)

        # svs_img = cv2.imread('C3L-00006-21.svs')
        # Img2Grey
        slide_img = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        # plt.imshow(slide_img)
        # plt.show()

        mask, contours = get_tissue(slide_img, contour_area_threshold=50000)
        #plt.imshow(mask)
        #plt.show()

        slide_img_new = slide_img.copy()
        for i, contour in enumerate(contours):
            cv2.drawContours(slide_img_new, contours, i, (76, 177, 34), 15)

        # plt.imshow(slide_img_new)
        # plt.show()

        get_cut = cut_patch(filename, mask, level, patch_size, save_dir)
