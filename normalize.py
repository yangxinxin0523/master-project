import glob

import imageio
import utils
from vahadane import vahadane
import os

TARGET_PATH = '/Users/xinxinyang/data/target.png'
INPUT_IMAGE_DIR = '/Users/xinxinyang/data/tumor_cut/'
save_dir = '/Users/xinxinyang/data/tumor_cut_1'

files = glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.png"))
files = sorted(files)

for file in files:
    SOURCE_PATH = file
    print(SOURCE_PATH)
    RESULT_PATH = os.path.join(save_dir, os.path.basename(os.path.normpath(
        SOURCE_PATH)))
    print(RESULT_PATH)
    # RESULT_PATH = SOURCE_PATH
    source_image = utils.read_image(SOURCE_PATH)
    target_image = utils.read_image(TARGET_PATH)

    vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)

    Ws, Hs = vhd.stain_separate(source_image)
    vhd.fast_mode = 0;
    vhd.getH_mode = 0;
    Wt, Ht = vhd.stain_separate(target_image)

    img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)

    imageio.imwrite(RESULT_PATH, img)
