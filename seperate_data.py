import logging
import os
import re
from random import random


def remove_other_files(extension, archive_dir):
    for root, dirs, files in os.walk(archive_dir):
        for current_file in files:

            if not current_file.lower().endswith(extension):

                try:
                    logging.debug("Removing resource: File [%s]", os.path.join(root, current_file))
                    os.remove(os.path.join(root, current_file))
                except OSError:
                    logging.error("Could not remove resource: File [%s]", os.path.join(root, current_file))


def create_category_folder(category0, category1, source_dir):
    for new_dir in ['train', 'predict']:
        for new_category in [category0, category1]:
            abspath_dir = os.path.abspath(os.path.join(source_dir, new_dir, new_category))

            logging.info(
                "Creating resource: Directory [%s]", abspath_dir)
            os.makedirs(abspath_dir)


def label_files(directory, source_dir):
    predict_ratio = 0.2

    for root, dirs, files in os.walk(directory + '/normal_cut_1'):
        for file in files:
            if random() < predict_ratio:
                train_test_dir = 'predict/'
            else:
                train_test_dir = 'train/'

            try:
                logging.debug("Moving %s from %s to %s", file, root,
                    os.path.join(source_dir,
                    train_test_dir, 'normal'))

                os.rename(os.path.join(root, file),
                os.path.join(source_dir, train_test_dir, 'normal', file))

            except OSError:
                logging.error("Could not move %s ", os.path.join(root, file))

    for root, dirs, files in os.walk(directory + '/tumor_cut_1'):
        for file in files:
            if random() < predict_ratio:
                train_test_dir = 'predict/'
            else:
                train_test_dir = 'train/'

            try:
                logging.debug("Moving %s from %s to %s", file, root,
                                os.path.join(source_dir, train_test_dir,
                                             'tumor'))

                os.rename(
                    os.path.join(root, file),
                    os.path.join(source_dir, train_test_dir, 'tumor', file))

            except OSError:
                logging.error("Could not move %s ", os.path.join(root, file))

data_org_directory = '/Users/xinxinyang/data'  # raw data path
data_mdy_directory = '/Users/xinxinyang/imagedata'  # modified data stored path

# remove_other_files('.png', data_org_directory)
create_category_folder('normal', 'tumor', data_mdy_directory)
label_files(data_org_directory, data_mdy_directory)