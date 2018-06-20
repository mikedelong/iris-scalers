import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def make_scatter(arg_df, arg_filename):
    fig, axes = plt.subplots(ncols=2)
    column_names = df.columns.values
    combinations = [(column_names[0], column_names[1]), [column_names[2], column_names[3]]]
    for index, combination in enumerate(combinations):
        left = combination[0]
        right = combination[1]
        logger.debug('%s %s' % (left, right))
        axes[index].scatter(arg_df[left], arg_df[right])
    plt.savefig(arg_filename)
    plt.close()


if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    input_folder = '../data/'
    output_folder = '../output/'

    input_folder_exists = os.path.isdir(input_folder)
    if not input_folder_exists:
        logger.warning('input folder %s does not exist. Quitting.' % input_folder)
        quit()
    output_folder_exists = os.path.isdir(output_folder)
    if not output_folder_exists:
        logger.warning('output folder %s does not exist. Quitting.' % output_folder)
        quit()

    train_file = input_folder + 'iris_w_labels.csv'
    df = pd.read_csv(train_file)
    df.drop(['target'], inplace=True, axis=1)
    logger.debug('the original training dataset has shape %d x %d' % df.shape)
    column_names = df.columns.values
    logger.debug(column_names)

    make_scatter(df, output_folder + 'original-scatter.png')

    run_data = {
        'min_max.png': MinMaxScaler,
        'max_abs.png': MaxAbsScaler,
        'standard.png': StandardScaler,
        'robust.png': RobustScaler
    }
    for key, value in run_data.items():
        logger.debug('doing case %s' % key)

        scaled_df = df.copy(deep=True)
        for column in column_names:
            scaler = value()
            scaled_df[[column]] = scaler.fit_transform(X=df[[column]])
        make_scatter(scaled_df, output_folder + key)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
