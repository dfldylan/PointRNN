import os
import pandas as pd
import numpy as np
from hash import hash
import shutil


root_path = r'/root/datasets/normal'
# log_path = os.path.join(root_path, 'log.csv')
# pred_fps = [350, 450]
output_folder_root = r'/root/datasets/pred/pointrnn'


class Datasets(object):
    def __init__(self, seq_length=10):
        # self.df = pd.read_csv(log_path)
        self.seq_length=seq_length
        self.index = None
        self.start_csv = r'/root/datasets/large/all_particles_390.csv'
        self.pred_loop_num = 30


    def __len__(self):
        return 1

    def __getitem__(self, item):
        start_csv = self.start_csv
        # prepare
        self.fps = int(start_csv.split(r'/')[-1].split(r'.')[0].split(r'_')[-1])
        id = hash(start_csv)
        print(id, start_csv)
        self.pred_folder = r'./pred/' + id
        self.output_folder = os.path.join(output_folder_root, id)
        os.makedirs(self.pred_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        log_data_list = []
        for i in range(self.seq_length):
            log_data = pd.read_csv(start_csv, dtype=float).iloc[:, :3].values
            log_data_list.append(log_data)
            # print(str(log_data.shape[0]), end=' ')
        print('')
        stack = np.stack(log_data_list, axis=0)
        # if stack.shape[1] > 46000:
        #     raise
        return stack

    def write_csv(self, batch_data):
        # concat csv data with solid particles
        df = pd.read_csv(self.start_csv)
        # df = df[df.isFluidSolid == 1]
        # data_fluid = np.concatenate((pos, vel), axis=1)  # [-1, 6]
        df0 = pd.DataFrame(
            {df.columns[0]: batch_data[:, 0], df.columns[1]: batch_data[:, 1], df.columns[2]: batch_data[:, 2], df.columns[7]: df.iloc[:, 7].values},
            columns=df.columns[:18], index=range(batch_data.shape[0]))
        # df0[df.columns[6:18]] = 0
        df = df0

        print(str(self.fps) + ' ok!')
        # write csv -- fast mode start
        df.to_csv(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), index=False)
        if self.output_folder is not None:
            shutil.copy(os.path.join(self.pred_folder, 'all_particles_' + str(self.fps) + '.csv'), self.output_folder)

