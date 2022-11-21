import os
import pandas as pd
import numpy as np

root_path = r'/root/datasets/normal'
log_path = os.path.join(root_path, 'log.csv')
pred_fps = [350, 450]


class Datasets(object):
    def __init__(self, seq_length=10):
        self.df = pd.read_csv(log_path)
        self.seq_length=seq_length
        self.index = None

    def __len__(self):
        return len(pred_fps) * len(self.df)

    def __getitem__(self, item):
        # self.index = item
        folder_num = item // len(pred_fps)
        fps_num = item % len(pred_fps)
        folder = str(self.df.iloc[folder_num, 0])
        fps = str(pred_fps[fps_num])
        path = os.path.join(root_path, folder)
        # path = os.path.join(path, r'all_particles_' + fps + '.csv')
        self.current_folder_index = folder_num
        self.current_fps = fps
        # return path
        folder = path
        start_fps = int(fps)
        print(folder, str(start_fps), end=' ')
        log_data_list = []
        for i in range(self.seq_length):
            fps = start_fps+ i
            path = os.path.join(folder, "all_particles_"+str(fps)+".csv")
            log_data = pd.read_csv(path, dtype=float).iloc[:, :3].values
            log_data_list.append(log_data)
            # print(str(log_data.shape[0]), end=' ')
        print('')
        stack = np.stack(log_data_list, axis=0)
        # if stack.shape[1] > 46000:
        #     raise
        return stack


    def save_time(self, time):
        self.df.loc[self.current_folder_index, self.current_fps] = time
        self.df.to_csv(r'log.csv', index=False)
