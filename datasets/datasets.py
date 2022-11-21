import os
import numpy as np
import pandas as pd


class Datasets(object):
    def __init__(self, root='/root/datasets/normal', seq_length=10):
        self.seq_length = seq_length
        self.root = root
        self.folder_path, self.down,  self.up = find_files(root, seq_length, range_down=300, range_up=500)


    def __len__(self):
        return len(self.folder_path) * (self.up-self.down)

    def __getitem__(self, _):

        folder = np.random.choice(self.folder_path)
        start_fps = np.random.randint(self.down, self.up)
        print(folder, str(start_fps), end=' ')
        log_data_list = []
        for i in range(self.seq_length):
            fps = start_fps+ i
            path = os.path.join(folder, "all_particles_"+str(fps)+".csv")
            log_data = pd.read_csv(path, dtype=float).iloc[:, :3].values
            log_data_list.append(log_data)
            print(str(log_data.shape[0]), end=' ')
        print('')
        try:
            stack = np.stack(log_data_list, axis=0)
            if stack.shape[1] > 46000:
                raise
            return stack
        except:
            print('again')
            return self.__getitem__(0)

def find_files(root_path, seq_length, range_up=None, range_down=None, scene_num=None):
    folders = []
    exist_folders = [item for item in os.listdir(root_path) if len(item.split(r'.')) < 2]
    if scene_num is not None:
        for each in scene_num:
            each = str(each)
            if each in exist_folders:
                folders.append(each)
            else:
                print("folder "+each+" doesn't exist!")
    else:
        folders = exist_folders

    folder_path = [os.path.join(root_path, item) for item in folders]
    down = 1 if range_down is None else range_down
    up = 1500 if range_up is None else range_up
    up -= seq_length
    if up <=down:
        print('failed')
        exit(-1)
    return folder_path, down, up

def get_fps(file):
    return int(file.split(r'.')[0].split(r'_')[-1])