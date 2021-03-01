import sqlite3
from tqdm import tqdm
import numpy as np
import array
import sys
import math
import os
import multiprocessing
import shutil
import pandas as pd
from scipy.signal import savgol_filter


class Reload:
    def __init__(self, path_pri, path_tra, fold):
        self.path_pri = path_pri
        self.path_tra = path_tra
        self.fold = fold

    def sqlite_read(self, path):
        """
        python读取sqlite数据库文件
        """
        mydb = sqlite3.connect(path)  # 链接数据库
        mydb.text_factory = lambda x: str(x, 'gbk', 'ignore')
        cur = mydb.cursor()  # 创建游标cur来执行SQL语句

        # 获取表名
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        Tables = cur.fetchall()  # Tables 为元组列表

        # 获取表结构的所有信息
        if path[-5:] == 'pridb':
            cur.execute("SELECT * FROM {}".format(Tables[3][0]))
            res = cur.fetchall()[-2][1]
        elif path[-5:] == 'tradb':
            cur.execute("SELECT * FROM {}".format(Tables[1][0]))
            res = cur.fetchall()[-3][1]
        return int(res)

    def read_with_time(self, time):
        conn_pri = sqlite3.connect(self.path_pri)
        result_pri = conn_pri.execute(
            "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")
        chan_1, chan_2, chan_3, chan_4 = [], [], [], []
        t = [[] for _ in range(len(time) - 1)]
        N_pri = self.sqlite_read(self.path_pri)
        for _ in tqdm(range(N_pri)):
            i = result_pri.fetchone()
            if i[-2] is not None and i[-2] >= 6 and i[-1] > 0:
                for idx, chan in zip(np.arange(1, 5), [chan_1, chan_2, chan_3, chan_4]):
                    if i[2] == idx:
                        chan.append(i)
                        for j in range(len(t)):
                            if time[j] <= i[1] < time[j + 1]:
                                t[j].append(i)
                                break
                        break
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return t, chan_1, chan_2, chan_3, chan_4

    def read_vallen_data(self, lower=2, t_cut=float('inf'), mode='all'):
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        if mode == 'all' or mode == 'tra only':
            conn_tra = sqlite3.connect(self.path_tra)
            result_tra = conn_tra.execute(
                "Select Time, Chan, Thr, SampleRate, Samples, TR_mV, Data, TRAI FROM view_tr_data")
            N_tra = self.sqlite_read(self.path_tra)
            for _ in tqdm(range(N_tra), ncols=80):
                i = result_tra.fetchone()
                if i[0] > t_cut:
                    continue
                data_tra.append(i)
        if mode == 'all' or mode == 'pri only':
            conn_pri = sqlite3.connect(self.path_pri)
            result_pri = conn_pri.execute(
                "Select SetID, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts, TRAI FROM view_ae_data")
            N_pri = self.sqlite_read(self.path_pri)
            for _ in tqdm(range(N_pri), ncols=80):
                i = result_pri.fetchone()
                if i[0] > t_cut:
                    continue
                if i[-2] is not None and i[-2] > lower and i[-1] > 0:
                    data_pri.append(i)
                    if i[2] == 1:
                        chan_1.append(i)
                    if i[2] == 2:
                        chan_2.append(i)
                    elif i[2] == 3:
                        chan_3.append(i)
                    elif i[2] == 4:
                        chan_4.append(i)
        data_tra = sorted(data_tra, key=lambda x: x[-1])
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4

    def read_pac_data(self, path, lower=2):
        os.chdir(path)
        dir_features = os.listdir(path)[0]
        data_tra, data_pri, chan_1, chan_2, chan_3, chan_4 = [], [], [], [], [], []
        with open(dir_features, 'r') as f:
            data_pri = np.array([j.strip(', ') for i in f.readlines()[1:] for j in i.strip("\n")])
        for _ in tqdm(range(N_tra), ncols=80):
            i = result_tra.fetchone()
            data_tra.append(i)
        for _ in tqdm(range(N_pri), ncols=80):
            i = result_pri.fetchone()
            if i[-2] is not None and i[-2] > lower and i[-1] > 0:
                data_pri.append(i)
                if i[2] == 1:
                    chan_1.append(i)
                if i[2] == 2:
                    chan_2.append(i)
                elif i[2] == 3:
                    chan_3.append(i)
                elif i[2] == 4:
                    chan_4.append(i)
        data_tra = sorted(data_tra, key=lambda x: x[-1])
        data_pri = np.array(data_pri)
        chan_1 = np.array(chan_1)
        chan_2 = np.array(chan_2)
        chan_3 = np.array(chan_3)
        chan_4 = np.array(chan_4)
        return data_tra, data_pri, chan_1, chan_2, chan_3, chan_4

    def export_feature(self, t, time):
        for i in range(len(time) - 1):
            with open(self.fold + '-%d-%d.txt' % (time[i], time[i + 1]), 'w') as f:
                f.write('SetID, TRAI, Time, Chan, Thr, Amp, RiseT, Dur, Eny, RMS, Counts\n')
                # ID, Time(s), Chan, Thr(μV), Thr(dB), Amp(μV), Amp(dB), RiseT(s), Dur(s), Eny(aJ), RMS(μV), Counts, Frequency(Hz)
                for i in t[i]:
                    f.write('{}, {}, {:.8f}, {}, {:.7f}, {:.7f}, {:.2f}, {:.2f}, {:.7f}, {:.7f}, {}\n'.format(
                        i[0], i[-1], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9]))


class Export:
    def __init__(self, chan, data_tra, features_path):
        self.data_tra = data_tra
        self.features_path = features_path
        self.chan = chan

    def find_idx(self):
        Res = []
        for i in self.data_tra:
            Res.append(i[-1])
        Res = np.array(Res)
        return Res

    def detect_folder(self):
        tar = './waveform'
        if not os.path.exists(tar):
            os.mkdir(tar)
        else:
            print("=" * 46 + " Warning " + "=" * 45)
            while True:
                ans = input(
                    "The exported data file has been detected. Do you want to overwrite it: (Enter 'yes' or 'no') ")
                if ans.strip() == 'yes':
                    shutil.rmtree(tar)
                    os.mkdir(tar)
                    break
                elif ans.strip() == 'no':
                    sys.exit(0)
                print("Please enter 'yes' or 'no' to continue!")

    def export_waveform(self, chan, thread_id=0, status='normal'):
        if status == 'normal':
            self.detect_folder()
        Res = self.find_idx()
        pbar = tqdm(chan, ncols=80)
        for i in pbar:
            trai = i[-1]
            try:
                j = self.data_tra[int(trai - 1)]
            except IndexError:
                try:
                    idx = np.where(Res == trai)[0][0]
                    j = self.data_tra[idx]
                except IndexError:
                    print('Error 1: TRAI:{} in Channel is not found in data_tra!'.format(trai))
                    continue
            if j[-1] != trai:
                try:
                    idx = np.where(Res == trai)[0][0]
                    j = self.data_tra[idx]
                except IndexError:
                    print('Error 2: TRAI:{} in Channel is not found in data_tra!'.format(trai))
                    continue
            sig = np.multiply(array.array('h', bytes(j[-2])), j[-3] * 1000)
            with open('./waveform/' + self.features_path[:-4] + '_{:.0f}_{:.8f}.txt'.format(trai, j[0]), 'w') as f:
                f.write('Amp(uV)\n')
                for a in sig:
                    f.write('{}\n'.format(a))
            pbar.set_description("Process: %s | Exporting: %s" % (thread_id, int(trai)))

    def accelerate_export(self, N=4):
        # check existing file
        self.detect_folder()

        # Multiprocessing acceleration
        each_core = int(math.ceil(self.chan.shape[0] / float(N)))
        pool = multiprocessing.Pool(processes=N)
        result = []
        for idx, i in enumerate(range(0, self.chan.shape[0], each_core)):
            result.append(pool.apply_async(self.export_waveform, (self.chan[i:i + each_core], idx + 1, 'accelerate',)))

        pool.close()
        pool.join()
        print('Finished export of waveforms!')
        return result


def material_status(component, status):
    if component == 'pure':
        if status == 'random':
            # 0.508, 0.729, 1.022, 1.174, 1.609
            idx_select_2 = [105, 94, 95, 109, 102]
            TRAI_select_2 = [4117396, 4115821, 4115822, 4117632, 4117393]
            # -0.264, -0.022
            idx_select_1 = [95, 60]
            TRAI_select_1 = [124104, 76892]

            idx_same_amp_1 = [45, 62, 39, 41, 56]
            TRAI_same_amp_1 = [88835, 114468, 82239, 84019, 104771]

            idx_same_amp_2 = [61, 118, 139, 91, 136]
            TRAI_same_amp_2 = [74951, 168997, 4114923, 121368, 4078227]

    elif component == 'electrolysis':
        if status == 'random':
            # 0.115, 0.275, 0.297, 0.601, 1.024
            idx_select_2 = [50, 148, 51, 252, 10]
            TRAI_select_2 = [3067, 11644, 3079, 28583, 1501]
            # 0.303, 0.409, 0.534, 0.759, 1.026
            idx_select_1 = [13, 75, 79, 72, 71]
            TRAI_select_1 = [2949, 14166, 14815, 14140, 14090]
        if status == 'amp':
            idx_select_2 = [90, 23, 48, 50, 29]
            TRAI_select_2 = [4619, 2229, 2977, 3014, 2345]

            idx_select_1 = [16, 26, 87, 34, 22]
            TRAI_select_1 = [3932, 7412, 16349, 9001, 6300]
        elif status == 'eny':
            idx_select_2 = [79, 229, 117, 285, 59]
            TRAI_select_2 = [4012, 22499, 7445, 34436, 3282]

            idx_select_1 = [160, 141, 57, 37, 70]
            TRAI_select_1 = [26465, 23930, 11974, 9379, 13667]
    return idx_select_1, idx_select_2, TRAI_select_1, TRAI_select_2


def validation(k):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    i = data_tra[k]
    sig = np.multiply(array.array('h', bytes(i[-2])), i[-3] * 1000)
    time = np.linspace(i[0], i[0] + pow(i[-5], -1) * (i[-4] - 1), i[-4])

    thr = i[2]
    valid_wave_idx = np.where(abs(sig) >= thr)[0]
    valid_time = time[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    start = time[valid_wave_idx[0]]
    end = time[valid_wave_idx[-1]]
    duration = (end - start) * pow(10, 6)
    max_idx = np.argmax(abs(sig))
    amplitude = max(abs(sig))
    rise_time = (time[max_idx] - start) * pow(10, 6)
    valid_data = sig[valid_wave_idx[0]:(valid_wave_idx[-1] + 1)]
    energy = np.sum(np.multiply(pow(valid_data, 2), pow(10, 6) / i[3]))
    RMS = math.sqrt(energy / duration)
    count, idx = 0, 1
    N = len(valid_data)
    for idx in range(1, N):
        if valid_data[idx - 1] >= thr > valid_data[idx]:
            count += 1
    # while idx < N:
    #     if min(valid_data[idx - 1], valid_data[idx]) <= thr < max((valid_data[idx - 1], valid_data[idx])):
    #         count += 1
    #         idx += 2
    #         continue
    #     idx += 1
    print(i[0], amplitude, rise_time, duration, energy / pow(10, 4), count, i[-1])


def val_TRAI(data_pri, TRAI):
    # Time, Amp, RiseTime, Dur, Eny, Counts, TRAI
    for i in TRAI:
        vallen = data_pri[i - 1]
        print('-' * 80)
        print('{:.8f} {} {} {} {} {:.0f} {:.0f}'.format(vallen[1], vallen[4], vallen[5], vallen[6],
                                                        vallen[-4], vallen[-2], vallen[-1]))
        validation(i - 1)


def save_E_T(Time, Eny, cls_1_KKM, cls_2_KKM, time, displace, smooth_load, strain, smooth_stress):
    df_1 = pd.DataFrame({'time_pop1': Time[cls_KKM[0]], 'energy_pop1': Eny[cls_KKM[0]]})
    df_2 = pd.DataFrame({'time_pop2': Time[cls_KKM[1]], 'energy_pop2': Eny[cls_KKM[1]]})
    df_3 = pd.DataFrame(
        {'time': time, 'displace': displace, 'load': smooth_load, 'strain': strain, 'stress': smooth_stress})
    df_1.to_csv('E-T_electrolysis_pop1.csv')
    df_2.to_csv('E-T_electrolysis_pop2.csv')
    df_3.to_csv('E-T_electrolysis_RawData.csv')


def load_stress(path_curve):
    data = pd.read_csv(path_curve, encoding='gbk').drop(index=[0]).astype('float32')
    data_drop = data.drop_duplicates(['拉伸应变 (应变 1)'])
    time = np.array(data_drop.iloc[:, 0])
    displace = np.array(data_drop.iloc[:, 1])
    load = np.array(data_drop.iloc[:, 2])
    strain = np.array(data_drop.iloc[:, 3])
    stress = np.array(data_drop.iloc[:, 4])
    sort_idx = np.argsort(strain)
    strain = strain[sort_idx]
    stress = stress[sort_idx]
    return time, displace, load, strain, stress


def smooth_curve(time, stress, window_length=99, polyorder=1, epoch=200, curoff=[2500, 25000]):
    y_smooth = savgol_filter(stress, window_length, polyorder, mode= 'nearest')
    for i in range(epoch):
        if i == 5:
            front = y_smooth
        y_smooth = savgol_filter(y_smooth, window_length, polyorder, mode= 'nearest')

    front_idx = np.where(time < curoff[0])[0][-1]
    rest_idx = np.where(time > curoff[1])[0][0]
    res = np.concatenate((stress[:40], front[40:front_idx], y_smooth[front_idx:rest_idx], stress[rest_idx:]))
    return res


def filelist_convert(data_path, tar=None):
    file_list = os.listdir(data_path)
    if tar:
        tar += '.txt'
    else:
        tar = data_path.split('/')[-1] + '.txt'
    if tar in file_list:
        exist_idx = np.where(np.array(file_list) == tar)[0][0]
        file_list.pop(exist_idx)
    file_idx = np.array([np.array(i[:-4].split('_')[1:]).astype('int64') for i in file_list])
    return file_list, file_idx