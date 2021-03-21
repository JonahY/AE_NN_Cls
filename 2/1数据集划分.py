import pandas as pd
import random
import shutil
import os

for i in range(3):
    os.makedirs("train/%d" % i, exist_ok=True)
    os.makedirs("test/%d" % i, exist_ok=True)

data = pd.read_csv("/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train info_cwt.csv", header=None)
data = data.values.argmax(axis=1).tolist()
data = [(f"{i + 1}.jpg", data[i]) for i in range(len(data))]
random.shuffle(data)
train_n = int(0.7 * len(data))
for i in range(len(data)):
    img_name, label = data[i]
    if i > train_n:
        shutil.copy('/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train dataset_cwt/%s' % img_name,
                    'test/%d/' % (label,))
    else:
        shutil.copy('/home/Yuanbincheng/data/Ni-tension test-pure-1-0.01-AE-20201030/train dataset_cwt/%s' % img_name,
                    'train/%d/' % (label,))
