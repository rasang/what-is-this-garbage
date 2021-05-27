import pandas as pd
from shutil import copyfile
from tensorflow import lite

#%% 将文件复制封装成函数

def copy_file_order_by_txt(text_name, dest_dir):
    data = pd.read_csv(text_name, header=None, sep=' ')
    base_dir = "./archive/garbage classification/"
    for indexs in data.index:
        line = data.loc[indexs].values[0:]
        name = line[0]
        garbage_type = None
        if name[0:2] == "pa":
            garbage_type = "paper/"
        elif name[0:2] == "ca":
            garbage_type = "cardboard/"
        elif name[0:2] == 'gl':
            garbage_type = "glass/"
        elif name[0:2] == "me":
            garbage_type = "metal/"
        elif name[0:2] == "pl":
            garbage_type = "plastic/"
        else:
            garbage_type = "trash/"
        copyfile(base_dir + garbage_type + name, dest_dir + garbage_type + name)