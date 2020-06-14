#改GTA 文件名
#放在目录下，，
import os
from glob import glob

if __name__=="__main__":
    filename_list=glob('*.npy')
    for filename in filename_list:
        os.rename(filename,filename[4:10]+'-feats.npy')