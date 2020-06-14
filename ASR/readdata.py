import platform as plat
import os
import wav
import numpy as np
import random
import time
from LZ_Error import LZ_Error
from tqdm import tqdm

from Threadsafe_iter import threadsafe_generator

class DataSpeech():
    def __init__(self,relpath, type):
        system_type = plat.system()
        abspath_file = os.path.abspath(os.path.dirname(__file__))
        self.type = type

        self.slash = ''
        if(system_type == 'Windows'):
            self.slash = '\\' # 反斜杠
        elif(system_type == 'Linux'):
            self.slash = '/' # 正斜杠
        else:
            print('[Message] Unknown System\n')
            self.slash = '/' # 正斜杠
        
        #self.datapath = abspath_file + self.slash + relpath + self.slash
        self.datapath = relpath + self.slash#其实相对路径就行

        #就全部算在一起吧先
        self.dic_wavlist_all = {}
        self.list_wavname_all = []
        self.dic_symbollist_all = {}
        self.DataNum_Total = 0
        self.LoadDataList()
        if (self.GetDataNum() != 1):
            raise LZ_Error("音频数量不等于标签数量")
        self.list_symbol = self.GetSymbolList()

    def LoadDataList(self):
        '''
        加载用于计算的数据列表
        参数：
            type：选取的数据集类型
                train 训练集
                dev 开发集
                test 测试集
        '''
        # 设定选取哪一项作为要使用的数据集
        if(self.type == 'train'):
            datalist = ['thchs30','st-cmds','aishell','prime','aidata']
        elif(self.type == 'dev'):
            datalist = ['thchs30','aishell','aidata']
        elif(self.type == 'test'):
            datalist = ['thchs30','aishell','aidata']
        else:
            raise LZ_Error('[QAQ]type有问题')

        self.get_wav_list_dic(datalist)
        if self.type=='train':
            np.random.shuffle(self.list_wavname_all)#训练集直接打乱~~
        self.get_symbol_dic(datalist)

        self.GetDataNum()
        print('总数据数：' + str(self.DataNum_Total))
        time.sleep(3)

    def GetData(self,wav_num):
        '''
        读取一个数据，返回输入矩阵输出label
        '''
        if wav_num > self.DataNum_Total:
            raise LZ_Error('没这么多数据哦(⊙o⊙)')

        wav_name = self.list_wavname_all[wav_num]
        wav_filepath = self.dic_wavlist_all[wav_name]
        symbol = self.dic_symbollist_all[wav_name]
        wav_filepath = self.datapath + wav_filepath
        wave_data,framerate,wavetime = wav.read_wav_data(wav_filepath)

        data_output = []
        for spell in symbol:
            if(spell != '' and spell != ' '):#我也不记得这里为什么了,就是为了避免奇奇怪怪的事情吧
                num=self.symbol2num(spell)#spell因为标签集的制作问题最后会有个空格，切割后会产生空字符，，空格就不记得了
                if num!=-1:
                    data_output.append(num)
                else:
                    raise LZ_Error(spell+' 不存在！！标签名为：' + wav_name + 
                                   ';音频路径为：' + wav_filepath)

        #feat_input=wav.GetMFCC(wave_data[0],framerate)
        data_input = wav.get_spectrogram(wave_data,framerate)
        data_input = data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #这是为了后面训练输入数据的格式要求，哔了狗！

        data_output = np.array(data_output)
        
        #emmm提到这来
        if data_input.shape[0] > 1600:
            raise LZ_Error('音频过长，删掉它！！音频名为：' + wav_name + 
                                   ';路径为：' + wav_filepath)
                                   
        return data_input,data_output

    @threadsafe_generator#为了线程安全，估计内部是采用多线程
    def speechmodel_generator(self,batch_size=32,audio_length=1600,string_length=64):
        '''
        数据生成函数，，，输入[emmm,200],输出[emmm] length这个emm有机会再改进
        '''

        label = np.zeros((batch_size,1),dtype=np.int16)#这好像是为了CTC的
        start=0
        end=batch_size#默认+1
        while True:#这个地方好像keras会自己处理生成器
            X = np.zeros((batch_size,audio_length,200,1),dtype=np.float64)
            y = np.zeros((batch_size,string_length),dtype=np.int16)

            input_length = np.zeros((batch_size,1),dtype=np.int64)
            label_length = np.zeros((batch_size,1),dtype=np.int64)
            
            #print(start,end)
            for i in range(start,end):
                data_input,data_label = self.GetData(i)#试一下什么情况

                if data_input.shape[0] > 1600:
                    wav_name = self.list_wavname_all[i]
                    wav_filepath = self.dic_wavlist_all[wav_name]
                    wav_filepath = self.datapath + wav_filepath
                    raise LZ_Error('音频过长，删掉它！！音频名为：' + wav_name + 
                                   ';路径为：' + wav_filepath)

                temp_index=i-start
                input_length[temp_index] = (min(data_input.shape[0] // 8 + data_input.shape[0] % 8,200))#这里要控制长度，，再研究下,200是避免超出最大
                #其实这上面是最后一个层数输出的东西，要考虑CTC
                label_length[temp_index] = len(data_label)
                X[temp_index,0:len(data_input)] = data_input
                y[temp_index,0:(len(data_label))] = data_label
            
            start+=batch_size
            end+=batch_size
            if end>self.DataNum_Total:
                if start>=self.DataNum_Total:
                    start=0
                    end=batch_size
                    np.random.shuffle(self.list_wavname_all)#打乱数据~
                else:
                    end=self.DataNum_Total
                    start=self.DataNum_Total-batch_size

            yield ({'input_data':X,
                   'label':y,
                   'input_length':input_length,
                   'label_length':label_length},{'ctc':label})

    def get_wav_list_dic(self,filename_list):
        for filename in filename_list:
            filepath = self.datapath + filename + self.slash + 'wav_' + self.type + '.txt'
            with open(filepath,'r',encoding='UTF-8') as file_object:
                file_text = file_object.read()
                file_text = file_text.replace('\\',self.slash)#这两行改文件可能没办法在Linux的服务器上用
                file_text = file_text.replace('/',self.slash)#所以在这里做替换
                file_lines = file_text.split('\n')
                print('load ' + filename + '的wav list')
                for line in tqdm(file_lines): 
                    if(line != ''):
                        temp_line = line.split(' ')
                        self.dic_wavlist_all[temp_line[0]] = temp_line[1]#个人觉得可能可以避免重名？其实根本不会发生
                        self.list_wavname_all.append(temp_line[0])
        return

    def get_symbol_dic(self,filename_list):
        for filename in filename_list:
            filepath = self.datapath + filename + self.slash + 'symbol_' + self.type + '.txt'
            with open(filepath,'r',encoding='UTF-8') as file_object:
                file_text = file_object.read()
                file_text.replace('\\',self.slash)#这两行改文件可能没办法在Linux的服务器上用
                file_text.replace('/',self.slash)#所以在这里做替换
                file_lines = file_text.split('\n')
                print('load ' + filename + '的symbol list')
                for line in tqdm(file_lines): 
                    if(line != ''):
                        temp_line = line.split(' ')
                        self.dic_symbollist_all[temp_line[0]] = temp_line[1:]
        return

    def GetDataNum(self):
        wav_num = len(self.dic_wavlist_all)
        symbol_num = len(self.dic_symbollist_all)
        if wav_num == symbol_num:
            self.DataNum_Total = wav_num
        else:
            print('[Error]数据和标签数量有误')
            return -1

        return 1

    def GetSymbolList(self):
        '''
            加载拼音符号列表，用于标记符号
            返回一个列表list类型变量
        '''
        #symbollist_name = 'dict.txt'
        symbollist_name = 'dict_LZ.txt'
        list_symbol = []
        with open(symbollist_name,'r',encoding='UTF-8') as file_object:
            file_text = file_object.read()
            file_lines = file_text.split('\n')
            print('load 拼音符号列表')
            for line in tqdm(file_lines): 
                if(line != ''):
                    symbol_ = line.split('\t')
                    list_symbol.append(symbol_[0])
            list_symbol.append(' ')#keras中 ctc默认最后一个作为分隔符（blank）
        return list_symbol

    def symbol2num(self,symbol):
        if(symbol != ''):
            try:
                num=self.list_symbol.index(symbol)
                return num
            except ValueError:
                print('该拼音不存在')
                return -1
        return 

    def num2symbol(self,vector):
        symbols = []
        for i in vector:
            if i==-1:
                return symbols #无奈之举，批量解码的问题
            symbols.append(self.list_symbol[int(i)])
        return symbols

if(__name__ == '__main__'):
    data = DataSpeech('dataset','train')
    #print(data.symbol2num('xian1'))
    #print(data.list_symbol)
    #print(len(data.list_symbol))
    #data.GetData(1005215)
