import platform as plat
import os
import wav
import numpy as np
import random

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
        
        self.datapath = abspath_file + self.slash + relpath + self.slash

        self.dic_wavlist_thchs30 = {}
        self.dic_symbollist_thchs30 = {}
        self.dic_wavlist_stcmds = {}
        self.dic_symbollist_stcmds = {}
        self.DataNum_Total = 0
        self.DataNum_thchs30 = 0
        self.DataNum_stcmds = 0
        self.LoadDataList()
        if (self.GetDataNum()!=1):
            print("[message]音频数量不等于标签数量")
        self.list_symbol = self.GetSymbolList()
        #print(self.Symbol2num('xue2'))

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
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'train.wav.lst'
            filename_wavlist_stcmds = 'st-cmds' + self.slash + 'train.wav.txt'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'train.syllable.txt'
            filename_symbollist_stcmds = 'st-cmds' + self.slash + 'train.syllable.txt'
        elif(self.type == 'dev'):
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'cv.wav.lst'
            filename_wavlist_stcmds = 'st-cmds' + self.slash + 'dev.wav.txt'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'cv.syllable.txt'
            filename_symbollist_stcmds = 'st-cmds' + self.slash + 'dev.syllable.txt'
        elif(self.type == 'test'):
            filename_wavlist_thchs30 = 'thchs30' + self.slash + 'test.wav.lst'
            filename_wavlist_stcmds = 'st-cmds' + self.slash + 'test.wav.txt'
            filename_symbollist_thchs30 = 'thchs30' + self.slash + 'test.syllable.txt'
            filename_symbollist_stcmds = 'st-cmds' + self.slash + 'test.syllable.txt'
        else:
            filename_wavlist = '' # 默认留空
            filename_symbollist = ''
        # 读取数据列表，wav文件列表和其对应的符号列表
        self.dic_wavlist_thchs30,self.list_wavname_thchs30 = self.get_wav_list(self.datapath + filename_wavlist_thchs30)
        self.dic_wavlist_stcmds,self.list_wavname_stcmds = self.get_wav_list(self.datapath + filename_wavlist_stcmds)
        
        self.dic_symbollist_thchs30,self.list_symbolname_thchs30 = self.get_symbol_list(self.datapath + filename_symbollist_thchs30)
        self.dic_symbollist_stcmds,self.list_symbolname_stcmds = self.get_symbol_list(self.datapath + filename_symbollist_stcmds)
        self.DataNum_Total = self.GetDataNum()

    def GetData(self,wav_num):
        '''
        读取一个数据，返回输入矩阵输出label
        '''
        if(wav_num<self.DataNum_thchs30):
            wavname = self.list_wavname_thchs30[wav_num]
            wav_filename=self.dic_wavlist_thchs30[wavname]
            symbol=self.dic_symbollist_thchs30[wavname]
        else:
            wavname=self.list_wavname_stcmds[wav_num-self.DataNum_thchs30]
            wav_filename=self.dic_wavlist_stcmds[wavname]
            symbol=self.dic_symbollist_stcmds[wavname]

        wav_filename=self.datapath+wav_filename
        wave_data,framerate,wavetime=wav.read_wav_data(wav_filename)

        data_output=[]
        for spell in symbol:
            if(spell!='' and spell!=' '):#博士这里总有奇怪的东西
                data_output.append(self.symbol2num(spell))

        #feat_input=wav.GetMFCC(wave_data[0],framerate)
        data_input=wav.Get_nlfeat(wave_data,framerate)
        data_input=data_input.reshape(data_input.shape[0],data_input.shape[1],1)
        #这是为了后面训练输入数据的格式要求，哔了狗！

        data_output=np.array(data_output)

        return data_input,data_output

    @threadsafe_generator#为了线程安全，估计内部是采用多线程
    def nl_speechmodel_generator(self,batch_size=32,audio_length=1600,string_length=64):
        '''
        数据生成函数，，，输入[emmm,200],输出[emmm] length这个emm有机会再改进
        '''

        label=np.zeros((batch_size,1),dtype=np.int16)#这好像是为了CTC的
        while True:
            X=np.zeros((batch_size,audio_length,200,1),dtype=np.float64)#nl的特征，，
            y=np.zeros((batch_size,string_length),dtype=np.int16)

            input_length=np.zeros((batch_size,1),dtype=np.int64);
            label_length=np.zeros((batch_size,1),dtype=np.int64);
            

            for i in range(batch_size):
                ran_num=random.randint(0,self.DataNum_Total-1)
                data_input,data_label=self.GetData(ran_num)#试一下什么情况

                #print(data_input.shape)
                #print(len(data_input))
                #print(data_label.shape)
                input_length[i]=(min(data_input.shape[0] // 8 + 
                                    data_input.shape[0] % 8,200))#这里要控制长度，，再研究下,200是避免超出最大
                #其实这上面是最后一个层数输出的东西，要考虑CTC
                label_length[i]=len(data_label)
                X[i,0:len(data_input)]=data_input;
                y[i,0:(len(data_label))]=data_label;
                
            yield ({'input_data':X,
                   'label':y,
                   'input_length':input_length,
                   'label_length':label_length},{'ctc':label})

    def get_wav_list(self,filename):
        with open(filename,'r',encoding='UTF-8') as file_object:
            file_text = file_object.read()
            file_text=file_text.replace('\\',self.slash)#这两行改文件可能没办法在Linux的服务器上用
            file_text=file_text.replace('/',self.slash)#所以在这里做替换
            file_lines = file_text.split('\n')
            dic_wavlist = {}
            list_wavnum = []
            for line in file_lines: 
                if(line != ''):
                    temp_line = line.split(' ')
                    dic_wavlist[temp_line[0]] = temp_line[1]
                    list_wavnum.append(temp_line[0])
        return dic_wavlist,list_wavnum

    def get_symbol_list(self,filename):
        with open(filename,'r',encoding='UTF-8') as file_object:
            file_text = file_object.read()
            file_text.replace('\\',self.slash)#这两行改文件可能没办法在Linux的服务器上用
            file_text.replace('/',self.slash)#所以在这里做替换
            file_lines = file_text.split('\n')
            dic_symbollist = {}
            list_symbolnum = []
            for line in file_lines: 
                if(line != ''):
                    temp_line = line.split(' ')
                    dic_symbollist[temp_line[0]] = temp_line[1:]
                    list_symbolnum.append(temp_line[0])
        return dic_symbollist,list_symbolnum#感觉这里多余了 19.7.3

    def GetDataNum(self):
        num_w_thchs30 = len(self.dic_wavlist_thchs30)
        num_s_thchs30 = len(self.dic_symbollist_thchs30)
        num_w_stcmds = len(self.dic_wavlist_stcmds)
        num_s_stcmds = len(self.dic_symbollist_stcmds)

        if((num_w_thchs30 == num_s_thchs30) & (num_w_stcmds == num_s_stcmds)):
            self.DataNum_Total = num_w_thchs30 + num_w_stcmds
            self.DataNum_stcmds = num_w_stcmds 
            self.DataNum_thchs30=num_w_thchs30
        else:
            return -1

        return 1

    def GetSymbolList(self):
        '''
            加载拼音符号列表，用于标记符号
            返回一个列表list类型变量
        '''
        symbollist_name = 'dict.txt'
        list_symbol = []
        #list_symbol.append(' ')#0把是
        with open(symbollist_name,'r',encoding='UTF-8-sig') as file_object:#这里涉及一个编码什么BOM的问题所以用 UTF-8-sig
            file_text = file_object.read()
            file_lines = file_text.split('\n')
            for line in file_lines: 
                if(line!=''):
                    symbol_ = line.split('\t')
                    list_symbol.append(symbol_[0])
            list_symbol.append(' ')#keras中 ctc默认最后一个作为分隔符（blank）
            symbolnum = len(list_symbol)
            #print(symbolnum)
        return list_symbol

    def symbol2num(self,symbol):
        if(symbol != ''):
            return self.list_symbol.index(symbol)
        return #不知道咋办，博士是放拼音总数，，，

if(__name__ == '__main__'):
    data = DataSpeech('dataset','train')
    #print(data.symbol2num('xian1'))
    #print(data.list_symbol)
    #print(len(data.list_symbol))
    #data.nl_speechmodel_generator()
    #print(data.datapath)
    #print(data.dic_wavlist_thchs30)
    #print(data.list_wavnum_thchs30)
    #print(data.dic_symbollist_stcmds)
    #print(data.DataNum_Total)
    #print(data.GetData(1))
    #generator_test=data.nl_speechmodel_generator()
    #len(generator_test)#明明没有