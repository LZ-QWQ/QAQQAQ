#这是读取网上别人的文件，考虑到thchs30以及stcmds的变调问题出此下策
import re

def get_pinyin(path_filename,save_path_wav,save_path_symbol):
		with open(path_filename,'r',encoding='UTF-8') as file_object,\
        open(save_path_wav,'w',encoding='UTF-8') as file_object2,\
		open(save_path_symbol,'w',encoding='UTF-8') as file_object3:
			temp_string=file_object.readlines()
			for temp_line in temp_string:
				temp=temp_line.split('\t')
				name=temp[0].split('/')
				file_object2.write(name[-1][0:-4]+' '+temp[0]+'\n')
				temp_pinyin_str=add_qingsheng(temp[1])
				file_object3.write(name[-1][0:-4]+' '+temp_pinyin_str+'\n')

def add_qingsheng(str):#这个地方没有规避打多空格导致空元素的情况 已手动修改
	list_pinyin=str.split(' ')
	for i in range(0,len(list_pinyin)):
		if re.search('\d',list_pinyin[i]):pass
		else:list_pinyin[i]+='5'
	return ' '.join(list_pinyin)

if __name__=='__main__':
	path_filename='data\\stcmd.txt'
	save_path_wav='data\\wav_all.txt'
	save_path_symbol='data\\symbol_all.txt'
	get_pinyin(path_filename,save_path_wav,save_path_symbol)
	path_filename='data\\thchs_train.txt'
	save_path_wav='data\\wav_train.txt'
	save_path_symbol='data\\symbol_train.txt'
	get_pinyin(path_filename,save_path_wav,save_path_symbol)
	path_filename='data\\thchs_test.txt'
	save_path_wav='data\\wav_test.txt'
	save_path_symbol='data\\symbol_test.txt'
	get_pinyin(path_filename,save_path_wav,save_path_symbol)
	path_filename='data\\thchs_dev.txt'
	save_path_wav='data\\wav_dev.txt'
	save_path_symbol='data\\symbol_dev.txt'
	get_pinyin(path_filename,save_path_wav,save_path_symbol)