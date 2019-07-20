
path='lan_model\\'
filename='word.3gram.lm'
out_name_1='unig.json'
#unig
unig={}
with open(path+filename,'r',encoding='UTF-8') as file_object:
    lines=file_object.readlines()
    all_1=lines[2][8:-1]
    unig['unigram']=int(all_1)
    for i in range() 