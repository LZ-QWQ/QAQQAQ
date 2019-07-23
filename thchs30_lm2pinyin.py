from pypinyin import pinyin,Style
import re

#改用汉字转拼音写吧
if __name__=='__main__':
    path='lan_model\\'
    with open(path+'lexicon.txt','r',encoding='UTF-8') as file_object,\
        open(path+'lz_lexicon.txt','w',encoding='UTF-8') as file_object2:
        temp_list=file_object.readlines()
        len_templist=len(temp_list)
        for i in range(4,len_templist):
            temp=temp_list[i]
            temp2=temp[:-1].split(' ')#这他喵有个换行符
            word=temp2[0]
            out=pinyin(word,Style.TONE3)
            len_out=len(out)
            res=[]
            for i in range(0,len_out):
                if re.search('\d',out[i][0]):pass
                else: 
                    out[i][0]+='5'
                res.append(out[i][0])
            out_str=' '.join(res)
            end_str=word+' '+out_str+'\n'
            file_object2.write(end_str)



