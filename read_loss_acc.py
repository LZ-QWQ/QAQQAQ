import json

#这里在保存文件的时候放了个巨大的错误1016
if(__name__=='__main__'):
    with open('lz_test2.json','r',encoding='UTF-8') as file_object:
        #open('lz_temp.json',mode='w',encoding='UTF-8') as file_object2:
        
        i=0
        result=[]
        temps=file_object.readlines()
        len_all=len(temps)
        while True:
            temp_string=''
            for j in range(i,i+1016):
                if j==i and j!=0 :
                    temp_string+=temps[j][-2]
                elif j==i+1015 or j==len_all-1:temp_string+=temps[j][0]
                else : temp_string+=temps[j][0:-1]
            temp_re=json.loads(temp_string)
            #temp_out=json.dumps(temp_re)
            #file_object2.write(temp_out+'\n')
            if(j==len_all-1):break
            i+=1015
