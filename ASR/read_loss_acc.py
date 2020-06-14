import json
#其实因为ctc的原因这个acc废掉了。。
if(__name__=='__main__'):
    with open('loss_acc_save\\lz_test2.json','r',encoding='UTF-8') as file_object:
        for line in file_object:
            loss_acc=json.loads(line)


            pass#哪天想画图了再来