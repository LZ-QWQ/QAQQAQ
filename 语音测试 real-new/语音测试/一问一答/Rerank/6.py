import tkinter as tk
import tkinter.font as tf
root = tk.Tk()
root.geometry('300x400')
root.title('测试')
 
text = tk.Text(root,width=20,height=20)
text.pack()
 
char = ['你好','你好呀','你今天真好看','谢谢'] #插入到text中
####更改部分##########
ft = tf.Font(family='微软雅黑',size=10) ###有很多参数
text.tag_add('tag',a) #申明一个tag,在a位置使用
text.tag_config('tag',foreground='red',font =ft ) #设置tag即插入文字的大小,颜色等
 
text.tag_add('tag_1',a1)
text.tag_config('tag_1',foreground = 'blue',background='pink',font = ft)
#foreground字体颜色
#font字体样式,大小等
#background 背景色
 
for i in range(4):
    if i%2 == 0 :
        a = str(i+1)+'.0'
        text.insert(a,char[i]+'\n','tag') #申明使用tag中的设置
    else :
        a1 = str(i+1)+'.0'
        text.insert(a1,char[i]+'\n','tag_1') #申明使用tag__1中的设置
####更改部分##########
root.mainloop()
