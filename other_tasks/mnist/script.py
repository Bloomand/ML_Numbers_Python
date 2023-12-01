# -*- coding: utf-8 -*-
import os
import pandas as pn
def save_labels(folder):
    path = folder   #рath это имя папки, в которой и работает функция(вводимая переменная)
    fds = os.listdir(path)
    out=[]
    names=['Index','fname','label']
    for img in fds:   #проход по папке
        img=os.path.splitext(img)
        img=list(img)
        q1=img[0]
        q1=q1.split('-num')
        img[0]=int(q1[0])
        img.append(img[1])
        img[1] = q1[1]
        out.append([img[0],img[1],img[2]])
    dt = pn.DataFrame(out,columns=names)
    dt.to_csv('text',index=False)


if __name__=='__main__':
    b = str(input())
    save_labels(b)