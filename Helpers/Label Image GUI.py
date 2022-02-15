from tkinter import *
import tkinter.ttk as tpx
from PIL import Image, ImageTk, ImageOps
import os
import cv2

def key(event):
        nextImage()
    
def nextImage():
    if(imgcounter>=totalfiles):
        quit()
    imgpath=globals()['path']+str(FileName[globals()['imgcounter']])
    imgnew= cv2.imread(imgpath)
    cv2.imwrite(f"{pathx}LabelledWord/{textlabel.get()}_{str(FileName[globals()['imgcounter']])}",imgnew)
    
    globals()['imgcounter']+=1
    globals()['cnt']+=1
    # print(FileName[globals()['imgcounter']])
    textlabel.delete(0,END)
    try:
        imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
        width=imo.size[0]
        height=imo.size[1]
        imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
        imglabel.configure(image=imgtk,height=height,width=width)
        imglabel.image = imgtk
        imglabel.grid(column=0, row=0,padx=0)
    except:
        pass

def skipImage():
    if(imgcounter>=totalfiles):
        quit()
    textlabel.delete(0,END)
    imgpath=globals()['path']+str(FileName[globals()['imgcounter']])
    imgnew= cv2.imread(imgpath)
    cv2.imwrite(f"{pathx}LabelledWord/SkippedWord/{str(FileName[globals()['imgcounter']])}",imgnew)
    globals()['imgcounter']+=1
    globals()['cnt']+=1
    # print(FileName[globals()['imgcounter']])
    textlabel.delete(0,END)
    try:
        imo=Image.open(globals()['path']+FileName[globals()['imgcounter']])
        width=imo.size[0]
        height=imo.size[1]
        imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
        imglabel.configure(image=imgtk,height=height,width=width)
        imglabel.image = imgtk
        imglabel.grid(column=0, row=0,padx=0)
    except:
        pass

FileName=[]
pathx=os.getcwd()+'/Desktop/'
os.chdir(pathx) 
if not os.path.exists('LabelledWord'):
    os.makedirs('LabelledWord')
    os.chdir('LabelledWord')
    os.makedirs('SkippedWord')
os.chdir(pathx) 

path=os.getcwd()+'/move/'
filename= os.listdir(path)
imgcounter=0
cnt=0


for name in filename:
    FileName.append(name)
totalfiles=len(FileName)
r = Tk() 
r.configure(background='gray')
r.geometry("650x350")
r.title('Dataset Creator')

label = Label(r,text="Enter Label:")
label.grid(column=0,row=1,padx=5, pady=5,sticky=W)

textlabel = Entry(r)
textlabel.grid(column=0,row=1,padx=100, pady=5,sticky=W)
textlabel.bind('<Return>',key)

nextbtn = Button(r,text="Label Image",command=nextImage)
nextbtn.grid(column=0, row=2,padx=5, pady=5,sticky=W)

skipbtn = Button(r,text="Skip Image",command=skipImage)
skipbtn.grid(column=0, row=2,padx=100, pady=5,sticky=W)


imo=Image.open(path+FileName[imgcounter])
width=imo.size[0]
height=imo.size[1]
imglabel = Label(r,height=height,width=width)
imglabel.configure(background='black')

imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (width,height)))
imglabel.configure(image=imgtk)
imglabel.image = imgtk
imglabel.grid(column=0, row=0,padx=0)
r.mainloop()

