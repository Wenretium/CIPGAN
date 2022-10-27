from tkinter import *
from tkinter import filedialog, ttk
import tkinter as tk
import threading
import time
import os
from PIL import Image, ImageTk
from demoTest import demoTest
from demo_files.signText import signText

def showImage(img_path):
    img_open = Image.open(img_path)
    w, h = img_open.size
    img_open = resize(w, h, w_box, h_box, img_open)
    img_png = ImageTk.PhotoImage(img_open)
    return img_png

def transfer(model_name):
    thresh = s1.get()
    print('---------已选择模型{}-----------'.format(model_name))
    global onshow_path
    if onshow_path == 'NOINPUT':
        output_content.set('没有选择输入图片')
    model_name = model_name
    input_name = onshow_path.split('/')[-1]
    # if not os.path.exists(result_path):  # 没有生成过
    #     # 运行模型
    #     demoTest(model_name, onshow_path, input_name, thresh)
    #     print(onshow_path)
    # 运行模型
    demoTest(model_name, onshow_path, input_name, thresh)

    # 展示图片
    global result_png
    result_path = 'demo_files/results/{}_{}_fake.png'.format(model_name, input_name.split('.')[0])
    result_png = showImage(result_path)
    label_img_right['image'] = result_png
    output_content.set('生成模型 {} 的结果'.format(model_name))

def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
    '''
    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def openImage():
    global img_png
    img_path = filedialog.askopenfilenames(initialdir='./datasets/CIP_dataset1_process/testA')
    try:
        img_path = img_path[0] # 如果选择多张图像，只取第一张处理
        img_open = Image.open(img_path)
        w, h = img_open.size
        img_open = resize(w, h, w_box, h_box, img_open)
        img_png = ImageTk.PhotoImage(img_open)
        label_img_left['image'] = img_png
        global input_path
        global onshow_path
        input_path = img_path
        onshow_path = img_path
        input_content.set('已选择图像 {}'.format(img_path))
    except Exception:
        print('未选择图像')

def sign():

    # 输入文本
    text = input_text.get()  # 调用get()方法，将Entry中的内容获取出来

    # 按钮操作
    global flag_sign
    global input_path
    global onshow_path  # path用作图像间切换
    global onshow_png  # png用作函数结束后图像显示

    flag_sign = not flag_sign

    if flag_sign:
        # 留下署名
        word_size = s2.get()
        onshow_path = signText(input_path, text, (10, 10), word_size)
        onshow_png = showImage(onshow_path)
        label_img_left['image'] = onshow_png
        btn_set5['text'] = '取消署名'

    else:
        # 取消署名
        onshow_path = input_path
        onshow_png = showImage(input_path)
        label_img_left['image'] = onshow_png
        btn_set5['text'] = '署名'

    window.update()



if __name__ == '__main__':

    window = Tk()
    window.geometry("1200x600")
    window.title("Demo")
    # 预置
    flag_sign = False
    input_path = 'NOINPUT'
    onshow_path = 'NOINPUT'
    text_ori = '寒蝉凄切,对长亭晚,骤雨初歇'
    # 期望图像显示的大小
    w_box = 400
    h_box = 400
    # 标签
    lbl = Label(window, text="中国画风格迁移demo", font=("Arial Bold", 18))
    lbl.place(y=10, x=450, width=300, height=40)
    # 选择图片
    btn = Button(window, text="选择图片", command=openImage)
    btn.place(y=500, x=300, width=200, height=40)
    # 设置按钮
    btn_set1 = Button(window, text="原模型", command=lambda:transfer(model_name='dataset1'))
    btn_set1.place(y=100, x=40, width=130, height=40)
    btn_set2 = Button(window, text="更换数据集", command=lambda:transfer(model_name='dataset2'))
    btn_set2.place(y=200, x=40, width=130, height=40)
    btn_set3 = Button(window, text="+特征层约束", command=lambda:transfer(model_name='dataset1_vgg'))
    btn_set3.place(y=300, x=40, width=130, height=40)
    btn_set4 = Button(window, text="+特征层约束和自注意力", command=lambda:transfer(model_name='dataset1_vgg_atten'))
    btn_set4.place(y=400, x=40, width=130, height=40)
    btn_set5 = Button(window, text="署名", command=lambda:sign())
    btn_set5.place(y=500, x=40, width=100, height=40)
    # 交互显示区域
    input_content = tk.StringVar(value='没有输入图像')  # 这个就是我们创建的容器，类型为字符串类型
    input = tk.Label(window, compound=CENTER, textvariable=input_content, wraplength = 400)  # 用textvariable与容器绑定
    input.place(y=450, x=200, width=400)

    output_content = tk.StringVar(value='没有生成结果')
    output = tk.Label(window, compound=CENTER, textvariable=output_content, wraplength = 400)
    output.place(y=450, x=700, width=400)
    # 图像展示
    label_img_left = Label(window, image='')
    label_img_left.place(y=100, x=200)
    label_img_right = Label(window, image='')
    label_img_right.place(y=100, x=700)
    # 输入区域
    entry_var = tk.StringVar(value='寒蝉凄切,对长亭晚,骤雨初歇')
    input_text = Entry(window, width=20, textvariable=entry_var)
    input_text.place(y=550, x=50, width=300)
    # 拖动条：字体大小
    s2 = Scale(window, from_=10, to=24, resolution=1, orient="horizontal")
    s2.place(y=500, x=150)
    # 拖动条：map阈值
    s1 = Scale(window, from_=0, to=1, resolution=0.02, orient="horizontal")
    s1.place(y=350, x=50)

    # main
    window.mainloop()
