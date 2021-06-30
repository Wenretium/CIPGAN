# 功能：竖排文字 通过模板图片 写入文字到指定位置，并分别保存成新的图片
# 功能说明：根据","换行（也可以根据"\n"换行）
# 环境：PyDev 6.5.0   Python3.5.2
# 说明：PIL仅支持到python2.7，python3要使用PIL需安装pip3 install Pillow
# python2与python3共存配置方法https://www.cnblogs.com/thunderLL/p/6643022.html

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw


def signText(img_path, text, position, word_size=12):
    # 初始化参数
    x = position[0]
    y = position[1]
    word_size = word_size  # 文字大小
    word_css = "demo_files/段宁毛笔行书.ttf"  # 字体文件   行楷
    # STXINGKA.TTF华文行楷   simkai.ttf 楷体  SIMLI.TTF隶书  minijianhuangcao.ttf  迷你狂草    kongxincaoti.ttf空心草

    # 设置字体
    font = ImageFont.truetype(word_css, word_size)

    # 分割得到数组
    im1 = Image.open(img_path)  # 打开图片
    draw = ImageDraw.Draw(im1)

    # draw.text((x, y),s.replace(",","\n"),(r,g,b),font=font) #设置位置坐标 文字 颜色 字体

    right = 0  # 往右位移量
    down = 0  # 往下位移量
    w = 500  # 文字宽度（默认值）
    h = 500  # 文字高度（默认值）
    row_hight = 0  # 行高设置（文字行距）
    word_dir = 0;  # 文字间距
    # 一个一个写入文字

    for k, s2 in enumerate(text):
        if k == 0:
            w, h = font.getsize(s2)  # 获取第一个文字的宽和高
        if s2 == "," or s2 == "\n":  # 换行识别
            right = right + w + row_hight
            down = 0
            continue
        else:
            down = down + h + word_dir
        # print("序号-值", k, s2)
        # print("宽-高", w, h)
        # print("位移", right, down)
        # print("坐标", x + right, y + down)
        draw.text((x + right, y + down), s2, (0, 0, 0), font=font)  # 设置位置坐标 文字 颜色 字体

    # 保存图像
    # 以时间戳来命名修改后的图像（可能短时间有多个版本），防止命名重复覆盖
    import time
    time_tuple = time.localtime(time.time())
    signs_path = 'demo_files/signs/'+img_path.split('/')[-1].split('.')[0]+'_sign_{}_{}_{}.png'.format(time_tuple[3],time_tuple[4],time_tuple[5])
    im1.save(signs_path)

    del draw  # 删除画笔
    im1.close()

    return signs_path
    img = Image.open(img_path)
    noise_numpy = np.random.rand(img.size[1],img.size[0],3)
    print(noise_numpy)
    img += noise_numpy
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    print('dddddd',img_tensor.shape)