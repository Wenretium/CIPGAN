import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
import torch
import numpy as np
from util.getMap import *


def demoTest(model_name, input_path, input_name, attention_thresh):
    # 指定图片读取路径
    opt = TestOptions().parse_demo()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # 修改设置
    img_path = input_path
    model_path = 'demo_files/models/'+model_name
    opt.no_dropout = True
    if 'atten' in model_name:
        opt.self_attention = True
        opt.self_attention_thresh = attention_thresh
    else:
        opt.self_attention = False
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup_demo(opt, model_path)               # regular setup: load and print networks; create schedulers

    # 将图像转为符合原输入的格式，data
    from torchvision import transforms
    from PIL import Image
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(img_path)
    # # 尝试往输入图像加入噪声
    # noise_numpy = np.random.rand(img.size[1],img.size[0],3)*100
    # img += noise_numpy

    img_tensor = trans(img).type(torch.FloatTensor)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    if opt.self_attention:
        # 处理出 map
        transform_map = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
        map = transform_map(getMap(img, attention_thresh))
        map = torch.unsqueeze(map, dim=0)

        data = {'A':img_tensor,'A_paths':img_path,'A_map':map}
    else:
        data = {'A': img_tensor, 'A_paths': img_path}

    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results

    # save image
    image_dir = 'demo_files/results/'

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        save_path = image_dir+'{}_{}_{}.png'.format(model_name, input_name.split('.')[0], label)
        print('生成图像：',save_path)
        util.save_image(im, save_path, aspect_ratio=1.0)



