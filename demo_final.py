#-*- coding:utf-8 -*-
import os
import ocr_whole
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import Levenshtein
from tgrocery import Grocery
import pickle
import re
from tgrocery.classifier import *
from stupid_addrs_rev import stupid_revise

def is_alphabet(uchar):
   """判断一个unicode是否是英文字母"""
   if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
      return True
   else:
      return False


def demo_flask(image_file):
    grocery = Grocery('Addrss_NLP')
    model_name=grocery.name
    text_converter=None
    if (os.path.exists(model_name)):
        tgM=GroceryTextModel(text_converter,model_name)
        tgM.load(model_name)
        grocery.model=tgM
        print('load!!!!!')
    else:
        add_file = open('pkl_data/address2.pkl', 'rb')
        other_file = open('pkl_data/others2.pkl', 'rb')
        add_list = pickle.load(add_file)
        other_list = pickle.load(other_file)
        add_file .close()
        other_file .close()
        grocery = Grocery('Addrss_NLP')
        add_list.extend(other_list)
        grocery.train(add_list)
        print (grocery.get_load_status())
        grocery.save()
        # print('train!!!!!!!!')
    addrline = [] 
    t = time.time()
    #result_dir = '/data/share/nginx/html/bbox'
    result_dir = './'
    image = np.array(Image.open(image_file).convert('RGB'))
    result, image_framed = ocr_whole.model(image)
    output_file = os.path.join(result_dir, image_file.split('/')[-1])
    Image.fromarray(image_framed).save(output_file)
    ret_total = ''
    #print(result)
    for key in result:
        string1 = result[key][1]
        print("predict line text :",string1)
        string2 = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*{}[]+", "", string1)
        print("predict line text :",string2)
        ret_total += string1
        
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
    print('\nRecongition Result:\n')
    print(ret_total)
    return output_file,ret_total
