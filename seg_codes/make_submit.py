# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import argparse
from PIL import Image

def make_json(annotations_path, categories):
    count = 0
    P = os.listdir(annotations_path)
    json_data = {}
    for p in P:
        if '.png' in p:
            name = p.split('.')[0]+'.jpg'
            json_data[name]={}
            img = Image.open(os.path.join(annotations_path,p)).convert('RGB')
            for category in categories:
                category_segments = {}
                x,y=np.where((np.array(img)==categories[category]).sum(axis=2)==3)
                category_pix = {}
                for i,j in zip(x,y):
                    if i not in category_pix:
                        category_pix[i]=[]
                    category_pix[i].append(j)
                for l in category_pix:
                    segments = []
                    num_segments = 0
                    for i,v in enumerate(sorted(category_pix[l])):
                        if i==0:
                            start=v
                            end=v
                        else:
                            if v==end+1:
                                end = v
                            else:
                                segments.append([int(start),int(end)])
                                start = v
                                end = v
                                num_segments+=1
                    segments.append([int(start),int(end)])
                    category_segments[int(l)]=segments
                if len(category_pix):
                    json_data[name][category]=category_segments
            count+=1
    return json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_annotations', type = str, help = 'path to annotations', nargs='?')
    args = parser.parse_args()
    
    categories = {'car':[0,0,255],'pedestrian':[255,0,0],'lane':[69,47,142],'signal':[255,255,0]}
    
    json_data = make_json(args.path_to_annotations, categories)
    with open('submit.json', 'w') as f:
        json.dump(json_data,f, sort_keys=True,separators=(',', ':'))
    