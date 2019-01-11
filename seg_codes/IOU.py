# -*- coding: utf-8 -*-

import json
import argparse

def IOU(true, pred):
    iou_over_all = 0
    num_images = len(true)
    images_intersection = set(true).intersection(set(pred))
    for image_intersection in images_intersection:
        iou_per_image = 0
        y_true_categories = true[image_intersection]
        num_categories = len(y_true_categories)
        y_pred_categories = pred[image_intersection]
        categories_intersection = set(y_true_categories).intersection(set(y_pred_categories))
        iou_per_category = 0
        for category in categories_intersection:
            y_pred = y_pred_categories[category]
            y_true = y_true_categories[category]
            iou_per_category += compute_iou_pix(y_pred, y_true)
        iou_per_image += iou_per_category
        iou_per_image/=num_categories
        iou_over_all += iou_per_image
    return iou_over_all/num_images

def compute_area(data):
    """
    data: dict
    {x_1,segments_1, x_2:segments_2}
    """
    area = 0
    for segments in data.values():
        for segment in segments:
            area += max(segment)-min(segment)+1
    return area

def compute_iou_pix(y_pred, y_true):
    """
    y_pred, y_true: dict
    {x_1:segments_1, x_2:segments_2,...}
    """
    area_true = compute_area(y_true)
    area_pred = compute_area(y_pred)
    x_intersection = set(y_true).intersection(set(y_pred))
    area_intersection = 0
    for x in x_intersection:
        segments_true = y_true[x]
        segments_pred = y_pred[x]
        for segment_true in segments_true:
            max_segment_true = max(segment_true)
            min_segment_true = min(segment_true)
            del_list = []
            for segment_pred in segments_pred:
                max_segment_pred = max(segment_pred)
                min_segment_pred = min(segment_pred)
                if max_segment_true>=min_segment_pred and min_segment_true<=max_segment_pred:
                    seg_intersection = min(max_segment_true, max_segment_pred)-max(min_segment_true, min_segment_pred)+1
                    area_intersection += seg_intersection
                    if max_segment_true>=max_segment_pred and min_segment_true<=min_segment_pred:
                        del_list.append(segment_pred)
            for l in del_list:
                segments_pred.remove(l)

    area_union = area_true + area_pred - area_intersection

    return area_intersection/area_union

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--path_to_ground_truth', type = str, help = 'path to ground truth', nargs='?')
    parser.add_argument('-p', '--path_to_prediction', type = str, help = 'path to prediction', nargs='?')
    args = parser.parse_args()
    
    with open(args.path_to_ground_truth) as f:
        true = json.load(f)
    with open(args.path_to_prediction) as f:
        pred = json.load(f)
    
    score = IOU(true, pred)
    print(score)