import numpy as np
import json
import pandas as pd
import sys

def image(row):
    image = {}
    image["height"] = row.height
    image["width"] = row.width
    image["id"] = row.fileid
    image["file_name"] = row.filename
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[2]
    return category

def annotation(row):
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation

def convert(csv_file, coco_file_destination):
    data = pd.read_csv(csv_file)

    images = []
    categories = []
    annotations = []

    category = {}
    category["supercategory"] = 'none'
    category["id"] = 0
    category["name"] = 'None'
    categories.append(category)

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid']= pd.Categorical(data['class'],ordered= True).codes
    data['categoryid'] = data['categoryid']+1
    data['annid'] = data.index
    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        categories.append(category(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    json.dump(data_coco, open(coco_file_destination, "w"), indent=4)

if __name__ == "__main__":
    args = sys.argv()
    path_to_csv_file = args[0]
    save_coco_path = args[1]
    convert(path_to_csv_file, save_coco_path)
