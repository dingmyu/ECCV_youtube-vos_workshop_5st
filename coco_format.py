import os
import sys
import cv2
import json
data = {}
images = []
annotations = []
categories = []

info = {}
info['supercategory'] = "cat"
info['name'] = "cat"
info['id'] = 1

categories.append(info)

img_id = 0
anno_id = 0

count = 0
for folder in os.listdir("JPEGImages"):
    print folder
    print count 
    count += 1
    folder_path = "JPEGImages/"+folder
    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
	if img.shape[0]!=720 or img.shape[1]!=1280:
		img_resized = cv2.resize(img, (1280, 720))
	else:
		img_resized = img
        hei = 720
        wid = 1280
        img = {}
        img['id'] = img_id
        img['file_name'] = file_path.replace("/","_")
        img['width'] = int(wid)
        img['height'] = int(hei)
        images.append(img)
	import shutil
	cv2.imwrite("coco_format_imgs/"+file_path.replace("/","_"), img_resized)
        #shutil.copyfile(file_path, "coco_format_imgs/"+file_path.replace("/","_")) 
        bbox = [1, 2, 3, 4]
        anno = {}
        anno['id'] = anno_id
        anno['iscrowd'] = 0
        anno['bbox'] = bbox
        anno['image_id'] = img_id
        anno['area'] = float(4.0)
        anno['category_id'] = int(1)
        anno_id += 1
        annotations.append(anno)
        img_id += 1
#     break
data['images'] = images
data['annotations'] = annotations
data['categories'] = categories

with open('coco_format_all_test_set.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False)
    
    


