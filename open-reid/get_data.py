data_dir = '/mnt/lustre/sunpeng/Research/video-seg-workshop/train/'
import json
f = open(data_dir + 'meta.json')
data = json.load(f)
data = data['videos']
import cv2
all_dict = {}
import os
import numpy as np
for key, value in data.items():
#	print key
	video = value['objects']
	for obj, frame in video.items():
		if frame['category'] not in all_dict:
			all_dict[frame['category']] = []
		all_dict[frame['category']].append((key, int(obj), frame['frames']))

for key, keylist in all_dict.items():
	print key
	os.mkdir(key)
	for index, (video, num, mylist) in enumerate(keylist):
		print index
		os.mkdir(key + '/' + str(index))
		a = []
		new_list = []
		for item in mylist:
			label = cv2.imread('/mnt/lustre/sunpeng/Research/video-seg-workshop/mat_label/train/' +video + '/' + item +'.png')[:,:,0]
			a.append((label == num).sum())
		maxa = max(a)
		for nnn, item in enumerate(mylist):
			if a[nnn] > 0.3 * maxa:
				new_list.append(item)
		for nnn, item in enumerate(new_list):
			pic = data_dir + 'JPEGImages/' +video + '/' + item +'.jpg'
			picture = cv2.imread(pic)
			label = cv2.imread('/mnt/lustre/sunpeng/Research/video-seg-workshop/mat_label/train/' +video + '/' + item +'.png')[:,:,0]
			y, x = np.where(label == num)
			w = x.max()- x.min()
			h = y.max() - y.min()
			x1,x2,y1,y2 =  max(x.min()- int(0.05*w),0),min(x.max()+int(0.05*w),label.shape[1]-1), max(y.min()-int(0.05*h),0), min(y.max()+int(0.05*h),label.shape[0]-1)
			print x1,x2,y1,y2
			cv2.imwrite(key + '/' + str(index) + '/%03d.png' % nnn, picture[y1:y2,x1:x2,:])
	#os.mkdirs(key)
	
		#print frame['category'], obj
		#print len(frame['frames'])

