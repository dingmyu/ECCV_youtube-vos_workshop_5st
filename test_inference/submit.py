import os
import shutil
all = 473

predict = "/mnt/lustre/sunpeng/Research/video-seg-workshop/VS-ReID/results_muti_label/result"
example = "/mnt/lustre/sunpeng/Research/video-seg-workshop/submits/Annotations"
index = 0
for dir in os.listdir(example):
	dir_e = sorted(os.listdir(os.path.join(example, dir)))
	dir_p = sorted(os.listdir(os.path.join(predict, dir)))
	assert len(dir_e) == len(dir_p)
	c = zip(dir_p, dir_e)
	for (p, e) in c:
		f1 = os.path.join(os.path.join(predict, dir), p)
		f2 = os.path.join(os.path.join(example, dir), e)
		#print f1, f2
		import cv2
		a = cv2.imread(f1)
		a = cv2.resize(a, (1280, 720))
		GrayImage=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
		cv2.imwrite(f2, GrayImage)
		#shutil.copy(f1, f2)
	index += 1
	print index
	#break



