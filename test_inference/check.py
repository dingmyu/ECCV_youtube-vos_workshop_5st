import os

path = "/mnt/lustre/sunpeng/Research/video-seg-workshop/valid/Annotations"
index = 0

f = open("muti_label.txt","w")

for line in os.listdir(path):
	dir = os.path.join(path, line)
	if len(os.listdir(dir)) > 2:
		index += 1
		print >> f,"%s" % (dir)
		print index
	
