import sys
import os

f = open("valid_list.txt","w")


lista = os.listdir("/mnt/lustre/share/sunpeng/video-seg-workshop/valid_all_frames/JPEGImages/")


for line in lista:
	print >> f,"%s" % (line)
