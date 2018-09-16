import os
import json
Dir = '/mnt/lustre/dingmingyu/workspace/experiments/youtube/open-reid/examples/data/finetune/images/'
meta = {}
splits = []
Type = os.listdir('.')
Type.remove('get_json.py')
Type.remove('get_data.py')
meta['shot'] = 'multple'
meta['name'] = 'finetune'
meta['num_cameras'] = 2
meta['identities'] = []

query_list = []
num = -1
for itype in Type:
	print itype
	query_list.append(num+1)
	Itype = os.listdir(itype)
	for iitype in Itype:
		num += 1
		Item = os.listdir(itype + '/' + iitype)
		inum = 0
		ilist1 = []
		ilist2 = []
		for index, item in enumerate(Item):
			name =  itype + '/' + iitype + '/' + item
			if index % 2 == 0:
				new_name =  '%08d_00_%04d.png'%(num, inum)
				ilist1.append(new_name)
			else:
				new_name =  '%08d_01_%04d.png'%(num, inum)
				ilist2.append(new_name)
				inum += 1
			os.system("cp %s /mnt/lustre/dingmingyu/workspace/experiments/youtube/open-reid/examples/data/finetune/images/%s" % (name, new_name))
		meta['identities'].append([ilist1,ilist2])
	query_list.append(num)
	#break
trainval_list = range(num+1)
for item in query_list:
	trainval_list.remove(item)
splits.append({"trainval": trainval_list,"query": query_list, "gallery": query_list})

json.dump(meta,open('/mnt/lustre/dingmingyu/workspace/experiments/youtube/open-reid/examples/data/finetune/meta.json','w'))
json.dump(splits,open('/mnt/lustre/dingmingyu/workspace/experiments/youtube/open-reid/examples/data/finetune/splits.json','w'))
