from edge_detection import run_edge_detection

import glob
folder='./'

edges_folder=folder[:-1]+"\\edges"
formats=['png','jpeg','jpg','bmp','svg']
alllist=[]
for format in formats:
    alllist.extend(glob.glob(folder+"*."+format))    

print(alllist)
print(edges_folder)
if edges_folder in alllist:
    alllist.remove(edges_folder)
for file in alllist:
    file="/".join(file.rsplit('\\',1))
    img=run_edge_detection(file,mode='full')
    img.save()
#     img.show()