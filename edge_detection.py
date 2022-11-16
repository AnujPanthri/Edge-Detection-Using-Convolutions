import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
def get_fil():
    fil=np.array([[-1,0,1],
                  [-1,0,1],
                  [-1,0,1],
                ])
    fil=np.expand_dims(fil,axis=-1)
    fil=np.concatenate([fil,np.moveaxis(np.transpose(fil),0,2)],axis=-1)
    fil=np.expand_dims(fil,axis=-2)
    return fil
def get_img(s,mode='gray'):
    img=PIL.Image.open(s)
    if mode=='gray':
        img=np.array(img.convert("L"))/255
        img=np.expand_dims(img,axis=-1)
    elif mode=='r':
        img=np.array(img)[...,0:1]/255
    elif mode=='g':
        img=np.array(img)[...,1:2]/255
    elif mode=='b':
        img=np.array(img)[...,2:3]/255
    elif mode=='full':
        img=np.array(img)/255
    return img
def show_img(img):
    plt.figure()
    plt.imshow(img,cmap='gray')
def show_fil(fil):
    plt.figure()
    plt.imshow(fil[...,0],cmap='gray')
    plt.figure()
    plt.imshow(fil[...,1],cmap='gray')
def conv(img,fil,p=0,s=1,debug=False):
    ih,iw,idepth=img.shape
    fh,fw,_,fd=fil.shape
    w,h=int((iw-fw-(2*p))/s)+1,int((ih-fh-(2*p))/s)+1
    if debug:
        print(h,w)
    i0=np.repeat(np.arange(fw),fh).reshape(-1,1)
    i1=np.repeat(np.arange(0,(h)*s,s),w).reshape(1,-1)
    i=i0+i1
#     print(i0)
#     print(i1)
#     print(i)
    j0=np.tile(np.arange(fw),fh).reshape(-1,1)
    j1=np.tile(np.arange(0,w*s,s),h).reshape(1,-1)
    j=j0+j1
#     print(j0)
#     print(j1)
#     print(j)
    fil=np.moveaxis(fil,-1,0).reshape(fd,1,1,-1)
#     selected=np.expand_dims(img[i,j,:],axis=0)
    selected=np.moveaxis(img[i,j,:],2,0)
#     selected=img[i,j,:]
    
    if debug:
        print('fil',fil.shape,'img',selected.shape)
    out=fil@selected
    if debug:
        print('out',out.shape)
        print("numberoffilter,imagedepth,height,width")
#     out=np.moveaxis(out.reshape(fd,h,w),0,2)
    out=np.moveaxis(out.reshape(fd,idepth,h,w),(0,1),(3,2))
    return out
class run_edge_detection:
    def __init__(self,path,s=1,p=0,mode="gray",debug=False):
        self.path=path
        self.img=get_img(path,mode)
        self.fil=get_fil()
        self.out=conv(self.img,self.fil,p=p,s=s,debug=debug)
        self.out=abs(self.out)
        print(self.out.shape)
        self.mode=mode
        self.adjust_image()
    def show(self):
        plt.figure()
        plt.axis("off")
        if self.mode=='full':
            plt.imshow(self.out,cmap='gray')
        else:
            plt.imshow(self.out,cmap='gray')
        plt.show()
    def adjust_image(self):
        
#         out_img=np.sum(self.out,axis=-1)
        max_val=np.array([np.max(self.out[...,i]) for i in range(self.out.shape[-1])])
        limit=255
        out_img=self.out
        for i,val in enumerate(max_val):
            if val>limit:
                out_img[...,i]=out_img[...,i]*(val/limit)
    #             print(np.max(arr))
    #             print("max_val>limit",max_val,limit)
            else:
                out_img[...,i]=out_img[...,i]*(limit/val)
    #             print("max_val<limit",max_val,limit) 
        out_img=np.sum(out_img,axis=-1)
        out_img=np.clip(out_img,0,255)
        if self.mode!='full':
            out_img=out_img[...,0]
        self.out=out_img.astype("uint8")
    def save(self):
        save_dir="".join(self.path.rsplit('/',1)[:-1])+"/edges"
        # print(self.path.rsplit('/',1))
        # print(save_dir)
        if not os.path.exists(save_dir):
            print('Creating edges folder.')
            os.mkdir(save_dir)
        save=PIL.Image.fromarray(self.out)
#         print(self.path)
        out_str=self.path.rsplit('.')
        out_str[-2]=out_str[-2]+"_"+self.mode
        out_str='.'.join(out_str)
        
        out_str=out_str.split('/')
        # print(out_str)
        out_str[-1]="edges/"+out_str[-1] # inside a folder edges/abc_full.jpg
        out_str='/'.join(out_str)
        print(out_str)
        save.save(out_str)
    def helper(self):
        txt="""from edge_detection import run_edge_detection
edges=run_edge_detection("C:/Users/admin/Desktop/Penguins.jpg")
edges.show()"""
        return txt
    
def ed(img,fil,p=0,s=1):
    iw,ih=img.shape
    fw,fh,fd=fil.shape
    w,h=int((iw-fw-(2*p))/s)+1,int((ih-fh-(2*p))/s)+1
    out=np.ones((w,h,fd))
    img=np.expand_dims(img,axis=-1)
    for i in range(0,w):
        startw=i*s
        endw=startw+fw
        for j in range(0,h):
            starth=j*s
            endh=starth+fh
#             print('img',img[startw:endw,starth:endh,:].shape,'fil',fil.shape)
            out[i,j,:]=np.sum(np.sum(np.multiply(img[startw:endw,starth:endh,:],fil),axis=0),axis=0)
    return out