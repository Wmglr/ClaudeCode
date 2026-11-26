import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
from pymatting import estimate_alpha_cf, estimate_foreground_ml
img8=np.array(Image.open('00_originals/jeune_homme.jpg').convert('RGB')); img=img8.astype(np.float64)/255
H,W=img.shape[:2]
pb=np.load('V2/masks/jeune_birefnet.npy'); pi=np.load('01_masks/jeune_prob.npy')
p=0.65*pb+0.35*pi
C=p>0.5
n,l,st,_=cv2.connectedComponentsWithStats(C.astype(np.uint8)); C=np.isin(l,[i for i in range(1,n) if st[i,4]>3000])
# local backdrop model (slowly varying): normalised convolution of pure-backdrop pixels
bgm=(p<0.03)&(ndi.distance_transform_edt(~C)>12)
w=bgm.astype(np.float32); B=np.stack([cv2.GaussianBlur(img8[...,c].astype(np.float32)*w,(0,0),40)/np.maximum(cv2.GaussianBlur(w,(0,0),40),1e-4) for c in range(3)],-1)
print('bg range',B.min((0,1)),B.max((0,1)))
B=np.full(img8.shape,242.0,np.float32)
dist=np.abs(img8.astype(np.float32)-B).max(2)
yy,xx=np.mgrid[:H,:W]
din=ndi.distance_transform_edt(C); dout=ndi.distance_transform_edt(~C)
head=(yy<590)
# head zone: pixel-accurate colour trimap (no network blockiness)
fg_h=(dist>75)&(din>1.5)
chroma=(img8[...,0].astype(int)-img8[...,2].astype(int)); Lum=img8.astype(float).mean(2)
facebox=(np.abs(xx-590)<92)&(yy>395)&(yy<700)
bg_h=((dist<9)&(din<60))|((dout>2)&(dist<45))|((dist<32)&(chroma<26)&~facebox&(yy<500))|((Lum>150)&(chroma<42)&~facebox&~((np.abs(xx-590)<102)&(yy>318))&(yy<500))
# body zone: network mask band
r=4
fg_b=din>r; bg_b=dout>r
tri=np.full((H,W),0.5)
fg=np.where(head,fg_h,fg_b)|((din>25)&head&(np.abs(xx-590)<88)&(yy>330))
bg=np.where(head,bg_h,bg_b)|(dout>14)
tri[fg]=1; tri[bg]=0
a=np.clip(estimate_alpha_cf(img,tri,laplacian_kwargs={'epsilon':1e-6}),0,1)
np.save('V2/masks/jeune_alpha_raw.npy',a.astype(np.float32))
# antialiased, slightly tightened body contour (smooth the network mask contour instead of the noisy matte)
bodyw=np.clip((yy-520)/90,0,1)
Cb=cv2.GaussianBlur(C.astype(np.float32),(0,0),1.5)
crisp=np.clip((Cb-0.56)*3.5+0.5,0,1)
a=a*(1-bodyw)+crisp*bodyw
# pink spill on the right trouser edge (source photo artefact)
r_,g_,b_=[img8[...,c].astype(int) for c in range(3)]
kill=cv2.dilate((((r_-np.maximum(g_,b_))>16)&(yy>1200)&(xx>770)).astype(np.uint8),np.ones((5,5),np.uint8))>0
a=np.where(kill,0,a); a=np.where((~C)&(dout>10),0,a)
# true foreground colours (backdrop removed by ML foreground estimation, then stitched with edge-inward colours on the body)
F=estimate_foreground_ml(img,a)
din2=ndi.distance_transform_edt(a>0.5)
core=(a>0.985)&(din2>3.5); idx=ndi.distance_transform_edt(~core,return_distances=False,return_indices=True); Fn=img[idx[0],idx[1]]
wb=np.clip((3.5-din2)/2.5,0,1)*(~head)
F=F*(1-wb[...,None])+Fn*wb[...,None]
hz0=head&(yy<470)&~facebox&~((np.abs(xx-590)<78)&(yy>295))&(chroma<52)
wzn=np.clip((470-yy)/45,0,1)*hz0
red=np.clip((182-Lum)/(182-112),0,1)**0.8
a=a*(1-wzn*(1-red))     # beige/olive mixed pixels = backdrop contribution, not hair
F=estimate_foreground_ml(img,a)
hz=head&(yy<500)&~facebox&~((np.abs(xx-590)<78)&(yy>295))
hcore=(a>0.97)&(Lum<72)&hz&(ndi.distance_transform_edt(a>0.9)>2)
idh=ndi.distance_transform_edt(~hcore,return_distances=False,return_indices=True); Fh=img[idh[0],idh[1]]
wh=(np.clip((0.97-a)/0.35,0,1)*hz)[...,None]
F=F*(1-wh)+Fh*wh
gap=cv2.dilate(((a<0.6)&hz).astype(np.uint8),np.ones((9,9),np.uint8))>0
wl=(0.85*cv2.GaussianBlur((gap&hz&(chroma<52)).astype(np.float32),(0,0),1.2))[...,None]
F=F*(1-wl)+Fh*wl
np.save('V2/masks/jeune_alpha.npy',a.astype(np.float32)); np.save('V2/layers/jeune_F_raw.npy',F.astype(np.float32))
Image.fromarray((a*255).astype(np.uint8)).save('V2/masks/jeune_alpha.png')
for nm,bgc in [('dark',(0.20,0.23,0.33)),('mid',(0.45,0.5,0.6)),('white',(1,1,1))]:
    c=F*a[...,None]+np.array(bgc)*(1-a[...,None])
    Image.fromarray((np.clip(c,0,1)*255).astype(np.uint8)).crop((440,230,740,520)).resize((900,870),Image.LANCZOS).save(f'V2/work/matte_{nm}.png')
