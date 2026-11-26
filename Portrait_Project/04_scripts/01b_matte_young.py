import numpy as np, cv2
from PIL import Image
from scipy import ndimage as ndi
from pymatting import estimate_alpha_cf
img8=np.array(Image.open('00_originals/jeune_homme.jpg').convert('RGB'))
img=img8.astype(np.float64)/255
H,W=img.shape[:2]
p=np.load('01_masks/jeune_prob.npy')
C=(p>0.5)
n,l,st,_=cv2.connectedComponentsWithStats(C.astype(np.uint8))
C=np.isin(l,[i for i in range(1,n) if st[i,4]>3000])
# bg model: near-uniform light grey
bg=np.median(img8[(p<0.02)].reshape(-1,3),axis=0).astype(float); print('bg',bg)
dist=np.abs(img8.astype(float)-bg).max(2)           # chebyshev distance to bg colour
din=ndi.distance_transform_edt(C); dout=ndi.distance_transform_edt(~C)
yy=np.arange(H)[:,None]*np.ones((1,W))
# adaptive band: wide in hair zone, narrow along shirt/arms
r=np.interp(yy,[470,600],[11,4])
bgclose=(dist<26)&(dout<40)&(yy<520)                  # background seen through curls (hair zone only)
fg=(din>r)&~bgclose
bgd=(dout>r)|(bgclose&(din<45)&(yy<470))
tri=np.full((H,W),0.5); tri[fg]=1; tri[bgd]=0
al=np.clip(estimate_alpha_cf(img,tri),0,1)
# tighten the soft tail (kills the residual halo)
al=np.clip((al-0.10)/0.82,0,1)
# hair zone: luminance-difference alpha against the uniform backdrop (neutral pixels only -> skin untouched)
L=img8.astype(float).mean(2); chroma=img8[...,0].astype(float)-img8[...,2]
zone=((dout<55)|(din<45))&(yy<500)&(chroma<38)&(L>40)
zone=zone&(ndi.distance_transform_edt(~((p>0.5)&(yy>380)&(yy<640)&(np.abs(np.arange(W)[None,:]-590)<75)))>0) # keep face core out
adiff=np.clip((bg.mean()-L)/(bg.mean()-60.0),0,1)**1.15
al=np.where(zone,np.minimum(al,1.0)*0+adiff*(al>0.02)+np.where(adiff>0.9,1,0)*0,al)
al=np.where(zone&(dout>0)&(p<0.2),0,al)
xx=np.ones((H,1))*np.arange(W)[None,:]
eye=(np.abs(xx-590)<80)&(yy>395)&(yy<445)
leak=(dist<22)&(yy<480)&(chroma<22)&~eye
al=np.where(leak,0,al)
al=np.where(zone&(al<0.25),0,al)
# body zone: smooth contour, then tighten ~1px so no backdrop-contaminated pixel survives
body=np.interp(yy,[470,600],[0,1])
Cb=(al>0.5).astype(np.float32)
ms=cv2.GaussianBlur(Cb,(0,0),1.6)
crisp=np.clip((ms-0.58)*3.5+0.5,0,1)
al=al*(1-body)+crisp*body
al=np.where(~C&(dout>14),0,al)
# pink spill line on the right trouser edge (present in the source photo)
r_,g_,b_=[img8[...,c].astype(int) for c in range(3)]
kill=((r_-np.maximum(g_,b_))>16)&(yy>1200)&(xx>770)
kill=cv2.dilate(kill.astype(np.uint8),np.ones((5,5),np.uint8))>0
al=np.where(kill,0,al)
al=cv2.GaussianBlur(al.astype(np.float32),(0,0),0.6).astype(np.float64)*1.0+0*al
# colour decontamination: edge pixels take the colour of the solid interior (no backdrop-white left)
solid=al>0.985
din2=ndi.distance_transform_edt(al>0.5)
core=solid&(din2>4.0)
idx=ndi.distance_transform_edt(~core,return_distances=False,return_indices=True)
Fnear=img[idx[0],idx[1]]
hairz=(yy<500)
w_body=np.clip((4.0-din2)/2.5,0,1)*(~hairz)                 # 0..1 within ~4px of the silhouette
w_hair=np.clip((1.0-al)/0.25,0,1)*hairz                     # fringe of the curls
w=np.clip(w_body+w_hair,0,1)[...,None]
Fh=Fnear*0.62                                                 # fringe of the hair: darker, less brown-olive
Fd=np.where(hairz[...,None],Fh,Fnear)
F=img*(1-w)+Fd*w
np.save('02_layers/young_F.npy',F.astype(np.float32)); np.save('02_layers/young_A.npy',al.astype(np.float32))
Image.fromarray((al*255).astype(np.uint8)).save('01_masks/jeune_alpha_v2.png')
chk=F*al[...,None]+np.array([0.16,0.2,0.32])*(1-al[...,None])
Image.fromarray((chk*255).astype(np.uint8)).crop((430,230,760,520)).resize((990,870),Image.LANCZOS).save('03_intermediate/z_young_hair.png')
Image.fromarray((chk*255).astype(np.uint8)).crop((270,640,560,900)).resize((870,780),Image.LANCZOS).save('03_intermediate/z_young_sh.png')
