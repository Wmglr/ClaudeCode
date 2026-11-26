import numpy as np, cv2, json
from PIL import Image
from skimage import color
from scipy import ndimage as ndi
G=json.load(open('04_scripts/geometry.json')); D=G['D']
d=np.load('03_intermediate/placed.npz'); plate=np.load('03_intermediate/plate.npy')/255.0
Fj,aj,Fw,aw,Fm,am=d['Fj'],d['aj'],d['Fw'],d['aw'],d['Fm'],d['am']
H,W=aj.shape
s2l=lambda x: np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
l2s=lambda x: np.where(x<=0.0031308,x*12.92,1.055*np.clip(x,0,None)**(1/2.4)-0.055)
P=json.load(open('04_scripts/grade_params.json'))
# ---------- 1. white balance from the shirt (neutralise towards the woman's blazer cast)
yy,xx=np.mgrid[:H,:W]
shirt=(aj>0.99)&(yy>720)&(yy<1000)&(np.abs(xx-795)<150)
cur=s2l(Fj[shirt]).mean(0)
ref=s2l(np.array([220,218,225])/255.0)          # woman's blazer colour
tgt=ref/ref[1]*cur[1]*P['shirt_exposure']
gain=tgt/cur
gain=1+(gain-1)*P['wb_strength']
Lj=s2l(Fj)*gain
print('WB gains',gain.round(3))
# ---------- 2. skin colourimetry (Lab, skin-hue weighted) towards the two reference faces
sj=l2s(np.clip(Lj,0,1))
lab=color.rgb2lab(np.clip(sj,0,1))
skin_w=np.clip((lab[...,1]-8)/8,0,1)*np.clip((lab[...,2]-8)/8,0,1)*np.clip((lab[...,0]-25)/15,0,1)
skin_w=cv2.GaussianBlur(skin_w,(0,0),1.5)*(aj>0.5)
lab[...,1]=lab[...,1]*(1-skin_w*(1-P['skin_a']))
lab[...,2]=lab[...,2]*(1-skin_w*(1-P['skin_b']))
lab[...,0]=lab[...,0]+skin_w*P['skin_L']
sj=np.clip(color.lab2rgb(lab),0,1)
# ---------- 3. lighting: key light from camera-left (as on the duo), gentle fall-off, local dodge&burn on the face
cx=795
gx=1-P['side_grad']*np.clip((xx-cx)/300,-1,1)
gy=1-P['fall_off']*np.clip((yy-500)/900,0,1)
# face dodge/burn: brighten forehead / left cheek, darken under-jaw & right cheek edge
face_c=(cx+0,300+0)
fx=(xx-cx)/110.0; fy=(yy-330)/150.0
dodge=np.exp(-((fx+0.35)**2+(fy+0.25)**2)/1.2)*P['face_dodge']
burn=np.exp(-((fx-0.9)**2+(fy-0.1)**2)/0.9)*P['face_burn'] + np.exp(-((fx)**2+((yy-455)/40.0)**2)/1.2)*P['jaw_burn']
facem=cv2.GaussianBlur(((yy<500)&(np.abs(xx-cx)<135)).astype(np.float32),(0,0),12)
gain_map=(gx*gy)*(1+(dodge-burn)*facem)
sj=l2s(np.clip(s2l(sj)*gain_map[...,None],0,1))
# ---------- 4. contact occlusion from the two neighbours (soft, only where they are really close)
nb=np.maximum(aw,am)
occ=cv2.GaussianBlur(nb.astype(np.float32),(0,0),P['occ_sigma'])
occ=np.clip(occ-0.35,0,1)/0.65
sj=l2s(np.clip(s2l(sj)*(1-P['occ_strength']*occ[...,None]),0,1))
# ---------- 5. optical match: modest local sharpening + subtle luma grain
bl=cv2.GaussianBlur(sj.astype(np.float32),(0,0),1.3)
sj=np.clip(sj+(sj-bl)*P['sharp'],0,1)
rng=np.random.default_rng(7)
n=cv2.GaussianBlur(rng.normal(0,1,(H,W)).astype(np.float32),(0,0),0.6); n/=n.std()
sj=np.clip(sj+n[...,None]*P['grain']/255.0*(aj[...,None]>0.5),0,1)
# ---------- 6. light wrap of the backdrop on the young man's silhouette (3-5 px)
edge=np.clip(aj-cv2.GaussianBlur(aj.astype(np.float32),(0,0),4.0),0,1)*2
wrap=cv2.GaussianBlur(plate.astype(np.float32),(0,0),7)
sj=sj*(1-P['wrap']*edge[...,None])+wrap*P['wrap']*edge[...,None]
Fj2=sj
# ---------- composite (bg < young man < woman, man)
out=plate.copy()
for F,A in [(Fj2,aj),(Fw,aw),(Fm,am)]:
    out=F*A[...,None]+out*(1-A[...,None])
np.save('03_intermediate/final_float.npy',out.astype(np.float32)); np.save('03_intermediate/young_graded.npy',Fj2.astype(np.float32))
Image.fromarray((np.clip(out,0,1)*255+0.5).astype(np.uint8)).save('03_intermediate/v2.png')
print('rendered')
