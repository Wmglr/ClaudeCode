import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as ndi
import sys; D=int(sys.argv[1]); W=1099+D; H=1431
img=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB')).astype(np.float32)
a=np.array(Image.open('01_masks/duo_alpha.png')).astype(np.float32)/255
person=cv2.dilate((a>0.01).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
gray=img.mean(2)
# --- seam fit
exp=[(49,86),(246,266),(437,447),(614,612)]
seams=[]
for yl,yr in exp:
    xs=[];ys=[]
    for x0 in range(0,1099-30,30):
        sl=slice(x0,x0+30)
        ye=yl+(yr-yl)*(x0+15)/1050
        lo=int(ye-14);hi=int(ye+14)
        if person[max(lo-6,0):hi+6,sl].any(): continue
        prof=gray[lo:hi,sl].mean(1)
        prof=ndi.gaussian_filter1d(prof,1.0)
        xs.append(x0+15); ys.append(lo+int(np.argmin(prof)))
    k,b=np.polyfit(xs,ys,1); seams.append((b,k)); print('seam',round(b,1),round(k,4),len(xs))
pass
sy=lambda x:np.array([b+k*x for b,k in seams])
Yref=sy(540.0)  # flattened coordinate of each seam
# --- flatten original: y'(x,y)
def maps(Wd,xoff_src,seam_fn):
    # returns map for remap: for output (x',y') -> source y. here used both directions
    pass
Hf=H
def flatten(arr,x_range):
    out=np.zeros((Hf,len(x_range))+arr.shape[2:],np.float32)
    yp=np.arange(Hf,dtype=np.float32)
    mapy=np.zeros((Hf,len(x_range)),np.float32); mapx=np.zeros_like(mapy)
    for j,x in enumerate(x_range):
        s=sy(x)
        ys_knots=np.concatenate([[s[0]-1000],s,[s[-1]+3000]]); Yk=np.concatenate([[Yref[0]-1000],Yref,[Yref[-1]+3000]])
        mapy[:,j]=np.interp(yp,Yk,ys_knots); mapx[:,j]=x
    return cv2.remap(arr,mapx,mapy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
def unflatten(arrf,x_range,seam_fn):
    # arrf indexed (y',x_idx). produce (y,x): for each y find y'
    yy=np.arange(H,dtype=np.float32)
    mapy=np.zeros((H,len(x_range)),np.float32); mapx=np.zeros_like(mapy)
    for j,x in enumerate(x_range):
        s=seam_fn(x)
        ys_knots=np.concatenate([[s[0]-1000],s,[s[-1]+3000]]); Yk=np.concatenate([[Yref[0]-1000],Yref,[Yref[-1]+3000]])
        mapy[:,j]=np.interp(yy,ys_knots,Yk); mapx[:,j]=j
    return cv2.remap(arrf,mapx,mapy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
xr=np.arange(1099)
Ff=flatten(img,xr); Pf=flatten(person.astype(np.float32),xr)>0.02
# --- row-wise mirror extension in flattened space
PFc=np.zeros((Hf,W,3),np.float32)
cL=np.full(Hf,575.0); cR=np.full(Hf,855.0)
for y in range(Hf):
    xs=np.where(Pf[y])[0]
    if len(xs): cL[y]=min(xs[0],575); cR[y]=max(xs[-1]+1,855)
cLs=ndi.gaussian_filter1d(ndi.minimum_filter1d(cL,31),10)
cRs=ndi.gaussian_filter1d(ndi.maximum_filter1d(cR,31),10)
def mirror(src_row,axis,n,w,direction):
    out=np.zeros((n,3),np.float32)
    t=np.arange(n); p=t%(2*w); f=np.where(p<w,p,2*w-1-p)
    idx=(axis-1-f) if direction>0 else (axis+f)
    return src_row[idx]
for y in range(620):
    XL=int(cLs[y]); XR=int(cRs[y])
    wl=min(300,XL); wr=min(300,1099-XR)
    PFc[y,:XL]=Ff[y,:XL]
    if XR<1099: PFc[y,XR+D:]=Ff[y,XR:]
    gap=(XR+D if XR<1099 else W)-XL
    if gap<=0: continue
    has_l=wl>=25; has_r=(wr>=25)
    Lm=mirror(Ff[y],XL,gap,wl,+1) if has_l else None
    Rm=mirror(Ff[y],XR,gap,wr,-1)[::-1] if has_r else None
    if has_l and has_r:
        t=np.linspace(0,1,gap)[:,None]; wdt=min(0.5,150/max(gap,1))
        wgt=np.clip((t-(0.5-wdt/2))/wdt,0,1); PFc[y,XL:XL+gap]=Lm*(1-wgt)+Rm*wgt
    elif has_l: PFc[y,XL:XL+gap]=Lm
    elif has_r: PFc[y,XL:XL+gap]=Rm
# canvas seam geometry = continuation of original linear seams
xc=np.arange(W)
plate=unflatten(PFc,xc,sy)
# --- lower region (y>=700): smooth bg model under persons, then use orig left / shifted right
bgmask=(person==0)
P0=img.copy()
for y in range(H):
    ok=np.where(bgmask[y])[0]
    if len(ok)==0:
        P0[y]=P0[y-1] if y>0 else P0[y]; continue
    m=np.zeros(1099,bool); m[ok]=True
    for c in range(3):
        v=np.interp(np.arange(1099),ok,img[y,ok,c]); 
        P0[y,:,c]=np.where(m,img[y,:,c],ndi.uniform_filter1d(v,25))
P0=cv2.GaussianBlur(P0,(0,0),0.6)
P0full=np.zeros((H,W,3),np.float32)
P0full[:,:560]=P0[:,:560]
P0full[:,560+0:W]=0
for x in range(560,W):
    P0full[:,x]=P0[:,min(max(x-D,0),1098)] if x-D>=560 else P0[:,min(x,1098)]
# blend vertical: y<650 plate (flattened-synth), y>=720 P0full
wy=np.clip((np.arange(H)-585)/45.0,0,1)[:,None,None]
final=plate*(1-wy)+P0full*wy
# restore wood grain in the smoothed lower region: anisotropic noise calibrated on clean backdrop (rows 620-780, x<80)
clean=img[620:780,0:80]; hpc=clean-cv2.GaussianBlur(clean,(0,0),sigmaX=9,sigmaY=0.01)
std=np.array([1.9,1.9,2.2],np.float32); print('grain std (robust, measured on clean backdrop ~1.6)',std)
rng=np.random.default_rng(11)
nz=ndi.gaussian_filter(rng.normal(0,1,(H,W)),(0.8,8)); nz/=nz.std()
nz2=ndi.gaussian_filter(rng.normal(0,1,(H,W)),(1.6,30)); nz2/=nz2.std()
HP=(0.8*nz+0.5*nz2)[...,None]*std[None,None,:]*1.0
rr=np.zeros((H,W),bool); rr[:,560+D:]=person[:,560:1099]>0; rr[:,:560]=person[:,:560]>0; rr[:,560:560+D]=True
wy2=np.clip((np.arange(H)-585)/45.0,0,1)[:,None,None]
final=final+HP*wy2*rr[...,None]
np.save(f'V2/work/plate_mirror_D{D}.npy',final)
pass
print('plate ok',final.shape)
