import numpy as np, cv2, json, sys
from PIL import Image
from skimage import color
from scipy import ndimage as ndi
s2l=lambda x: np.where(x<=0.04045,x/12.92,((np.maximum(x,0)+0.055)/1.055)**2.4)
l2s=lambda x: np.where(x<=0.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-0.055)
orig=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB')).astype(np.float32)/255
src =np.array(Image.open('00_originals/jeune_homme.jpg').convert('RGB')).astype(np.float32)/255
# ---------------------------------------------------------------- reference analysis (faces, whites)
BOX={'woman':(285,235,485,480),'man':(610,150,850,440)}; YBOX=(498,320,692,612)
def skin_mask(rgb):
    lab=color.rgb2lab(rgb); return ((lab[...,1]>8)&(lab[...,2]>7)&(lab[...,0]>30)&(lab[...,0]<92))
def light_grid(rgb,box,n=6):
    x0,y0,x1,y1=box; c=rgb[y0:y1,x0:x1]; m=skin_mask(c).astype(np.float32)
    L=color.rgb2lab(c)[...,0]; sg=0.09*(x1-x0)
    num=cv2.GaussianBlur(L*m,(0,0),sg); den=cv2.GaussianBlur(m,(0,0),sg)
    f=num/np.maximum(den,1e-3); f=np.where(den>0.05,f,np.nan)
    f=np.nan_to_num(f,nan=np.nanmedian(f)); f=cv2.resize(f,(n*8,n*8),interpolation=cv2.INTER_AREA)
    f=cv2.resize(f,(n,n),interpolation=cv2.INTER_AREA); return f/np.nanmean(f), np.nanmean(L[m>0.5])
def lighting_analysis():
    gw,Lw=light_grid(orig,BOX['woman']); gm,Lm=light_grid(orig,BOX['man']); gy,Ly=light_grid(src,YBOX)
    T=(gw+gm)/2
    return dict(gw=gw,gm=gm,gy=gy,T=T,Lw=Lw,Lm=Lm,Ly=Ly)
# ---------------------------------------------------------------- selective denoise of the source layer
def denoise_young(F,A):
    u8=(np.clip(F,0,1)*255).astype(np.uint8)
    d_skin=cv2.fastNlMeansDenoisingColored(u8,None,1.6,1.6,5,15)
    d_cloth=cv2.fastNlMeansDenoisingColored(u8,None,4,4,7,21)
    lab=color.rgb2lab(np.clip(F,0,1)); skin=((lab[...,1]>12)&(lab[...,2]>10)&(lab[...,0]>25)).astype(np.float32)
    skin=cv2.GaussianBlur(skin,(0,0),2.0)[...,None]
    out=(d_skin.astype(np.float32)*skin+d_cloth.astype(np.float32)*(1-skin))/255
    return out
def hf_energy(rgb,mask,sigma=1.0):
    L=color.rgb2lab(np.clip(rgb,0,1))[...,0]; hp=L-cv2.GaussianBlur(L,(0,0),sigma); m=mask>0.5
    return float(np.sqrt((hp[m]**2).mean()))

# ---------------------------------------------------------------- reference colour statistics
def white_stats(rgb,box):
    x0,y0,x1,y1=box; c=rgb[y0:y1,x0:x1].reshape(-1,3); lab=color.rgb2lab(c.reshape(-1,1,3)).reshape(-1,3)
    ok=lab[:,1]**2+lab[:,2]**2<18**2; lab=lab[ok]; L=lab[:,0]; q=np.quantile(L,[0.33,0.66])
    zs=[lab[L<=q[0]],lab[(L>q[0])&(L<=q[1])],lab[L>q[1]]]
    return [z.mean(0) for z in zs], np.quantile(L,[0.1,0.5,0.9])
def zones_young_shirt(F,A):
    lab=color.rgb2lab(np.clip(F,0,1)); H,W=A.shape
    m=(A>0.99)&(lab[...,1]**2+lab[...,2]**2<14**2)&(lab[...,0]>60)
    m[:650]=False; m[1300:]=False
    return m,lab
def generate(s,D,tag,P):
    H=1431; W=1099+D
    aw=np.load('02_layers/woman_A.npy'); am=np.load('02_layers/man_A.npy'); Fw=np.load('02_layers/woman_F.npy'); Fm=np.load('02_layers/man_F.npy')
    lm=color.rgb2lab(np.clip(Fm,0,1)); wedge=(np.arange(1099)[None,:]<590)&(np.arange(H)[:,None]>520)&(np.arange(H)[:,None]<620)&((Fm[...,2]-Fm[...,0])<0.10)&(lm[...,0]<28)
    am=np.where(cv2.dilate(wedge.astype(np.uint8),np.ones((3,3),np.uint8))>0,0,am)
    Fy=np.load('V2/layers/jeune_F_raw.npy'); Ay=np.load('V2/masks/jeune_alpha.npy')
    Fy=np.clip(Fy,0,1)
    # 1. selective denoise (before any resampling)
    Fd=denoise_young(Fy,Ay)
    # 2. white balance from the shirt, toward the woman's blazer environment
    mS,lab0=zones_young_shirt(Fd,Ay)
    ref_z,ref_q=white_stats(orig,(250,1000,500,1250))
    cur=s2l(Fd[mS]).mean(0); refc=s2l(np.array([220,218,225])/255.0)
    gain=1+((refc/refc[1]*cur[1]*P['shirt_exposure'])/cur-1)*P['wb']
    Lg=np.clip(s2l(Fd)*gain,0,1.5); Fwb=l2s(np.clip(Lg,0,1))
    Lq=color.rgb2lab(np.clip(Fd,0,1))[...,0]; wwb=np.clip((Lq-22)/22,0,1)[...,None]      # very dark pixels (hair, trousers) keep their own colour
    Fw1=Fd*(1-wwb)+Fwb*wwb
    # 3. tonal-zone chroma match of the white shirt to the blazer (shadows / mids / highlights)
    lab=color.rgb2lab(np.clip(Fw1,0,1)); Ls=lab[...,0]
    mS,_=zones_young_shirt(Fw1,Ay); q=np.quantile(Ls[mS],[0.33,0.66])
    zm=[Ls[mS]<=q[0],(Ls[mS]>q[0])&(Ls[mS]<=q[1]),Ls[mS]>q[1]]
    cy=[lab[mS][z][:,1:].mean(0) for z in zm]
    wz=[np.clip((q[0]+6-Ls)/12,0,1),np.clip(1-np.abs(Ls-(q[0]+q[1])/2)/((q[1]-q[0])/2+6),0,1),np.clip((Ls-q[1]+6)/12,0,1)]
    shirt_like=cv2.GaussianBlur(((lab[...,1]**2+lab[...,2]**2<16**2)&(Ls>55)).astype(np.float32),(0,0),2)
    da=np.zeros_like(Ls); db=np.zeros_like(Ls)
    for i in range(3):
        da+=wz[i]*(ref_z[i][1]-cy[i][0]); db+=wz[i]*(ref_z[i][2]-cy[i][1])
    nz=sum(wz)+1e-3; lab[...,1]+=P['zone_chroma']*da/nz*shirt_like; lab[...,2]+=P['zone_chroma']*db/nz*shirt_like
    # highlight level of the shirt relative to the blazer (kept a little below: he stands further back)
    ys=np.quantile(lab[...,0][mS],0.9); lab[...,0]+=shirt_like*np.clip((ref_q[2]-P['white_drop']-ys),-6,6)*0.7*(Ls>60)
    # 4. skin: partial move toward the two reference skins (identity kept), lightness toward reference level
    sk=(np.clip((lab[...,1]-6)/8,0,1)*np.clip((lab[...,2]-5)/8,0,1)*np.clip((lab[...,0]-28)/10,0,1)).astype(np.float32); sk=cv2.GaussianBlur(sk,(0,0),2.5)*(Ay>0.5)
    skL=cv2.GaussianBlur(sk,(0,0),8.0)
    A_=lighting_analysis(); skw=color.rgb2lab(orig[BOX['woman'][1]:BOX['woman'][3],BOX['woman'][0]:BOX['woman'][2]]); 
    refab=[]; 
    for b in BOX.values():
        c=orig[b[1]:b[3],b[0]:b[2]]; l=color.rgb2lab(c); m=skin_mask(c); refab.append(l[m][:,1:].mean(0))
    refab=np.mean(refab,0); Lref=(A_['Lw']+A_['Lm'])/2
    yb=YBOX; ly=color.rgb2lab(Fw1[yb[1]:yb[3],yb[0]:yb[2]]); my=skin_mask(Fw1[yb[1]:yb[3],yb[0]:yb[2]]); curab=ly[my][:,1:].mean(0); curL=ly[my][:,0].mean()
    lab[...,1]+=sk*(refab[0]-curab[0])*P['skin_shift']; lab[...,2]+=sk*(refab[1]-curab[1])*P['skin_shift']; lab[...,0]+=skL*np.clip(sk*3,0,1)*(Lref-curL)*P['skin_L']
    Fc=np.clip(color.lab2rgb(np.clip(lab,[0,-100,-100],[100,100,100])),0,1)
    yy0,xx0=np.mgrid[:1632,:1179]
    hzm=((yy0<470)&~((np.abs(xx0-590)<102)&(yy0>318))).astype(np.float32)
    chr_=(Fc[...,0]-Fc[...,2]); hzm*= (chr_<0.2)
    wf=np.clip((0.95-Ay)/0.5,0,1)*hzm*np.clip((470-yy0)/40,0,1)
    Fc=Fc*(1-wf[...,None]*0.45)
    labc=color.rgb2lab(Fc); skm=cv2.GaussianBlur(((labc[...,1]>8)&(labc[...,0]>30)).astype(np.float32),(0,0),2)
    fmask=np.zeros((1632,1179),np.float32); fmask[300:640,480:710]=1; fmask=cv2.GaussianBlur(fmask,(0,0),10)*skm
    Lc=labc[...,0]; labc[...,0]=np.where(Lc>76,Lc-fmask*(Lc-76)*0.5,Lc); Fc=np.clip(color.lab2rgb(labc),0,1)
    labt=color.rgb2lab(Fc); tb=np.zeros((1632,1179),np.float32); tb[360:470,664:700]=1; tb=cv2.GaussianBlur(tb,(0,0),3)
    pale=np.clip((40-np.hypot(labt[...,1],labt[...,2]))/20,0,1)*np.clip((labt[...,0]-50)/10,0,1)*tb
    cheek=labt[430:500,640:672]; ca,cb=np.median(cheek[...,1]),np.median(cheek[...,2])
    labt[...,1]+=pale*(ca*0.9-labt[...,1]); labt[...,2]+=pale*(cb*0.9-labt[...,2]); labt[...,0]-=pale*4
    Fc=np.clip(color.lab2rgb(labt),0,1)
    # 5. relighting: face grid gain from the reference faces + soft body gradient + under-chin / neck
    gy=A_['gy']; T=A_['T']; G=np.clip((T/gy)**P['relight'],0.82,1.22)
    x0,y0,x1,y1=YBOX; Gb=cv2.resize(G.astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_CUBIC); Gb=cv2.GaussianBlur(Gb,(0,0),9)
    gmap=np.ones((1632,1179),np.float32); gmap[y0:y1,x0:x1]=Gb
    fm=np.zeros((1632,1179),np.float32); fm[y0+14:y1-6,x0+10:x1-10]=1; fm=cv2.GaussianBlur(fm,(0,0),16)
    gmap=1+(gmap-1)*fm
    yy,xx=np.mgrid[:1632,:1179].astype(np.float32); cx=590
    body=1+P['side']*np.clip((cx-xx)/300,-1,1)-P['fall']*np.clip((yy-640)/800,0,1)
    neck=1-P['neck']*np.exp(-(((xx-cx)/70)**2+((yy-650)/45)**2))
    gall=gmap*body*neck
    Ly_=color.rgb2lab(Fc)[...,0]/100
    wDodge=1-0.75*np.clip((Ly_-0.70)/0.25,0,1); wBurn=1-0.5*np.clip((0.22-Ly_)/0.22,0,1)
    ge=np.where(gall>1,1+(gall-1)*wDodge,1-(1-gall)*wBurn)
    Fr=l2s(np.clip(s2l(Fc)*ge[...,None],0,1))
    # 5b. modelling contrast of the face (low-frequency luminance) matched to the references
    labr=color.rgb2lab(Fr); Lr=labr[...,0]; lp=cv2.GaussianBlur(Lr,(0,0),12)
    def lp_std(rgb,box):
        x0_,y0_,x1_,y1_=box; c=rgb[y0_:y1_,x0_:x1_]; l=color.rgb2lab(c)[...,0]; m=skin_mask(c); l=cv2.GaussianBlur(l,(0,0),0.09*(x1_-x0_)); return l[m].std()
    sref=np.mean([lp_std(orig,bx) for bx in BOX.values()]); sy=lp_std(Fr,YBOX)
    k=float(np.clip(sref/sy,1.0,1.35)); m0=Lr[YBOX[1]:YBOX[3],YBOX[0]:YBOX[2]].mean()
    labr[...,0]=np.where(fm>0,Lr+(1-1)*0+fm*((lp-m0)*(k-1)*P['contrast']),Lr)
    Fr=np.clip(color.lab2rgb(labr),0,1); print('  modelling std ref',round(sref,2),'young',round(sy,2),'k',round(k,2))
    # 6. warp to the canvas (premultiplied, Lanczos)
    ty=P['eye_y']-s*418; tx=(560+D/2)-s*590; M=np.float32([[s,0,tx],[0,s,ty]])
    wp=lambda x: cv2.warpAffine(x.astype(np.float32),M,(W,H),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    aj=np.clip(wp(Ay),0,1); Pj=wp(Fr*Ay[...,None]); Fj=np.where(aj[...,None]>1e-3,Pj/np.maximum(aj[...,None],1e-3),0)
    faceM=wp(np.pad(np.ones((y1-y0,x1-x0),np.float32),((y0,1632-y1),(x0,1179-x1))))>0.5
    skinM=faceM&skin_mask(np.clip(Fj,0,1))&(aj>0.99)
    # 7. optical match: apparent sharpness of the face skin vs the two reference portraits
    def e_ref():
        v=[]
        for b in BOX.values():
            c=orig[b[1]:b[3],b[0]:b[2]]; v.append(hf_energy(c,skin_mask(c).astype(np.float32)*cv2.erode(np.ones(c.shape[:2],np.uint8),np.ones((15,15)))))
        return float(np.mean(v))
    tgt=e_ref()*P['sharp_target']
    def sharpen(F,amt):
        lab=color.rgb2lab(np.clip(F,0,1)); L=lab[...,0]; bl=cv2.GaussianBlur(L,(0,0),1.1); lab[...,0]=np.clip(L+(L-bl)*amt,0,100)
        return np.clip(color.lab2rgb(lab),0,1)
    lo,hi=0.0,1.2
    for _ in range(7):
        mid=(lo+hi)/2; e=hf_energy(sharpen(Fj,mid),skinM)
        if e<tgt: lo=mid
        else: hi=mid
    amt=(lo+hi)/2; Fj=sharpen(Fj,amt); print(tag,'sharpen',round(amt,2),'ref hf',round(e_ref(),3),'young hf',round(hf_energy(Fj,skinM),3))
    plate=np.load(f'V2/work/plate_mirror_D{D}.npy')/255.0
    # 8. spatial integration
    nbL=cv2.GaussianBlur(np.pad(aw,((0,0),(0,W-1099))).astype(np.float32),(0,0),P['occ_sigma']); nbR=cv2.GaussianBlur(np.pad(am,((0,0),(D,0))).astype(np.float32),(0,0),P['occ_sigma'])
    oL=np.clip(nbL-0.3,0,1)/0.7; oR=np.clip(nbR-0.3,0,1)/0.7
    lin=s2l(Fj)
    lin*=1-P['occ']*(oL+oR)[...,None]
    cream=s2l(np.array([0.90,0.88,0.86])); navy=s2l(np.array([0.10,0.13,0.28]))
    lin=lin*(1-P['bounce']*oL[...,None])+lin.mean(-1,keepdims=True)*cream*P['bounce']*oL[...,None]
    lin=lin*(1-P['bounce']*oR[...,None]*1.2)+lin.mean(-1,keepdims=True)*navy*P['bounce']*oR[...,None]*1.2*3
    Fj=np.clip(l2s(np.clip(lin,0,1)),0,1)
    edge=np.clip(aj-cv2.GaussianBlur(aj,(0,0),3.5),0,1)*2
    Fj=Fj*(1-P['wrap']*edge[...,None])+cv2.GaussianBlur(plate.astype(np.float32),(0,0),6)*P['wrap']*edge[...,None]
    Fw2=np.zeros((H,W,3),np.float32); Aw2=np.zeros((H,W),np.float32); Fm2=Fw2.copy(); Am2=Aw2.copy()
    Fw2[:,:1099]=Fw; Aw2[:,:1099]=aw; Fm2[:,D:]=Fm; Am2[:,D:]=am
    out=plate.copy()
    for F_,A_ in [(Fj,aj),(Fw2,Aw2),(Fm2,Am2)]: out=F_*A_[...,None]+out*(1-A_[...,None])
    np.savez_compressed(f'V2/variants/{tag}.npz',Fj=Fj.astype(np.float32),aj=aj,Fw=Fw2,aw=Aw2,Fm=Fm2,am=Am2,plate=plate.astype(np.float32),out=out.astype(np.float32))
    Image.fromarray((np.clip(out,0,1)*255+0.5).astype(np.uint8)).save(f'V2/variants/{tag}.png')
    return out
PARAMS=dict(shirt_exposure=1.01,wb=1.0,zone_chroma=0.9,white_drop=3.0,skin_shift=0.5,skin_L=0.6,relight=0.85,side=0.04,fall=0.07,neck=0.10,
            eye_y=292,contrast=1.0,sharp_target=1.0,occ_sigma=15,occ=0.20,bounce=0.05,wrap=0.06)
if __name__=='__main__':
    tag=sys.argv[1]; s=float(sys.argv[2]); D=int(sys.argv[3]); generate(s,D,tag,PARAMS)
