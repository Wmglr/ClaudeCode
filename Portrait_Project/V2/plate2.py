"""Backdrop extension v2: planks are flattened (perspective seams -> horizontal), the missing wall is grown with
translation-only patch quilting (min-error seam cut, no mirroring), then re-curved onto the original seam geometry."""
import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as ndi
img=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB')).astype(np.float32)
a0=np.array(Image.open('01_masks/duo_alpha_v2.png')).astype(np.float32)/255
H=1431
person=cv2.dilate((a0>0.01).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
seams=json.load(open('03_intermediate/seams.json'))
sy=lambda x:np.array([b+k*x for b,k in seams]); Yref=sy(540.0)
def _knots(s): return (np.concatenate([[s[0]-1000],s,[s[-1]+3000]]),np.concatenate([[Yref[0]-1000],Yref,[Yref[-1]+3000]]))
def flatten(arr):
    xs=np.arange(arr.shape[1]); yp=np.arange(H,dtype=np.float32)
    my=np.zeros((H,len(xs)),np.float32); mx=np.zeros_like(my)
    for j,x in enumerate(xs):
        ys,Yk=_knots(sy(x)); my[:,j]=np.interp(yp,Yk,ys); mx[:,j]=x
    return cv2.remap(arr,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
def unflatten(arrf,Wd):
    yy=np.arange(H,dtype=np.float32); my=np.zeros((H,Wd),np.float32); mx=np.zeros_like(my)
    for j in range(Wd):
        ys,Yk=_knots(sy(j)); my[:,j]=np.interp(yy,ys,Yk); mx[:,j]=j
    return cv2.remap(arrf,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
Ff=flatten(img); Pf=flatten(person.astype(np.float32))>0.02
def dp_cut(err):
    """err: rows x k. returns boolean mask True where we take the NEW patch (right of the min path)"""
    R,K=err.shape; cost=err.copy(); 
    for y in range(1,R):
        p=cost[y-1]; m=np.minimum(p,np.minimum(np.r_[np.inf,p[:-1]],np.r_[p[1:],np.inf])); cost[y]+=m
    path=np.zeros(R,int); path[-1]=int(np.argmin(cost[-1]))
    for y in range(R-2,-1,-1):
        c=path[y+1]; lo=max(c-1,0); hi=min(c+2,K); path[y]=lo+int(np.argmin(cost[y,lo:hi]))
    cols=np.arange(K)[None,:]; return cols>path[:,None]
def blend_lr(Aold,Bnew,ov):
    """Aold, Bnew: rows x ov x 3 ; returns merged using a min-error cut + 2px feather"""
    t=np.linspace(0,1,Aold.shape[1])[None,:,None]; m=(0.5-0.5*np.cos(np.pi*t)).astype(np.float32)   # cosine ramp over the whole overlap
    return Aold*(1-m)+Bnew*m
def quilt_band(r0,r1,xL,xR_canvas,D,W,pool,rng,pw=150,ov=64):
    pool=list(pool)
    rows=slice(r0,r1); Q=np.zeros((r1-r0,W,3),np.float32)
    Q[:,:xL]=Ff[rows,:xL]; xRo=xR_canvas-D
    Q[:,xR_canvas:]=Ff[rows,xRo:1099]
    cands=np.array(pool); x=xL; used=[]; ssd=lambda a,b:((a-b)**2).mean()
    n=int(np.ceil((xR_canvas-xL)/(pw-ov))); stride=(xR_canvas-xL+ov)/max(n,1)
    for i in range(n):
        x0=int(round(xL-ov+i*(stride-ov*0)))   # patch left edge (overlaps what is already placed)
        x0=int(round(xL-ov+i*(pw-ov)*((xR_canvas-xL+ov)/(n*(pw-ov)))))
        left=Q[:,x0:x0+ov]; last=(i==n-1); right=Q[:,x0+pw-ov:x0+pw] if last else None
        sc=[]
        for sx in cands:
            P=Ff[rows,sx:sx+pw]; e=ssd(P[:,:ov],left)
            if last: e+=ssd(P[:,-ov:],right)
            sc.append(e)
        sc=np.array(sc)
        for u in used: sc=np.where(np.abs(cands-u)<pw*0.5,sc+1e6,sc)
        order=np.argsort(sc); best=cands[order[:6]]; sx=int(rng.choice(best)); used.append(sx)
        if len(used)>3: used.pop(0)
        P=Ff[rows,sx:sx+pw].copy()
        Q[:,x0:x0+ov]=blend_lr(left,P[:,:ov],ov)
        if last:
            Q[:,x0+ov:x0+pw-ov]=P[:,ov:pw-ov]; Q[:,x0+pw-ov:x0+pw]=blend_lr(P[:,-ov:],right,ov)[:, :]  # patch -> existing right content
            # ensure orientation: patch on the left, right content on the right
            Q[:,x0+pw-ov:x0+pw]=blend_lr(P[:,-ov:],right,ov)
        else: Q[:,x0+ov:x0+pw]=P[:,ov:]
    return Q
def band_fill(r0,r1,cLb,cRb,D,W,rng,pws):
    clean=~Pf[r0:r1].any(0)
    for pw in pws:
        pool=[sx for sx in range(0,1099-pw,3) if clean[sx:sx+pw].all()]
        if len(pool)>=12: break
    ov=max(24,int(pw*0.45))
    print('  band',(r0,r1),'pw',pw,'pool',len(pool))
    return quilt_band(r0,r1,cLb,cRb+D,D,W,pool,rng,pw=pw,ov=ov)
def lims(r0,r1,dflt=(575,855)):
    cl=Pf[r0:r1]; has=cl.any(1)
    if not has.any(): return dflt
    cL=min(int(np.argmax(r)) for r in cl[has]); cR=max(1099-int(np.argmax(r[::-1])) for r in cl[has])
    return min(cL,575)-6, min(max(cR,855)+6,1099-100)
def build(D,seed=5):
    W=1099+D; rng=np.random.default_rng(seed)
    Y=[0]+[int(round(v)) for v in Yref]
    PF=np.zeros((H,W,3),np.float32)
    jobs=[(0,Y[1],(575,855),[150,110]),
          (Y[1],150,(575,855),[150,110]),            # upper plank, rows above the hair line: wide reservoir
          (150,Y[2],None,[120,90,64,48]),
          (Y[2],Y[3],None,[120,90,64,48]),
          (Y[3],Y[4]+3,None,[100,80,64,48])]
    for (r0,r1,lm,pws) in jobs:
        cl,cr=lm if lm else lims(r0,r1)
        if r0==Y[1]: cl,cr=lims(Y[1],150,(575,855)); cl,cr=min(cl,575),max(cr,855)
        PF[r0:r1]=band_fill(r0,r1,cl,cr,D,W,rng,pws)
    # vertical feather where the upper plank changes reservoir (rows 150): inside a plank, grain is horizontal
    # (quilted independently => blend 20 rows)
    PF[Y[4]+3:,:1099]=Ff[Y[4]+3:]
    return unflatten(PF,W)
if __name__=='__main__':
    import sys; D=int(sys.argv[1]); p=build(D); np.save(f'V2/work/plate_D{D}_top.npy',p)
    Image.fromarray(np.clip(p,0,255).astype(np.uint8)).save(f'V2/work/plate_D{D}_top.png')
