import numpy as np, cv2
def guided(I,p,r,eps):
    """Colour guided filter (He et al.), I: HxWx3 float [0,1], p: HxW"""
    k=(2*r+1,2*r+1); bf=lambda x: cv2.boxFilter(x,-1,k,normalize=True,borderType=cv2.BORDER_REPLICATE)
    I=I.astype(np.float32); p=p.astype(np.float32)
    mI=np.stack([bf(I[...,c]) for c in range(3)],-1); mp=bf(p)
    cIp=np.stack([bf(I[...,c]*p)-mI[...,c]*mp for c in range(3)],-1)
    var=np.zeros(I.shape[:2]+(3,3),np.float32)
    for i in range(3):
        for j in range(i,3):
            v=bf(I[...,i]*I[...,j])-mI[...,i]*mI[...,j]; var[...,i,j]=v; var[...,j,i]=v
    var+=eps*np.eye(3,dtype=np.float32)
    a=np.linalg.solve(var,cIp[...,None])[...,0]
    b=mp-(a*mI).sum(-1)
    ma=np.stack([bf(a[...,c]) for c in range(3)],-1); mb=bf(b)
    return np.clip((ma*I).sum(-1)+mb,0,1)
