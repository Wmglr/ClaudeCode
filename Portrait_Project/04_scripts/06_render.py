import numpy as np, json, sys
from PIL import Image
d=np.load('03_intermediate/placed.npz'); plate=np.load('03_intermediate/plate.npy')/255.0
def comp(Fj=None):
    Fj=d['Fj'] if Fj is None else Fj
    out=plate.copy()
    for F,A in [(Fj,d['aj']),(d['Fw'],d['aw']),(d['Fm'],d['am'])]:
        out=F*A[...,None]+out*(1-A[...,None])
    return out
if __name__=='__main__':
    out=comp(); Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8)).save('03_intermediate/v1_raw.png')
