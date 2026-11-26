import numpy as np, cv2, json, sys
from PIL import Image
G=json.load(open('04_scripts/geometry.json'))
D=G['D']; S=G['scale']; W=1099+D; H=1431
L=lambda n:(np.load(f'02_layers/{n}_F.npy').astype(np.float64),np.load(f'02_layers/{n}_A.npy').astype(np.float64))
Fw,aw=L('woman'); Fm,am=L('man'); Fj,aj=L('young')
def place_orig(F,A,dx):
    Fo=np.zeros((H,W,3)); Ao=np.zeros((H,W))
    Fo[:,dx:dx+1099]=F; Ao[:,dx:dx+1099]=A
    return Fo,Ao
Fw2,aw2=place_orig(Fw,aw,0); Fm2,am2=place_orig(Fm,am,D)
# young man: affine scale+translate
tx=G['axis_dst_x']-S*G['axis_src_x']; ty=G['eye_dst_y']-S*G['eye_src_y']
M=np.float32([[S,0,tx],[0,S,ty]])
def warp(x,interp=cv2.INTER_AREA if S<1 else cv2.INTER_CUBIC):
    return cv2.warpAffine(x,M,(W,H),flags=cv2.INTER_AREA if False else cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
# premultiply for correct resampling
Pj=Fj*aj[...,None]
Pj2=warp(Pj.astype(np.float32)).astype(np.float64); aj2=np.clip(warp(aj.astype(np.float32)).astype(np.float64),0,1)
Fj2=np.where(aj2[...,None]>1e-4,Pj2/np.maximum(aj2[...,None],1e-4),0)
np.savez('03_intermediate/placed.npz',Fw=Fw2,aw=aw2,Fm=Fm2,am=am2,Fj=Fj2,aj=aj2)
union=1-(1-aw2)*(1-am2)*(1-aj2)
Image.fromarray((union*255).astype(np.uint8)).save('03_intermediate/union.png')
print(W,H,tx,ty)
