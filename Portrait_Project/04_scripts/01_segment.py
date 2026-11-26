import sys, numpy as np, cv2, onnxruntime as ort
from PIL import Image
from pymatting import estimate_alpha_cf, estimate_foreground_ml
M='/tmp/claude-0/models/'
def isnet(img, model):
    s=ort.InferenceSession(M+model, providers=['CPUExecutionProvider'])
    h,w=img.shape[:2]
    x=cv2.resize(img,(1024,1024),interpolation=cv2.INTER_AREA).astype(np.float32)
    x=x/max(x.max(),1e-6)
    x=(x-[0.485,0.456,0.406])/[1.0,1.0,1.0]
    x=x.transpose(2,0,1)[None].astype(np.float32)
    o=s.run(None,{s.get_inputs()[0].name:x})[0][0,0]
    o=(o-o.min())/(o.max()-o.min()+1e-8)
    return cv2.resize(o,(w,h),interpolation=cv2.INTER_CUBIC).clip(0,1)
def matte(path,out,model,er=6,di=10):
    img=np.array(Image.open(path).convert('RGB'))
    p=isnet(img,model)
    np.save(out+'_prob.npy',p)
    m=(p>0.5).astype(np.uint8)
    # keep largest comps
    n,l,st,_=cv2.connectedComponentsWithStats(m)
    keep=np.zeros_like(m)
    for i in range(1,n):
        if st[i,4]>3000: keep[l==i]=1
    k=lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    fg=cv2.erode(keep,k(er)); bg=1-cv2.dilate(keep,k(di))
    tri=np.full(keep.shape,0.5,np.float32); tri[fg==1]=1; tri[bg==1]=0
    a=estimate_alpha_cf(img/255.0,tri)
    a=np.clip(a,0,1)
    Image.fromarray((a*255).astype(np.uint8)).save(out+'_alpha.png')
    Image.fromarray((tri*255).astype(np.uint8)).save(out+'_trimap.png')
    print(out,'done')
matte(sys.argv[1],sys.argv[2],sys.argv[3])
