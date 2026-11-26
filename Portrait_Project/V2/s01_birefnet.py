import numpy as np, cv2, onnxruntime as ort, sys
from PIL import Image
M='/tmp/claude-0/models/BiRefNet-portrait-epoch_150.onnx'
s=ort.InferenceSession(M,providers=['CPUExecutionProvider'],sess_options=__import__('onnxruntime').SessionOptions())
for name,src in [('duo','duo_reference')]:
    img=np.array(Image.open(f'00_originals/{src}.jpg').convert('RGB')); h,w=img.shape[:2]
    x=cv2.resize(img,(1024,1024),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    x=((x-[0.485,0.456,0.406])/[0.229,0.224,0.225]).transpose(2,0,1)[None].astype(np.float32)
    o=s.run(None,{s.get_inputs()[0].name:x})[-1][0,0]
    p=1/(1+np.exp(-o)); p=cv2.resize(p,(w,h),interpolation=cv2.INTER_CUBIC).clip(0,1)
    np.save(f'V2/masks/{name}_birefnet.npy',p.astype(np.float32))
    print(name,p.mean(),(p>0.5).mean())
