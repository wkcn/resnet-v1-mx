import caffe
import numpy as np
import cv2

MEAN_COLOR = np.array([103.0626238, 115.90288257, 123.15163084]).reshape((1, 3, 1, 1)) # BGR 

deploy = "./ResNet-101-deploy.prototxt"
caffe_model = "./ResNet-101-model.caffemodel"
nn = caffe.Net(deploy, caffe_model, caffe.TEST) 

fn = "./mobula.jpg"

img = cv2.imread(fn) # BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, ts = img.shape
if rows < cols:
    r = 224
    c = cols * 224 // rows
    o = (c - 224) // 2
    img = cv2.resize(img, (c, r))
    if o > 0:
        img = img[:, o:-o, :]
else:
    c = 224
    r = rows * 224 // cols
    o = (r - 224) // 2
    img = cv2.resize(img, (c, r))
    if o > 0:
        img = img[o:-o, :, :]

img = img.transpose((2,0,1)).reshape((1, 3, 224, 224))

img = img.astype(np.float32)

img = img - MEAN_COLOR # N,C,H,W

nn.blobs["data"].data[...] = img
nn.forward()

fc = nn.blobs["fc1000"].data[0]
ma = np.max(fc, keepdims = True)
e = np.exp(fc - ma)
p = e / np.sum(e, keepdims = True)

pred = [(i, p) for i, p in enumerate(p)]
pred.sort(key = lambda x : x[1], reverse = True)

fin = open("./inet.txt")
cls = [line.strip() for line in fin]
assert len(cls) == 1000, cls

for t in range(5):
    u = pred[t]
    i = u[0]
    print (cls[i], u[1])
