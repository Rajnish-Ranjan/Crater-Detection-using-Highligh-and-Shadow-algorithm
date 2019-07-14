import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
import csv

def findIOU(xa_min,xa_max,ya_min,ya_max,xb_min,xb_max,yb_min,yb_max):
    xA=max(xa_min,xb_min)
    xB=min(xa_max,xb_max)
    yA=max(ya_min,yb_min)
    yB=min(ya_max,yb_max)
    iA=max(0,xB-xA+1)*max(0,yB-yA+1)

    bA=(xa_max-xa_min+1)*(ya_max-ya_min+1)
    bB=(xb_max-xb_min+1)*(yb_max-yb_min+1)

    iou=iA/float(bA+bB-iA)
    return iou


Im=cv2.imread('tile3_25.pgm')
[rows,cols,z]=np.shape(Im)
res=np.array([])

with open('test1.csv', 'rt') as f:
    csv_reader = csv.reader(f)

    for line in csv_reader:
        res=np.append(res,line)
        

gt=np.array([])

with open('ground_truth.csv', 'rt') as f:
    csv_reader = csv.reader(f)

    for line in csv_reader:
        #print(line)
        gt=np.append(gt,line)

n_r=np.size(res)
n_g=np.size(gt)

plt.figure()
plt.imshow(Im)

gg=[]
dim=(32,32)
for i in range(6,n_g-5,6):
    x_c=float(gt[i+3])/8.32
    y_c=float(gt[i+4])/8.33
    r=float(gt[i+5])/8.325
    rt=np.sqrt(2)
    yn=np.subtract(x_c,int(rt*float(r)))
    yx=x_c+int(rt*float(r))
    xn=y_c-int(rt*float(r))
    xx=y_c+int(rt*float(r))
    yn=int(yn)
    xn=int(xn)
    yx=int(yx)
    xx=int(xx)
    if(xn<0):
        xn=0
    if(yn<0):
        yn=0
    if(xx>=rows):
        xn=rows-1
    if(yx>=cols):
        yx=cols-1
    gg.append(xn)
    gg.append(yn)
    gg.append(xx)
    gg.append(yx)
    plt.plot([yn,yx],[xn,xn],'c')
    plt.plot([yx,yx],[xn,xx],'c')
    plt.plot([yx,yn],[xx,xx],'c')
    plt.plot([yn,yn],[xx,xn],'c')
    """
    v=int(i/6)
    img=Im[xn:xx,yn:yx]
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # fix
    cv2.imwrite('craters/im'+str(v)+'.tif',img)
    """


"""
[n_r,z]=np.shape(res)

[n_g,z]=np.shape(gt)
"""

pos=n_r/4
tp=0
fp=0
tn=0
fn=0

n_g=np.size(gg)
tru=n_g/4
for i in range(4,n_r-3,4):
    mx=0
    k=0
    for j in range(4,n_g-3,4):
        xa_min=float(res[i])
        ya_min=float(res[i+1])
        xa_max=float(res[i])+float(res[i+2])
        ya_max=float(res[i+1])+float(res[i+3])
        xb_min=int(gg[j])
        yb_min=int(gg[j+1])
        xb_max=int(gg[j+2])
        yb_max=int(gg[j+3])
        iou=findIOU(xa_min,xa_max,ya_min,ya_max,xb_min,xb_max,yb_min,yb_max)
        if(iou>mx):
            mx=iou
            k=i
    
    if(mx>=0.2):
        tp=tp+1

print('precision% = ',float(tp*100)/float(pos))
tp=0
iu=0.0
for j in range(4,n_g-3,4):
    mx=0
    k=0
    for i in range(4,n_r-3,4):
        xa_min=float(res[i])
        ya_min=float(res[i+1])
        xa_max=float(res[i])+float(res[i+2])
        ya_max=float(res[i+1])+float(res[i+3])
        xb_min=int(gg[j])
        yb_min=int(gg[j+1])
        xb_max=int(gg[j+2])
        yb_max=int(gg[j+3])
        iou=findIOU(xa_min,xa_max,ya_min,ya_max,xb_min,xb_max,yb_min,yb_max)
        if(iou>mx):
            mx=iou
            k=i
    
    if(mx>=0.1):
        tp=tp+1
    iu=iu+mx
print('iou = ',float(iu)/float(tru))

print(tru,' ',tp)

print('Recall% = ',float(tp*100)/float(tru))

plt.show()

