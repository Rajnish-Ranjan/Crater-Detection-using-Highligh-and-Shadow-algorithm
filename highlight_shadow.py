import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
import csv

""" To find parent in disjoint set
"""
def find(mask,k):
    l=k
    while(mask[l]!=l):
        l=mask[l]
    mask[k]=l
    return l

def findYDist(ya_min,ya_max,yb_min,yb_max):
    yl=0
    if(ya_max<=yb_min):
        yl=yb_min-ya_max
    else:
        if(yb_max<=ya_min):
            yl=ya_min-yb_max
    
    return yl

def findXDist(xa_min,xa_max,xb_min,xb_max):
    xl=0
    if(xa_max<=xb_min):
        xl=xb_min-xa_max
    else:
        if(xb_max<=xa_min):
            xl=xa_min-xb_max

    return xl

def findBoxDist(xa_min,xa_max,ya_min,ya_max,xb_min,xb_max,yb_min,yb_max):
    xl=findXDist(xa_min,xa_max,xb_min,xb_max)
    yl=findYDist(ya_min,ya_max,yb_min,yb_max)
    ds=pow(pow(xl,2)+pow(yl,2),0.5)
    return ds


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


def union(keep1,keep2,i,j):
    res=np.zeros([8],dtype=np.long)
    res[0]=keep1[i][0]+keep2[j][0]
    res[1]=keep1[i][1]+keep2[j][1]
    res[1]=res[1]/2
    res[2]=keep1[i][2]+keep2[j][2]
    res[2]=res[2]/2
    res[3]=min(keep1[i][3],keep2[j][3])
    res[4]=min(keep1[i][4],keep2[j][4])
    res[5]=max(keep1[i][5],keep2[j][5])
    res[6]=max(keep1[i][6],keep2[j][6])
    return res
    

def unionSame(keep,i,j):
    res=np.zeros([8],dtype=np.long)
    res[0]=keep[i][0]+keep[j][0]
    res[1]=keep[i][1]+keep[j][1]
    res[1]=res[1]/2
    res[2]=keep[i][2]+keep[j][2]
    res[2]=res[2]/2
    res[3]=min(keep[i][3],keep[j][3])
    res[4]=min(keep[i][4],keep[j][4])
    res[5]=max(keep[i][5],keep[j][5])
    res[6]=max(keep[i][6],keep[j][6])
    return res
    

"""
This function returns label for each of the components
Arguments
Iv - original image
I - Positive image or Negative image(255-I)
r - figure number for plot of image inside the function using subplot

"""
def connectedComp(Iv,I,r):
    [rows,cols]=np.shape(I)
    mn=np.amin(I)
    mx=np.amax(I)
    
    """ Converting I from float array to uint array having values between 0 and 255
    """
    I=I.astype(np.float32)
    I=np.add(I,-mn)
    I=np.divide(I,mx-mn)  # I - min(I)/(max(I) - min[I])
    I = 255 * I
    I = I.astype(np.uint8)
    f1 = plt.figure(r)
    plt.imshow(I,cmap='gray')

    """Calculating threshold, 90% of the pixels are to be blackened
    """
    th=np.average(I)
    hist = cv2.calcHist([I],[0],None,[256],[0,256])
    hist=np.divide(hist,rows*cols)
    hist=np.multiply(hist,100)
    sm=0
    print(np.shape(hist),'\n')
    for i in range(0,256):    # is it same for postive and negative image
        sm=sm+hist[i]
        if(sm>89.9):
            th=i
            break
    
    #print(th,'\n')
    th=np.float(th)

    """ Thresholding I
    """
    # what is ret,
    # It: Binary image
    ret,It = cv2.threshold(I,th,255,cv2.THRESH_BINARY)  # show this image

    """ Morphological opening to the thresholding binary image
    """
    kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    It = cv2.morphologyEx(It, cv2.MORPH_OPEN, kernel)  # show this image

    """ Iniatializing labels on finding horizontally connected pixels
    """
    l=0
    label=np.zeros([rows,cols],dtype=np.uint32)   
    for i in range(0,rows):
        if(It[i][0]):
            l=l+1
            label[i][0]=l
        for j in range(1,cols):
            if(It[i][j]):
                if(It[i][j-1]):
                    label[i][j]=l
                else:
                    l=l+1
                    label[i][j]=l

    """ We will be using disjoint set to find connected horizontal lines
        Parent will contain
        size <- mask[1,:], x-sum <- mask[2,:], y-sum <- mask[3,:],
        x-min <- mask[4,:], y-min <- mask[5,:], x-max <- mask[6,:], y-max <- mask[7,:]
    """ 
    mask=np.zeros([2,l+1],dtype=np.ulonglong)
    # mask contains parent, size, x-sum, y-sum , x-min, y-min, x-max, y-max
    
    for i in range(0,l+1):
        mask[0][i]=i
        mask[1][i]=0
        
    """ Initializing and storing size of each labelled part
    """
    for i in range(0,rows):
        for j in range(0,cols):
            k=label[i][j]
            mask[1][k]=mask[1][k]+1

    """Connecting (horizontal) parts vertically
    """
    for i in range(0,rows-1):
        for j in range(0,cols-1):
            if(label[i][j]):
                k=label[i][j]
                q=find(mask[0],k)
                if(label[i+1][j]):
                    z=label[i+1][j]
                    mask[0][z]=q
                    mask[1][q]=mask[1][z]+mask[1][q]
                if(label[i][j+1]):
                    z=label[i][j+1]
                    mask[0][z]=q
                    mask[1][q]=mask[1][z]+mask[1][q]
                if(label[i+1][j+1]):
                    z=label[i+1][j+1]
                    mask[0][z]=q
                    mask[1][q]=mask[1][z]+mask[1][q]
                
    
    for i in range(0,rows):
        for j in range(0,cols):
            k=label[i][j]
            mask[0][k]=find(mask[0],k)
            label[i][j]=mask[0][k]

    """ Repeat connecting(Iteration-2) to avoid mistakes
    """
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if(label[i][j]):
                k=label[i][j]
                q=find(mask[0],k)
                if(label[i-1][j]):
                    z=label[i-1][j]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i][j-1]):
                    z=label[i][j-1]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i+1][j]):
                    z=label[i+1][j]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i][j+1]):
                    z=label[i][j+1]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i-1][j-1]):
                    z=label[i-1][j-1]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i-1][j+1]):
                    z=label[i-1][j+1]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                if(label[i+1][j+1]):
                    z=label[i+1][j+1]
                    if(find(mask[0],z)!=q):
                        mask[0][q]=z
                        mask[1][z]=mask[1][q]+mask[1][z]
                        
                
    for i in range(0,rows):
        for j in range(0,cols):
            k=label[i][j]
            label[i][j]=find(mask[0],k)

    """ Re-assigning consecutive number to labels
    """
    lst=np.zeros([0],dtype=np.uint16)
    v=0
    lst=np.append(lst,0)
    for i in range(0,rows):
        for j in range(0,cols):
            if(label[i][j]):
                h=label[i][j]
                sz=mask[1][h]
                d=np.where(lst==h)
                if(sz<4):
                    label[i][j]=0
                    continue
                if(np.size(d)):
                    k=d[0][0]
                    label[i][j]=k
                else:
                    v=v+1
                    lst=np.append(lst,h)
                    label[i][j]=v
    v=v+1
    
    print('\nNo. of components = ',v)
    lab=label

    
    """The code to plot the connected regions
    """
    
    mx=np.amax(label)
    mn=np.amin(label)
    label=np.float32(label)/np.float32(mx)
    label=np.multiply(label,255)
    l1=np.mod(np.multiply(label,14),255)
    l2=np.mod(np.multiply(label,29),255)
    l3=np.mod(np.multiply(label,10),255)
    l1=l1.astype(np.uint8)
    l2=l2.astype(np.uint8)
    l3=l3.astype(np.uint8)
    
    Im=np.zeros([rows,cols,3],dtype=np.uint8)
    Im[:,:,0]=l1
    Im[:,:,1]=l2
    Im[:,:,2]=l3
    for i in range(0,rows):
        for j in range(0,cols):
            if(Im[i][j].all()==0):
                Im[i][j]=Iv[i][j]
    f1 = plt.figure(r+1)
    plt.imshow(Im)    
    
    # matrix show    
    
    return (lab,v)

def find1(high,i,j):
    if(high[i][7]==i):
        return i
    while(high[i][7]!=i):
        i=high[i][7]
    return i

def combine(keep,n):
    high=[]
    m=n
    for i in range(n):
        res=np.zeros([8],dtype=np.long)
        for j in range(0,7):
            res[j]=keep[i][j]
        res[7]=np.long(i)
        high.append(res)
    #res=np.zeros([8],dtype=np.long)

    print('check ', n)

    
    return high



"""
################################################ Code flow starts from here ################################################################
"""

# code starts here


# Loading the input ortho image
Im=cv2.imread('test_ortho.tif')

[rows,cols,ch]=np.shape(Im)

I=Im[:,:,0]

#I = cv2.equalizeHist(I)


O=np.ones([rows,cols],dtype=np.uint8)
O=np.multiply(255,O)

# Negating image I to N

N=O-I

# The large features are removed using median filters for I and N respectively


MI = cv2.medianBlur(I, 101)
I=np.subtract(np.float32(I),np.float32(MI))

MN = cv2.medianBlur(N, 101)
N=np.subtract(np.float32(N),np.float32(MN))

# Im and 1 used for plotting purpose
# INPUT:==
#
# OUTPUTS:==
#  label1 : labeled mask
#  n1 : no. of connected componant in postive/Negative image (heighlights)  
(label1,n1)=connectedComp(Im,I,1) # For positive image 
(label2,n2)=connectedComp(Im,N,3) # For negative image


# Initializing keep1 for highlight regions and
#                keep2 for shadow regions

# keep (keep1/keep2) contains size <- keep[:,0], x-centre <- keep[:,1], y-centre <- keep[:,2],
# x-min <- keep[:,3], y-min <- keep[:,4], x-max <- keep[:,5], y-max <- keep[:,6], bbox-size <- keep[:,7]

# To find bounding box, boundary values are obtained
keep1=np.zeros([n1,9],dtype=np.long)
keep2=np.zeros([n2,9],dtype=np.long)


# Inttialization 
keep1[:,3]=rows
keep1[:,4]=cols

keep2[:,3]=rows
keep2[:,4]=cols

#
# 
for i in range(0,rows):
    for j in range(0,cols):
        k=label1[i][j]
        if(not(k)):
            continue
        keep1[k][0]=keep1[k][0]+1     # no. of componant in each connected componant of an image 
        keep1[k][1]=keep1[k][1]+i    # center x
        keep1[k][2]=keep1[k][2]+j    # center y
        keep1[k][3]=min(i,keep1[k][3])
        keep1[k][4]=min(j,keep1[k][4])
        keep1[k][5]=max(i,keep1[k][5])
        keep1[k][6]=max(j,keep1[k][6])


        
for i in range(n1):
    if(keep1[i][0]):
        keep1[i][1]=keep1[i][1]/keep1[i][0]
        keep1[i][2]=keep1[i][2]/keep1[i][0]
    xl=keep1[i][5]-keep1[i][3]+1
    yl=keep1[i][6]-keep1[i][4]+1
    keep1[i][7]=xl*yl
"""
for i in range(0,n1):
    print(keep1[i][0],'',keep1[i][7])
"""

for i in range(0,rows):
    for j in range(0,cols):
        k=label2[i][j]
        if(not(k)):
            continue
        keep2[k][0]=keep2[k][0]+1
        keep2[k][1]=keep2[k][1]+i
        keep2[k][2]=keep2[k][2]+j
        keep2[k][3]=min(i,keep2[k][3])
        keep2[k][4]=min(j,keep2[k][4])
        keep2[k][5]=max(i,keep2[k][5])
        keep2[k][6]=max(j,keep2[k][6])
        
for i in range(0,n2):
    if(keep2[i][0]):
        keep2[i][1]=keep2[i][1]/keep2[i][0]
        keep2[i][2]=keep2[i][2]/keep2[i][0]
    xl=keep2[i][5]-keep2[i][3]+1   #
    yl=keep2[i][6]-keep2[i][4]+1
    keep2[i][7]=xl*yl
    #if(keep2[i][0]>keep2[i][7]):
    #    print(keep2[i][0],' ',keep2[i][7],'\n')

high=combine(keep1,n1)
shad=combine(keep2,n2)
[n1,z]=np.shape(high)
[n2,z]=np.shape(shad)

box=[]
# Use threshold, selecting the desired bounding box
for i in range(1,n1):    # no. of connected componant in highlight
    for j in range(1,n2):
        xl=abs(high[i][1]-shad[j][1])  # x1 = abs(x_center_highlight -  x_center_shadow)
        yl=abs(high[i][2]-shad[j][2])
        distt=pow((pow(xl,2)+pow(yl,2)),0.5)  # distance(bi,bj)
        #finding maximum area
        sz1=max(high[i][0],shad[j][0])
        sz2=max(shad[j][7],high[i][7]) # fix
        sz=max(sz1,sz2)
        sz=np.sqrt(sz)
        if(distt<2*sz):
            temp=union(high,shad,i,j)
            box.append(temp)  # i: highlight index, j: shadow index
            #merge shadow and highlight





#Plotting bounding boxes and writing them as subsamples(hypothesis)
[n,z]=np.shape(box)
for i in range(0,n):
    box[i][7]=i
for i in range(n):
    for j in range(n):
        iou=findIOU(box[i][3],box[i][5],box[i][4],box[i][6],box[j][3],box[j][5],box[j][4],box[j][6])
        if(iou>0.7):
            box[j][7]=i
            temp=unionSame(box,i,j)
            box[i]=temp
            box[i][7]=i
        
bbox=[]
for i in range(n):
    if(box[i][7]==i):
        bbox.append(box[i])

[cnt,z]=np.shape(bbox)

dim=(32,32)
print('cnt ',cnt)

plt.figure()
plt.imshow(Im)
fields = ['x', 'y', 'width', 'height']

data=[]

for i in range(0,cnt):
    xn=bbox[i][3]   #xmin
    xx=bbox[i][5]
    yn=bbox[i][4]
    yx=bbox[i][6]
    data.append([xn,yn,xx-xn+1,yx-yn+1])
    """
    img=Im[xn:xx,yn:yx]
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # fix
    cv2.imwrite('samples/im'+str(i)+'.tif',img)
    """
    plt.plot([yn,yx],[xn,xn],'c')
    plt.plot([yx,yx],[xn,xx],'c')
    plt.plot([yx,yn],[xx,xx],'c')
    plt.plot([yn,yn],[xx,xn],'c')

print(np.shape(data))
"""
header = ['x', 'y', 'w', 'l']
with open('craters.csv', 'wt',newline='') as f:
    csv_writer = csv.writer(f)
 
    csv_writer.writerow(header) # write header
 
    for row in data:
        csv_writer.writerow(row)

"""  
np.savetxt('test.csv', data,header = 'x,y,w,h', delimiter=',')

plt.show()
