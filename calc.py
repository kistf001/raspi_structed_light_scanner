import cv2
import numpy as np
import math

def triangulate(pxpy,Rotation,transformation):

    ####################################################
    # # function for calculating X,Y, and Z of points   
    ####################################################

    # focal lenght
    ccddr,ccddl=1,1         # focal lenght (right,left) camera mm

    # left 카메라와 비교해서 right 카메라는 얼마나 이동되어져 잇는지
    x0r=transformation[0]                   # left camera translation in X direction
    y0r=transformation[1]                   # right camera translation in Y direction
    z0r=transformation[2] 

    # left 카메라와 비교해서 right 카메라는 얼마나 각이 틀어져 잇는지
    Rotation

    ###################################################
    # right camera
    ###################################################
    x0,y0,z0=x0r,y0r,z0r

    x1 = -pxpy[0]   #px1
    y1 =  pxpy[1]   #py1
    z1 =  ccddr

    XA=np.array(Rotation.dot(([[x1],[y1],[z1]]))+([[x0],[ y0],[ z0]]))

    x1,y1,z1=XA[0],XA[1],XA[2] 

    ###################################################
    # left camera
    ###################################################
    x2,y2,z2=0,0,0

    xh3 = -pxpy[2]   #px2
    yh3 =  pxpy[3]   #py2 
    zh3 =  ccddl 

    XB=np.array([[xh3],[yh3],[zh3]])

    x3,y3,z3=XB[0],XB[1],XB[2]

    ###################################################
    #
    ###################################################
    u=[float(x1)-x0,float(y1)-y0,float(z1)-z0]
    v=[float(x3-x2),float(y3-y2),float(z3-z2)]
    p=[x0,y0,z0]
    q=[x2,y2,z2]
    w=np.subtract(p,q)
    denomst=np.inner(v,u)*np.inner(v,u)-np.inner(v,v)*np.inner(u,u)
    s=(np.inner(w,u)*np.inner(v,v)-np.inner(v,u)*np.inner(w,v))/denomst
    t=(np.inner(v,u)*np.inner(w,u)-np.inner(u,u)*np.inner(w,v))/denomst
    xyz=np.divide((np.add(p,np.multiply(s,u))+np.add(q,np.multiply(t,v))),2)
    abdist=np.add(p,np.multiply(s,u))-np.add(q,np.multiply(t,v))

    return xyz,abdist