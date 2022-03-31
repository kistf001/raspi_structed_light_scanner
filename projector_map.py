import numpy as np
import setting

Gray2Bin = []
Bin2Gray = []

def graycode_lookup_table( size ):

    qqqqq = 1 << (size)
    
    Gray2Bin = list(range(qqqqq))
    Bin2Gray = list(range(qqqqq))

    for binary in range(qqqqq):
        gray_code = binary ^ (binary >> 1)
        Bin2Gray[binary] = gray_code
        Gray2Bin[gray_code] = binary

    return Gray2Bin, Bin2Gray

def graycode_line(a=5):

    Gray2Bin,Bin2Gray = graycode_lookup_table(a)

    bbbb = []
    for ff in range(0,a):
        aaaa = []
        for dd in range(0,1 << (a)):
            aaaa.append(int(0!=(Bin2Gray[dd]&(1<<ff))))
        bbbb.append(aaaa)

    return bbbb

def graycode_map(sizesss = 10,hv_select="h"):
    aaaa = graycode_line(sizesss)
    dddd = np.zeros((sizesss,1024,1024),np.float32)
    if(hv_select=="h"):
        for ac in range(sizesss):
            dddd[ac].T[0:] = aaaa[ac]
    elif(hv_select=="v"):
        for ac in range(sizesss):
            dddd[ac][0:] = aaaa[ac]
    return dddd

def lamp(a=0):
    if(a==0):
        return np.zeros((1024,1024),np.float32)
    elif(a==1):
        return np.ones((1024,1024),np.float32)

Gray2Bin,Bin2Gray = graycode_lookup_table( setting.resolution_bit )
if __name__ == "__main__":
    
    import cv2

    m = graycode_map(10,"h")
    for fdfd in m:
        cv2.imshow('Resized Window', fdfd)
        cv2.waitKey(3000)

    m = graycode_map(10,"v")
    for fdfd in m:
        cv2.imshow('Resized Window', fdfd)
        cv2.waitKey(3000)
    
    cv2.imshow('Resized Window', lamp(a=1))
    cv2.waitKey(3000)

    cv2.imshow('Resized Window', lamp(a=0))
    cv2.waitKey(3000)
    