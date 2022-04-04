import cv2
import numpy as np
import glob
import setting

def gary(cam_select=0,bit_position=0,angle="h"):

    #if(cam_select==0):
    #    if(angle=="h"):
    #        dir = "./data/aaaa/IMG_79"+str(bit_position+45)+".JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)
    #    elif(angle=="v"):
    #        dir = "./data/aaaa/IMG_79"+str(bit_position+55)+".JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)

    #elif(cam_select==1):
    #    if(angle=="h"):
    #        dir = "./data/aaaa/IMG_79"+str(bit_position+68)+".JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)
    #    elif(angle=="v"):
    #        dir = "./data/aaaa/IMG_79"+str(bit_position+78)+".JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)

    if(cam_select==0):
        if(angle=="h"):
            dir = "./data/bbbb/IMG_"+str(bit_position+8060)+".JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
        elif(angle=="v"):
            dir = "./data/bbbb/IMG_"+str(bit_position+8070)+".JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)

    elif(cam_select==1):
        if(angle=="h"):
            dir = "./data/bbbb/IMG_"+str(bit_position+8082)+".JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
        elif(angle=="v"):
            dir = "./data/bbbb/IMG_"+str(bit_position+8092)+".JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
    
def base(cam_select=0,color=0):

    #if(cam_select==0):
    #    
    #    if(color==0):
    #        dir = "./data/aaaa/IMG_7966.JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)
    #    elif(color==1):
    #        dir = "./data/aaaa/IMG_7965.JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)
    #
    #elif(cam_select==1):

    #    if(color==0):
    #        dir = "./data/aaaa/IMG_7989.JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)
    #    elif(color==1):
    #        dir = "./data/aaaa/IMG_7988.JPG"
    #        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #        data = cv2.resize(data, dsize=setting.camera_size)
    #        return np.array(data)

    if(cam_select==0):
        
        if(color==0):
            dir = "./data/bbbb/IMG_8081.JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
        elif(color==1):
            dir = "./data/bbbb/IMG_8080.JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
    
    elif(cam_select==1):

        if(color==0):
            dir = "./data/bbbb/IMG_8103.JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)
        elif(color==1):
            dir = "./data/bbbb/IMG_8102.JPG"
            data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
            data = cv2.resize(data, dsize=setting.camera_size)
            return np.array(data)

def chassboard(cam_select=0):

    #if(cam_select==0):
    #    dir = "./data/aaaa/IMG_7944.JPG"
    #    data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #    data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
    #    return [np.array(data),np.array(data)]

    #elif(cam_select==1):
    #    dir = "./data/aaaa/IMG_7967.JPG"
    #    data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
    #    data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
    #    return [np.array(data),np.array(data)]

    if(cam_select==0):
        a = []
        dir = "./data/bbbb/IMG_8047.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8050.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8051.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8053.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8055.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        return a

    elif(cam_select==1):
        a = []
        dir = "./data/bbbb/IMG_8047.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8050.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8051.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8053.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        dir = "./data/bbbb/IMG_8055.JPG"
        data = cv2.cvtColor(cv2.imread(dir),cv2.COLOR_BGR2GRAY)
        data = cv2.resize(data, dsize=setting.camera_size, interpolation=cv2.INTER_AREA)
        a.append(data)
        return a

if __name__ == "__main__":
    
    import cv2
    cv2.imshow('Resized Window', chassboard(cam_select=0))
    cv2.waitKey(0)
    