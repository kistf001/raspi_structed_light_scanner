import cv2
import numpy as np
import image_load
import setting

# 블랙 이미지 불러와서
setting.gray_width = 90

def bitlize(b,g,w):
    return (
        np.array(
            (g>(w-setting.gray_width))&(g>(b+setting.gray_width)),
            #(g>(w-setting.gray_width)),
            #(g>(b+setting.gray_width)),
            np.uint16
        )
    )

def stacking(empty,bitlized,bit_postion):
    return empty+(bitlized<<bit_postion)

if __name__ == "__main__":
    
    import cv2

    aaaa = np.zeros(setting.camera_size_mat,np.uint16)
    bbbb = np.zeros(setting.camera_size_mat,np.uint16)

    for bit_position in range(10):
        
        b = np.array(image_load.base(0,0),np.int16)
        g = np.array(image_load.gary(0,bit_position,"v"),np.int16)
        w = np.array(image_load.base(0,1),np.int16)
        a = bitlize(b,g,w)
        aaaa = stacking(aaaa,a,bit_position)
        #b = bitlize(1,gg)
        #aaaa += a
        #bbbb += b
        cv2.imshow('Resized Window', aaaa<<6)
        cv2.waitKey(250)

    #cv2.imshow('Resized Window', bbbb<<6)
    #cv2.waitKey(1100)