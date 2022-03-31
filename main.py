import cv2
import numpy as np

import image_processing
import image_load
import setting
import projector_map


if __name__ == "__main__":

    if(0):
        
        #
        aaaa_H = np.zeros(setting.camera_size_mat,np.uint16)
        aaaa_V = np.zeros(setting.camera_size_mat,np.uint16)
        bbbb_H = np.zeros(setting.camera_size_mat,np.uint16)
        bbbb_V = np.zeros(setting.camera_size_mat,np.uint16)

        # structed 이미지를 single graycode 이미지로 쌓아올림
        for bit_position in range(10):
            
            #
            w = np.array(image_load.base(0,1),np.int16)
            b = np.array(image_load.base(0,0),np.int16)

            g = np.array(image_load.gary(0,bit_position,"h"),np.int16)
            a = image_processing.bitlize(b,g,w)
            aaaa_H = image_processing.stacking(aaaa_H,a,bit_position)
            
            g = np.array(image_load.gary(0,bit_position,"v"),np.int16)
            a = image_processing.bitlize(b,g,w)
            aaaa_V = image_processing.stacking(aaaa_V,a,bit_position)
            
            #
            w = np.array(image_load.base(1,1),np.int16)
            b = np.array(image_load.base(1,0),np.int16)
            
            g = np.array(image_load.gary(1,bit_position,"h"),np.int16)
            a = image_processing.bitlize(b,g,w)
            bbbb_H = image_processing.stacking(bbbb_H,a,bit_position)

            g = np.array(image_load.gary(1,bit_position,"v"),np.int16)
            a = image_processing.bitlize(b,g,w)
            bbbb_V = image_processing.stacking(bbbb_V,a,bit_position)

            cv2.imshow('Resized Window0', aaaa_H<<6)
            cv2.imshow('Resized Window1', aaaa_V<<6)
            cv2.imshow('Resized Window2', bbbb_H<<6)
            cv2.imshow('Resized Window3', bbbb_V<<6)
            
            cv2.waitKey(250)

            #cv2.imshow('Resized Window', bbbb<<6)
            #cv2.waitKey(1100)

            print(bit_position)
        np.save('./np_list_data/aaaa_H',   aaaa_H)
        np.save('./np_list_data/aaaa_V',   aaaa_V)
        np.save('./np_list_data/bbbb_H',   bbbb_H)
        np.save('./np_list_data/bbbb_V',   bbbb_V)

    if(1):

        #
        aaaa_H = np.load('./np_list_data/aaaa_H.npy')
        aaaa_V = np.load('./np_list_data/aaaa_V.npy')
        bbbb_H = np.load('./np_list_data/bbbb_H.npy')
        bbbb_V = np.load('./np_list_data/bbbb_V.npy')
        nnnn_left = np.zeros([1024,1024,2],np.uint16)
        nnnn_right = np.zeros([1024,1024,2],np.uint16)
        
        # single graycode 이미지를 coordinate image로 전환함
        for y in range(setting.camera_size_mat[0]):
            for x in range(setting.camera_size_mat[1]):
                a1,a2 = projector_map.Gray2Bin[aaaa_H[y][x]],projector_map.Gray2Bin[aaaa_V[y][x]]
                b1,b2 = projector_map.Gray2Bin[bbbb_H[y][x]],projector_map.Gray2Bin[bbbb_V[y][x]]
                #a1,a2 = aaaa_H[y][x],aaaa_V[y][x]
                #b1,b2 = bbbb_H[y][x],bbbb_V[y][x]
                nnnn_left[ a1 ][ a2 ][0] = x
                nnnn_left[ a1 ][ a2 ][1] = y
                nnnn_right[ b1 ][ b2 ][0] = x
                nnnn_right[ b1 ][ b2 ][1] = y
        
        #
        for dfd in range(4,10):
            cv2.imshow('Resized Window', nnnn_left[0:,0:,0]<<dfd)
            cv2.waitKey(0)
            print("DDDDDDD")