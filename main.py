import cv2
import numpy as np

import image_processing
import image_load
import setting
import projector_map
import calib
import calc



if __name__ == "__main__":

    R, t = [], []
    cam_mat_l, cam_mat_r = [], []
    pts_l,pts_r = [], []
    point_list = []

    # camera calibration
    if(1):
        
        img_L = image_load.chassboard(0)
        img_R = image_load.chassboard(1)

        objpoints_L,imgpoints_L = calib.get_chassboard_Corner(img_L)
        objpoints_R,imgpoints_R = calib.get_chassboard_Corner(img_R)

        ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = calib.get_camera_matrix(
            objpoints_L,imgpoints_L,img_L)
        ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = calib.get_camera_matrix(
            objpoints_R,imgpoints_R,img_R)
        
        R, t = calib.get_rotation_transpose(
            imgpoints_L[0,0:,0,0:],imgpoints_R[0,0:,0,0:],
            mtx_L,mtx_R
        )

        cam_mat_l, cam_mat_r = mtx_R, mtx_R
    
        #print(imgpoints_L[0,0:,0,0:].shape)
        #print(imgpoints_R[0,0:,0,0:].shape)
        
        #print(rvecs_L, tvecs_L)
        #print(rvecs_R, tvecs_R)

        print(R)
        print(t)

        print(mtx_L)
        print(mtx_R)

        print("1. camera calibration")

    # image to graycode
    if(0):
        
        #
        aaaa_H = np.zeros(setting.camera_size_mat,np.uint16)
        aaaa_V = np.zeros(setting.camera_size_mat,np.uint16)
        bbbb_H = np.zeros(setting.camera_size_mat,np.uint16)
        bbbb_V = np.zeros(setting.camera_size_mat,np.uint16)

        # structed 이미지를 single graycode 이미지로 쌓아올림
        for bit_position in range(setting.resolution_bit):
            
            # 베이스라인이 되는 이미지
            w = np.array(image_load.base(0,1),np.int16)
            b = np.array(image_load.base(0,0),np.int16)

            g = np.array(image_load.gary(0,bit_position,"h"),np.int16)
            a = image_processing.bitlize(b,g,w)
            aaaa_H = image_processing.stacking(aaaa_H,a,bit_position)
            
            g = np.array(image_load.gary(0,bit_position,"v"),np.int16)
            a = image_processing.bitlize(b,g,w)
            aaaa_V = image_processing.stacking(aaaa_V,a,bit_position)
            
            # 베이스라인이 되는 이미지
            w = np.array(image_load.base(1,1),np.int16)
            b = np.array(image_load.base(1,0),np.int16)
            
            g = np.array(image_load.gary(1,bit_position,"h"),np.int16)
            a = image_processing.bitlize(b,g,w)
            bbbb_H = image_processing.stacking(bbbb_H,a,bit_position)

            g = np.array(image_load.gary(1,bit_position,"v"),np.int16)
            a = image_processing.bitlize(b,g,w)
            bbbb_V = image_processing.stacking(bbbb_V,a,bit_position)

            #cv2.imshow('Resized Window0', aaaa_H<<(15-bit_position))
            #cv2.imshow('Resized Window1', aaaa_V<<(15-bit_position))
            #cv2.imshow('Resized Window2', bbbb_H<<(15-bit_position))
            #cv2.imshow('Resized Window3', bbbb_V<<(15-bit_position))
            
            #cv2.waitKey(250)

            #cv2.imshow('Resized Window', bbbb<<6)
            #cv2.waitKey(1100)

            print(bit_position)

        np.save('./np_list_data/aaaa_H',   aaaa_H)
        np.save('./np_list_data/aaaa_V',   aaaa_V)
        np.save('./np_list_data/bbbb_H',   bbbb_H)
        np.save('./np_list_data/bbbb_V',   bbbb_V)

    # graycode to position
    if(1):

        # 카메라메트릭스를 지워서 호모지니어스로 만들어야함

        #
        aaaa_H = np.load('./np_list_data/aaaa_H.npy')
        aaaa_V = np.load('./np_list_data/aaaa_V.npy')
        bbbb_H = np.load('./np_list_data/bbbb_H.npy')
        bbbb_V = np.load('./np_list_data/bbbb_V.npy')

        nnnn_left = np.zeros([setting.projector_size[0],setting.projector_size[1],2],np.float32)
        nnnn_right = np.zeros([setting.projector_size[0],setting.projector_size[1],2],np.float32)
        
        # single graycode 이미지를 coordinate image로 전환함
        for y in range(0,setting.camera_size_mat[0]):
            for x in range(0,setting.camera_size_mat[1]):
                a1,a2 = projector_map.Gray2Bin[aaaa_H[y][x]],projector_map.Gray2Bin[aaaa_V[y][x]]
                b1,b2 = projector_map.Gray2Bin[bbbb_H[y][x]],projector_map.Gray2Bin[bbbb_V[y][x]]
                #a1,a2 = aaaa_H[y][x],aaaa_V[y][x]
                #b1,b2 = bbbb_H[y][x],bbbb_V[y][x]
                nnnn_left[ a1 ][ a2 ][0] = x
                nnnn_left[ a1 ][ a2 ][1] = y
                nnnn_right[ b1 ][ b2 ][0] = x
                nnnn_right[ b1 ][ b2 ][1] = y

        s = 0
        for x in range(1,setting.projector_size[1]):
            for y in range(1,setting.projector_size[0]):
                if((nnnn_left[ x ][ y ][0]!=0)&(nnnn_right[ x ][ y ][0]!=0)):
                    point_list.append([
                        float(nnnn_left[ x ][ y ][0] ),float(nnnn_left[ x ][ y ][1] ),
                        float(nnnn_right[ x ][ y ][0]),float(nnnn_right[ x ][ y ][1]),
                    ])
                    s+=1
        print(s)

        #
        np.save('./np_list_data/nnnn_left',   nnnn_left)
        np.save('./np_list_data/nnnn_right',   nnnn_right)
        
        ##
        #cv2.imshow('Resized Window', nnnn_left[0:,0:,0]/1500)
        #cv2.waitKey(0)

        print("DDDDDDD")
    
    # triangulation
    if(0):
        
        #
        nnnn_left = np.load('./np_list_data/nnnn_left.npy').reshape(1,-1,2)
        nnnn_right = np.load('./np_list_data/nnnn_right.npy').reshape(1,-1,2)

        #
        a =  0.5+0.2
        b = -0.5+0.2

        #
        R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        t = np.array([2,0,0])

        #
        xyz = calc.triangulate(np.array([a,0,b,0]),R,t)

        print(xyz)