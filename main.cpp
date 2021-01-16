
#include <iostream>  
#include <string>
#include <time.h>
#include <vector>
#include <thread>

#include <unistd.h>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;  
using namespace std;

//######################################################

int w   = 640;//2592;
int h   = 480;//1944;
int bit =  10;//  10;

double k1  = 0.031021, k2 = -0.123024, p1 = -0.003518, p2 = 0.001929;
double m[] = { 600.802,0,332.532,  0,599.137,264.219,  0,0,1 };	// intrinsic parameters
double d[] = {k1, k2, p1, p2};		// k1,k2: radial distortion, p1,p2: tangential distortion

int square_size = 32;
int rows        =  7;
int cols        =  4;

int thresholder_black = 50;
int thresholder_white = 50;

VideoCapture cap(0);

cv::Mat buffer ;
int buffer_flag    = 0;
int buffer_counter = 0;

//######################################################

void graycode_lookup_table(  int size,  uint16_t *Gray2Bin,  uint16_t *Bin2Gray  );

void spliter(Mat &AAAA);
void capture(VideoCapture &A, Mat &B);
void aqqq(Mat &frame);

void image_to_plane( Mat &gray );
Mat  image_to_bit( Mat &standard_L, Mat &standard_H,  Mat &img_A,  Mat &img_B  );
void image_to_bit_stacking( Mat &stack, Mat &bit, int stage );
void bit_to_cordination( 
    Mat &horizontal, Mat &vertical, Mat &Map_h, Mat &Map_v, 
    uint32_t h, uint32_t w, uint32_t h_im, uint32_t w_im  );
void cordination_to_point();

Mat  projector_map_GRAY(  int h,  int w,  int stage,  int inv,  int direction  );
Mat  projector_map_LAMP(  int h,  int w,  int inv  );

void custom_op_sum(  Mat &in_3d,  int h,  int w  );
void line_plane_intersection( Scalar N, Scalar T, Mat &p2,  int h,  int w );

void processing( VideoCapture &cap );

//######################################################
//#
//######################################################

uint16_t Gray2Bin[1024];
uint16_t Bin2Gray[1024];

void graycode_lookup_table(  int size,  uint16_t *Gray2Bin,  uint16_t *Bin2Gray  ){

    int qqqqq = 1 << (size);

    for(  int binary=0;  binary<qqqqq;  binary++  ){
        uint16_t gray_code = binary ^ (binary >> 1);
        Bin2Gray[binary   ] = gray_code;
        Gray2Bin[gray_code] = binary   ;
    }

}

//######################################################
//# image_control
//######################################################

void spliter(Mat &AAAA){

    //Mat img(5,5,CV_64FC3);
    //Mat AAAA;//, ch2, ch3;

    // "channels" is a vector of 3 Mat arrays:
    vector<Mat> channels(3);

    // split img:
    split(AAAA, channels);

    // get the channels (dont forget they follow BGR order in OpenCV)
    AAAA = channels[2];
    //ch2 = channels[1];
    //ch3 = channels[2];
}

void capture(VideoCapture &A, Mat &B){
    for ( int a = 0; a < 7; a++ ) A >> B;
    spliter(B);
}

void aqqq(Mat &frame){
    moveWindow( "camera1", 1, 1);
    imshow    ( "camera1", frame);  
}

void camera2(Mat &frame){
    moveWindow( "camera2", 1, 1);
    imshow    ( "camera2", frame);  
}

//######################################################
//# processing
//######################################################

void image_to_plane( Mat &gray ){
    
    int w   = 640;//2592;
    int h   = 480;//1944;
    int bit =  10;//  10;
    
    double k1 = 0.031021, k2 = -0.123024, p1 = -0.003518, p2 = 0.001929;
    double m[] = { 600.802,0,332.532,  0,599.137,264.219,  0,0,1 };	// intrinsic parameters
    double d[] = {k1, k2, p1, p2};		// k1,k2: radial distortion, p1,p2: tangential distortion

    int square_size = 32;
    int rows        =  7;
    int cols        =  4;

    //cv::Mat gray ;
    cv::Mat A(3, 3, CV_64FC1, m);						// camera matrix
    cv::Mat distCoeffs(4, 1, CV_64FC1, d);
    cv::Mat rvecs;
    cv::Mat tvecs;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoint;
    Size patternsize(7,4); //interior number of corners

    for( int k = 0; k < cols; k++ )
        for( int j = 0; j < rows; j++ )
            objectPoints.push_back(cv::Point3f(float( k*square_size ), float( j*square_size ), 0));
    
    //######################################################
    //# 체스보드에서 코너를 검출한다.
    //######################################################

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, imagePoint//,
                        //CALIB_CB_ADAPTIVE_THRESH + 
                        //CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
                        );

    //cout << "patternfound = " << endl << " "  << patternfound << endl << endl;

    if(patternfound){

        cornerSubPix(gray, imagePoint, Size(11, 11), Size(-1, -1),
                    TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        //cout << "corners = " << endl << " "  << imagePoint << endl << endl;

        //######################################################
        //# solvePNP 실행함
        //######################################################

        //if(imagePoint.size() != rows*cols)
        //    return;

        cv::solvePnP(  objectPoints,imagePoint,  A,  distCoeffs,  rvecs,tvecs  );

        cout << "rvecs" << endl << " " << rvecs << endl;
        cout << "tvecs" << endl << " " << tvecs << endl << endl;

        Mat rotation_vector;
        Rodrigues(rvecs, rotation_vector);
        transpose(rotation_vector, rotation_vector);
        rotation_vector.convertTo(rotation_vector, CV_32FC1);
        tvecs          .convertTo(tvecs          , CV_32FC1);
            
        float a[1][3] = {{ 0, 0, 1}};
        Mat AA = Mat(1, 3, CV_32FC1, a);
        Mat nomal_vector;
        nomal_vector = AA * rotation_vector;
        cout << "nomal_vector" << endl << " " << nomal_vector << endl << endl << endl;

    }

}

Mat  image_to_bit( Mat &standard_L, Mat &standard_H,  Mat &img_A,  Mat &img_B  ){
    return 
    (   (  (standard_L<img_A)  &  (standard_H<img_A)  )  )  &
    (  ~(  (standard_L<img_B)  &  (standard_H<img_B)  )  )  &  0b1 ;
}

void image_to_bit_stacking( Mat &stack, Mat &bit, int stage ){
    stack = stack | (bit * (0b1<<stage));
}

void bit_to_cordination( 
    Mat &horizontal, Mat &vertical, Mat &Map_h, Mat &Map_v, 
    uint32_t h, uint32_t w, uint32_t h_im, uint32_t w_im  ){
    
    uint16_t * pt_horizontal = (uint16_t*) horizontal.data;
    uint16_t * pt_vertical   = (uint16_t*) vertical  .data;
    uint16_t * pt_Map_h      = (uint16_t*) Map_h     .data;
    uint16_t * pt_Map_v      = (uint16_t*) Map_v     .data;

    uint32_t counter  = 0;
    uint32_t counter1 = 0;
    uint32_t asdasds  = 1024;

    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;

    for (uint32_t y = 0; y < h; y++)
    {
        for (uint32_t x = 0; x < w; x++)
        {
            a = Gray2Bin[pt_vertical  [counter]];
            b = Gray2Bin[pt_horizontal[counter]];
            if(a*b){
                c = a * asdasds + b;
                pt_Map_h[ c ] = x;
                pt_Map_v[ c ] = y;
                counter1++;
            }
            counter++;
        }
    }
    printf("counter1++ %d \n", counter1);

}

void cordination_to_point(){

}

//######################################################
//#
//######################################################

Mat projector_map_GRAY(  int h,  int w,  int stage,  int inv,  int direction  ){

    Mat map(h, w, CV_8UC1);
        
    uchar * pt = map.data;

    int asd = 0;

    if('h'==direction){

        uint16_t Bin2Gray_converted[w];
    
        for (int x = 0; x < w; x++) {
            if(inv) Bin2Gray_converted[x] = ~(( (Bin2Gray[x]>>stage) & 0b1 ) * 255);
            else    Bin2Gray_converted[x] =  (( (Bin2Gray[x]>>stage) & 0b1 ) * 255);
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                pt[asd] = Bin2Gray_converted[x], asd++;
            }
        }
        
    }
    
    if('v'==direction){

        uint16_t Bin2Gray_converted[h];
    
        for (int y = 0; y < h; y++) {
            if(inv) Bin2Gray_converted[y] = ~(( (Bin2Gray[y]>>stage) & 0b1 ) * 255);
            else    Bin2Gray_converted[y] =  (( (Bin2Gray[y]>>stage) & 0b1 ) * 255);
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                pt[asd] = Bin2Gray_converted[y], asd++;
            }
        }
        
    }

    return map;
}
Mat projector_map_LAMP(  int h,  int w,  int inv  ){

    int aa;

    if(inv){
        aa = 0xFF;
    }else{
        aa = 0b0;
    }

    Mat map(h, w, CV_8UC1);
    
    uchar * pt = map.data;

    int asd = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            pt[asd] = aa, asd++;
        }
    }

    return map;
}

//######################################################
//# operation function
//######################################################

void custom_op_sum(  Mat &in_3d,  int h,  int w  ){

    // 3차원을 sum 후 다시 같은 값으로 만들어줌

    in_3d.convertTo( in_3d, CV_32FC3);

    float *  in_3d_pt = (float *) in_3d.data;

    float adf = 0;

    uint32_t asd111 = 0;
    uint32_t asd112 = 1;
    uint32_t asd113 = 2;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            adf = in_3d_pt[asd111] + in_3d_pt[asd112] + in_3d_pt[asd113] ; 
            in_3d_pt[asd111] = adf ;
            in_3d_pt[asd112] = adf ;
            in_3d_pt[asd113] = adf ;
            asd111+=3,  asd112+=3,  asd113+=3;
        }
    }

}

void line_plane_intersection( Scalar N, Scalar T,  Mat &p2,  int h,  int w ){

    //#직선의 방정식
    //#P       #평면 위의 임의의 점 교차점
    //#P1      #선이 지나는 이미 알고 있는 2개의 점을 각각 P1, P2
    //#p2
    //#U       #선에 대한 기울기 값
    //#P=P1+U(P2-P1)

    //####구하는 방법####
    //# N*(P1+u(P2-P1))=N*P3
    //# U = [N*(P3-P1)] / [N*(P2-P1)]
    //# 직선의 방정식에 u를 집어넣는다.
        
        
    //P1=np.array(point1)
    //P2=np.array(point2)
    //P3=np.array(plane_point )
    //N =np.array(plane_vactor)
    //U = (    sum(N * (P3-P1))    /    sum(N * (P2-P1))    )
    //(P1+np.multiply(U,(P2-P1)))
    
    cv::Mat nomal_vector(  cv::Size(3, 3),  CV_32FC3,  N  ); 
    cv::Mat  translation(  cv::Size(3, 3),  CV_32FC3,  T  );
    cv::Mat           p1(  cv::Size(3, 3),  CV_32FC3,  Scalar(   0,   0,   0  )  );
    cv::Mat            U; 
    cv::Mat         sssU; 
        
    //cout << "         p1" << endl <<  " "  <<          p1 << endl << endl;
    //cout << "         p2" << endl <<  " "  <<          p2 << endl << endl;
    //cout << "translation" << endl <<  " "  << translation << endl << endl;

    U    = nomal_vector.mul(translation)  /  nomal_vector.mul(p2)  ;
    //cout << "          U" << endl <<  " "  <<           U << endl << endl;

    custom_op_sum(  U,  3,  3  );
    sssU = U.mul(p2);
    //cout << "       sssU" << endl <<  " "  <<        sssU << endl << endl;

}

//######################################################
//#
//######################################################

void graycode_capture  ( VideoCapture &cap ){
    
    int thresholder_black = 50;
    int thresholder_white = 50;

    int adfadf = 600;

    Mat projector;

    Mat black     (h, w, CV_16SC1);
    Mat white     (h, w, CV_16SC1);
    Mat buffer    (h, w, CV_16SC1);
    Mat buffer_inv(h, w, CV_16SC1);
    Mat bit_h     (h, w, CV_16UC1);
    Mat bit_v     (h, w, CV_16UC1);
    Mat bit_buffer(h, w, CV_16UC1);

    //######################################################

    projector = projector_map_LAMP(  1024,  1024,  0  );
    camera2(projector);
    waitKey(1000);
    capture(cap, black);
    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_h */
        projector = projector_map_GRAY(  1024, 1024,  stage,  0,  'h'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer);

        projector = projector_map_GRAY(  1024, 1024,  stage,  1,  'h'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer_inv);
        
    }

    projector = projector_map_LAMP(  1024,  1024,  0  );
    camera2(projector);
    waitKey(1000);
    capture(cap, black);

    projector = projector_map_LAMP(  1024,  1024,  1  );
    camera2(projector);
    waitKey(adfadf);
    capture(cap, white);
    
    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_v */
        projector = projector_map_GRAY(  1024, 1024,  stage,  0,  'v'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer);

        projector = projector_map_GRAY(  1024, 1024,  stage,  1,  'v'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer_inv);
        
    }

}


void processing0_ (){

    /*

    Mat black     (h, w, CV_16SC1);
    Mat white     (h, w, CV_16SC1);
    Mat buffer    (h, w, CV_16SC1);
    Mat buffer_inv(h, w, CV_16SC1);
    Mat bit_h     (h, w, CV_16UC1);
    Mat bit_v     (h, w, CV_16UC1);
    Mat bit_buffer(h, w, CV_16UC1);

    //######################################################

    capture(cap, black);
    capture(cap, white);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;

    for ( int stage = 0; stage < 10; stage++ )  
    {

        *//* graycode_h *//*
        capture(cap, buffer);
        capture(cap, buffer_inv);

        buffer    .convertTo(buffer    , CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_h,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }
    
    //######################################################

    capture(cap, black);
    capture(cap, white);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;
    
    for ( int stage = 0; stage < 10; stage++ )  
    {

        *//* graycode_v *//*
        capture(cap, buffer);
        capture(cap, buffer_inv);
        
        buffer    .convertTo(buffer    , CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_v,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }

    bit_to_cordination(  bit_h,bit_v,  bit_hasdas,bit_vasdas,  h,w,  1024,1024  );
   */
}
void processing0__( VideoCapture &cap ){

    int w   = 640;//2592;
    int h   = 480;//1944;
    int bit =  10;//  10;
    int thresholder_black = 50;
    int thresholder_white = 50;

    int adfadf = 600;

    Mat projector;

    Mat black     (h, w, CV_16SC1);
    Mat white     (h, w, CV_16SC1);
    Mat buffer    (h, w, CV_16SC1);
    Mat buffer_inv(h, w, CV_16SC1);
    Mat bit_h     (h, w, CV_16UC1);
    Mat bit_v     (h, w, CV_16UC1);
    Mat bit_buffer(h, w, CV_16UC1);

    //######################################################

    projector = projector_map_LAMP(  1024,  1024,  0  );
    camera2(projector);
    waitKey(1000);
    camera2(projector);
    waitKey(1000);
    capture(cap, black);

    projector = projector_map_LAMP(  1024,  1024,  1  );
    camera2(projector);
    waitKey(adfadf);
    capture(cap, white);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;

    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_h */
        projector = projector_map_GRAY(  1024, 1024,  stage,  0,  'h'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer);

        projector = projector_map_GRAY(  1024, 1024,  stage,  1,  'h'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer_inv);

        buffer    .convertTo(buffer    , CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_h,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }

    Mat bit_hqwe = bit_h*(1<<6);
    camera2(bit_hqwe);
    waitKey(12500);
    
    //######################################################

    projector = projector_map_LAMP(  1024,  1024,  0  );
    camera2(projector);
    waitKey(1000);
    camera2(projector);
    waitKey(1000);
    capture(cap, black);

    projector = projector_map_LAMP(  1024,  1024,  1  );
    camera2(projector);
    waitKey(adfadf);
    capture(cap, white);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;
    
    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_v */
        projector = projector_map_GRAY(  1024, 1024,  stage,  0,  'v'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer);

        projector = projector_map_GRAY(  1024, 1024,  stage,  1,  'v'  );
        camera2(projector);
        waitKey(adfadf);
        capture(cap, buffer_inv);
        
        buffer    .convertTo(buffer    , CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_v,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }

    Mat dfdbit_v = bit_v*(1<<6);
    camera2(dfdbit_v);
    waitKey(12500);

    //######################################################
    //#
    //######################################################

    Mat bit_hasdas   (1024, 1024, CV_16UC1);
    Mat bit_vasdas   (1024, 1024, CV_16UC1);

    bit_to_cordination(  bit_h,bit_v,  bit_hasdas,bit_vasdas,  h,w,  1024,1024  );

    bit_hasdas = bit_hasdas*100;
    camera2(bit_hasdas);
    waitKey(12500);
    bit_vasdas = bit_vasdas*100;
    camera2(bit_vasdas);
    waitKey(12500);
    
}
void processing1  ( VideoCapture &cap ){

}

void thread_camera(){

    if (!cap.isOpened())  printf("카메라를 열수 없습니다. \n");  

	cap.set( CAP_PROP_FRAME_WIDTH , w );
	cap.set( CAP_PROP_FRAME_HEIGHT, h );
	cap.set( CAP_PROP_ISO_SPEED, 100 );
	//cap.set( CAP_PROP_EXPOSURE, 0.1 );
	cap.set( CAP_PROP_FPS, 10);

    while(1){

        //usleep(10000);
        graycode_capture(cap);
        //if(buffer_flag==0) capture(cap, buffer), buffer_flag = 1, buffer_counter++;
        //cout << "captured!\n"<<buffer_counter<<endl<<endl;
        //printf("bbbbbbbbbbbbbbb\n");
    }

}
void thread_main  (){
    
    //######################################################
    clock_t start, start1, end, end1;
    double result, result1;
    int i,j;
    int sum = 0;
    start = clock(); //시간 측정 시작
    start1 = time(NULL); // 시간 측정 시작
    //######################################################

    graycode_lookup_table(  10,  Gray2Bin,  Bin2Gray  );
    namedWindow("camera1", 1);
    namedWindow("camera2", 1);

    //######################################################
    //#
    //######################################################

    cv::Mat gray ;
    cv::Mat A(3, 3, CV_64FC1, m);						// camera matrix
    cv::Mat distCoeffs(4, 1, CV_64FC1, d);
    cv::Mat rvecs;
    cv::Mat tvecs;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoint;
    Size patternsize(7,4); //interior number of corners

    for( int k = 0; k < cols; k++ )
        for( int j = 0; j < rows; j++ )
            objectPoints.push_back(cv::Point3f(float( k*square_size ), float( j*square_size ), 0));

    //######################################################
    //#
    //######################################################
    
    while(1){

        printf("aaaaaaaaaaaaaaaaaaa\n");
        
        usleep(10000);
        //if(buffer_flag!=0) buffer_flag = 0, aqqq(buffer), waitKey(20);

    }
   
    //######################################################
    end     = clock(); //시간 측정 끝
    end1    = time(NULL); // 시간 측정 끝
    result  = (double)(end - start);
    result1 = (double)(end1 - start1);
    printf("%f\n", result);
    printf("%f\n", result1); //결과 출력
    //######################################################
    
}

int main(int, char**)
{   

    thread hi0(thread_main  );
    thread hi1(thread_camera);
    hi0.join();
    hi1.join();
    printf("aaaaaaaaaaaaaaaaaaa");
    printf("aaaaaaaaaaaaaaaaaaa");

    //while(1){
        
    //}

    //capture( cap, gray );

    /*
    image_to_plane( gray );
    
    processing0( cap );

    cv::Mat point(  cv::Size(3, 3),  CV_32FC3,  Scalar(   1,   1, 100  )  );

    line_plane_intersection( 
        Scalar(    0,  0.5,  0.5  ),  Scalar(  222,  222,  222  ), 
        point,  3,  3
    );
    */
  
    return 0;

}
