
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
//# 장비정보
//######################################################

int adfadf = 600;

int projector_map_shape_x = 1024;
int projector_map_shape_y = 1024;

float focus = 600;

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

//######################################################
//# 함수선언
//######################################################

void graycode_lookup_table(  int size,  uint16_t *Gray2Bin,  uint16_t *Bin2Gray  );

void spliter(Mat &AAAA);
void capture(VideoCapture &A, Mat &B);
void camera1(Mat &frame);
void camera2(Mat &frame);

void image_to_plane( Mat &gray, Scalar &aaaaaadfffff );
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
//# 수표현도구
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
    A.grab(), A.grab(), A.grab(), A.grab(), A.grab(), A.grab(), A.retrieve(B); //A >> B;
    spliter(B);
}
void camera1(Mat &frame){
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

void image_to_plane( Mat &gray, float *aaaaaadfffff ){
    
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
        //cout << "nomal_vector" << endl << " " << nomal_vector << endl << endl << endl;
        //aaaaaadfffff = (Scalar)nomal_vector;

        float *dfdf;

        dfdf = (float*) nomal_vector.data;
        aaaaaadfffff[0] = dfdf[0];
        aaaaaadfffff[1] = dfdf[1];
        aaaaaadfffff[2] = dfdf[2];
        
        dfdf = (float*) tvecs.data;
        aaaaaadfffff[3] = dfdf[0];
        aaaaaadfffff[4] = dfdf[1];
        aaaaaadfffff[5] = dfdf[2];
        
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
    Mat &horizontal, Mat &vertical, Mat &Map_hv, 
    uint32_t h, uint32_t w, uint32_t h_im, uint32_t w_im  ){
    
    uint16_t * pt_horizontal = (uint16_t*) horizontal.data;
    uint16_t * pt_vertical   = (uint16_t*) vertical  .data;
    float    * pt_Map_hv     = (float   *) Map_hv    .data;

    uint32_t right_point_counter = 0;
    uint32_t counter             = 0;
    uint32_t x_size              = 1024;

    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t c = 0;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            a = Gray2Bin[pt_vertical  [counter]];
            b = Gray2Bin[pt_horizontal[counter]];
            if (a*b) {
                c = (a*x_size+b) * 3;
                pt_Map_hv[ c ] = x, c++;
                pt_Map_hv[ c ] = y, c++;
                pt_Map_hv[ c ] = 600;
                right_point_counter++;
            }
            counter++;
        }
    }

    printf("right_point_counter++ %d \n", right_point_counter);

}

void cordination_to_point( Mat &Map_hv, float *plane_vectors ){

    //cv::Mat point(  cv::Size(3, 3),  CV_32FC3,  Scalar(   1,   1, 100  )  );
    //line_plane_intersection( 
    //    Scalar(  plane_vectors[0],  plane_vectors[1],  plane_vectors[2]  ),
    //    Scalar(  plane_vectors[3],  plane_vectors[4],  plane_vectors[5]  ),
    //    point,  3,  3
    //);
    line_plane_intersection( 
        Scalar(  plane_vectors[0],  plane_vectors[1],  plane_vectors[2]  ),
        Scalar(  plane_vectors[3],  plane_vectors[4],  plane_vectors[5]  ),
        Map_hv,  1024,  1024
    );

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

VideoCapture cap(0);

Mat buffersasasa(h, w, CV_16SC1);

int image_type    = 0;  // background--graycode
int image_inv     = 0;  // graycode
int image_dir     = 0;  // graycode
int image_stage   = 0;  // graycode
int image_onoff   = 0;  // background
int ready_flag    = 0;
int capture_algorithm_state = 0;
void capture_algorithm  ( VideoCapture &cap ) {

    static Mat projector;

    //######################################################
    //#
    //######################################################
    while (capture_algorithm_state) {

        if(ready_flag == 0){

            if (image_type == 0) {
                projector = projector_map_LAMP(  1024,  1024,  image_onoff  );
                camera2(projector);
                waitKey(adfadf);
                capture(cap, buffersasasa);
                ready_flag = 1;
            } 

            else if (image_type == 1) {
                projector = projector_map_GRAY(  1024, 1024,  image_stage,  image_inv,  image_dir  );
                camera2(projector);
                waitKey(adfadf);
                capture(cap, buffersasasa);
                ready_flag = 1;
            }

            else if (image_type == 2) {
                capture(cap, buffersasasa);
                ready_flag = 1;
            }

        }

    }
    //######################################################


    //######################################################
    //#
    //######################################################
    if(capture_algorithm_state==0){

        image_type    = 0;
        image_inv     = 0;
        image_dir     = 0;
        image_stage   = 0;
        image_onoff   = 0;
        ready_flag    = 0;

    }
    //######################################################

}
void capture_algorithm_start(){
    capture_algorithm_state = 1;
}
void capture_algorithm_stop (){
    capture_algorithm_state = 0;
}
void capture_background( Mat &aafsa, int color ){

    image_type  = 0;
    image_onoff = color;

    ready_flag = 0;
    while (ready_flag==0) usleep(10000);
    aafsa = buffersasasa;
    ready_flag = 1;
    
}
void capture_graycode  ( Mat &aafsa, int stage, int invert, int dir ){
    
    image_type    = 1;
    image_inv     = invert;
    image_dir     = dir;
    image_stage   = stage;

    ready_flag = 0;
    while (ready_flag==0) usleep(10000);
    aafsa = buffersasasa;
    ready_flag = 1;

}
void capture_plane     ( Mat &aafsa ){
    
    image_type    = 2;

    ready_flag = 0;
    while (ready_flag==0) usleep(40000);
    aafsa = buffersasasa;
    ready_flag = 1;

}

//######################################################
//#
//######################################################


void graycode_map( Mat &aafsa ){

    Mat black     (h, w, CV_16SC1);
    Mat white     (h, w, CV_16SC1);
    Mat buffer    (h, w, CV_16SC1);
    Mat buffer_inv(h, w, CV_16SC1);
    Mat bit_h     (h, w, CV_16UC1);
    Mat bit_v     (h, w, CV_16UC1);
    Mat bit_buffer(h, w, CV_16UC1);

    Mat bit_hvsdas   (1024, 1024, CV_32FC3);

    capture_background(black,0);
    capture_background(white,1);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;

    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_h */
        capture_graycode(      buffer,  stage,  0,  'h'  );
        capture_graycode(  buffer_inv,  stage,  1,  'h'  );

        buffer    .convertTo(    buffer, CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_h,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }
    
    for ( int stage = 0; stage < 10; stage++ )  
    {

        /* graycode_v */
        capture_graycode(      buffer,  stage,  0,  'v'  );
        capture_graycode(  buffer_inv,  stage,  1,  'v'  );
        
        buffer    .convertTo(buffer    , CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);

        bit_buffer = image_to_bit(  black,  white,  buffer,  buffer_inv  );
        bit_buffer.convertTo(bit_buffer, CV_16UC1);

        image_to_bit_stacking(  bit_v,  bit_buffer,  stage  );

        printf("%d\n",stage);
        
    }

    //Mat ssbit_h = bit_h*(1<<6);
    //camera2(ssbit_h);
    //waitKey(6000);

    //Mat ssbit_v = bit_v*(1<<6);
    //camera2(ssbit_v);
    //waitKey(6000);

    //######################################################
    //#
    //######################################################

    bit_to_cordination(  bit_h,bit_v,  bit_hvsdas,  h,w,  1024,1024  );

    //bit_hasdas = bit_hasdas*100;
    //camera2(bit_hasdas);
    //waitKey(6000);

    //bit_vasdas = bit_vasdas*100;
    //camera2(bit_vasdas);
    //waitKey(6000);


    //    //######################################################
    //    //# 좌표를 3차원으로 바꾸고 
    //    //######################################################

    Mat adsadasdas   (1024, 1024, CV_32FC3);

    //    uint16_t * x = (uint16_t*) bit_hasdas.data;
    //    uint16_t * y = (uint16_t*) bit_vasdas.data;
    //    float    * w = (float   *) adsadasdas.data;
    //    
    //    uint64_t counter  = 0;
    //
    //    for (uint32_t y = 0; y < projector_map_shape_y; y++)
    //    {
    //        for (uint32_t x = 0; x < projector_map_shape_x; x++)
    //        {
    //            w[counter  ] = x;
    //            w[counter+1] = y;
    //            w[counter+2] = focus;
    //            counter += 3;
    //        }
    //    }

    aafsa = bit_hvsdas;
    
    adsadasdas = bit_hvsdas/600;
    camera2(adsadasdas);
    waitKey(6000);
    
}
void image_to_planeaaaaa(float *plane_vectors){

    //float plane_vectors[] = {0,0,0,0,0,0};
    Mat gray;
    capture_plane( gray );
    image_to_plane( gray, plane_vectors );

    printf("%f, %f, %f,    %f, %f, %f \n\n",
        plane_vectors[0], plane_vectors[1], plane_vectors[2],
        plane_vectors[3], plane_vectors[4], plane_vectors[5]
    );

}
void processing(){
    
    Mat gray;
    float plane_vectors[] = {0,0,0,0,0,0};

    graycode_map(gray);
    image_to_planeaaaaa(plane_vectors);
    cordination_to_point( gray, plane_vectors );
    //calc_point_map();
}


void thread_camera(){

    while(1){
        usleep(5000);
        capture_algorithm(cap);
        //if(buffer_flag==0) capture(cap, buffer), buffer_flag = 1, buffer_counter++;
        //cout << "captured!\n"<<buffer_counter<<endl<<endl;
        //printf("bbbbbbbbbbbbbbb\n");
    }

}
void thread_main  (){
    
    ////######################################################
    //clock_t start, start1, end, end1;
    //double result, result1;
    //int i,j;
    //int sum = 0;
    //start = clock(); //시간 측정 시작
    //start1 = time(NULL); // 시간 측정 시작
    ////######################################################

    graycode_lookup_table(  bit,  Gray2Bin,  Bin2Gray  );

    namedWindow("camera1", 1);
    namedWindow("camera2", 1);

    if (!cap.isOpened())  printf("카메라를 열수 없습니다. \n");  

	cap.set( CAP_PROP_FRAME_WIDTH , w );
	cap.set( CAP_PROP_FRAME_HEIGHT, h );
	cap.set( CAP_PROP_ISO_SPEED, 100 );
	//cap.set( CAP_PROP_EXPOSURE, 0.1 );
	cap.set( CAP_PROP_FPS, 30);

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

    for ( int k = 0; k < cols; k++ )
        for ( int j = 0; j < rows; j++ )
            objectPoints.push_back(cv::Point3f(float( k*square_size ), float( j*square_size ), 0));

    //######################################################
    //#
    //######################################################

    sleep(5);

    while (1) {
    
        capture_algorithm_stop ();
        capture_algorithm_start();

        printf("aaaaaaaaaaaaaaaaaaa\n");
        
        //image_processing0();
        processing();

    }
   
    ////######################################################
    //end     = clock(); //시간 측정 끝
    //end1    = time(NULL); // 시간 측정 끝
    //result  = (double)(end - start);
    //result1 = (double)(end1 - start1);
    //printf("%f\n", result);
    //printf("%f\n", result1); //결과 출력
    ////######################################################
    
}


int main (int, char**) {   

    printf("aaaaaaaaaaaaaaaaaaa");
    printf("aaaaaaaaaaaaaaaaaaa");
    thread hi0(thread_main  );
    thread hi1(thread_camera);
    hi0.join();
    hi1.join();
    
    return 0;
}
