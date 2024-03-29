#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>
#include <thread>
#include <cmath>
#include <random>

#include <unistd.h>

#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <gtk/gtk.h>

using namespace cv;
using namespace std;

//######################################################
//# 장비정보
//######################################################

int adfadf = 600;

int projector_map_shape_x = 1024;
int projector_map_shape_y = 1024;

float focus = 600;

int w = 640;  //2592;
int h = 480;  //1944;
int bit = 10; //  10;

double k1 = 0.031021, k2 = -0.123024, p1 = -0.003518, p2 = 0.001929;
double m[] = {600.802, 0, 332.532, 0, 599.137, 264.219, 0, 0, 1}; // intrinsic parameters
double d[] = {k1, k2, p1, p2};                                    // k1,k2: radial distortion, p1,p2: tangential distortion

int square_size = 32;
int rows = 7;
int cols = 4;

int thresholder_black = 50;
int thresholder_white = 50;

//######################################################
//# 함수선언
//######################################################

void graycode_lookup_table(int size, uint16_t *Gray2Bin, uint16_t *Bin2Gray);

void spliter(Mat &AAAA);
void capture(VideoCapture &A, Mat &B);
void camera2(Mat &frame);

void image_to_plane(Mat &gray, Scalar &aaaaaadfffff);
Mat image_to_bit(Mat &standard_L, Mat &standard_H, Mat &img_A, Mat &img_B);
void image_to_bit_stacking(Mat &stack, Mat &bit, int stage);
void bit_to_cordination(
    Mat &horizontal, Mat &vertical, Mat &Map_hv, float focus,
    uint32_t h, uint32_t w, uint32_t h_im, uint32_t w_im);
void cordination_to_point(Mat &Map_hv, float *plane_vectors, uint32_t h, uint32_t w);

Mat projector_map_GRAY(int h, int w, int stage, int inv, int direction);
Mat projector_map_LAMP(int h, int w, int inv);

void processing();

void gui_main();
void gui_preview_update(Mat &qwerqer);
static void move_button(GtkWidget *button);
static void gui_graycode_show(Mat &qwerqer);

//######################################################
//# 수표현도구
//######################################################

uint16_t Gray2Bin[1024];
uint16_t Bin2Gray[1024];

void graycode_lookup_table(int size, uint16_t *Gray2Bin, uint16_t *Bin2Gray)
{

    int qqqqq = 1 << (size);

    for (int binary = 0; binary < qqqqq; binary++)
    {
        uint16_t gray_code = binary ^ (binary >> 1);
        Bin2Gray[binary] = gray_code;
        Gray2Bin[gray_code] = binary;
    }
}

//######################################################
//# processing
//######################################################

void image_to_plane(Mat &gray, float *aaaaaadfffff)
{

    int w = 640;  //2592;
    int h = 480;  //1944;
    int bit = 10; //  10;

    //double k1 = 0.031021, k2 = -0.123024, p1 = -0.003518, p2 = 0.001929;
    //double m[] = {600.802, 0, 332.532, 0, 599.137, 264.219, 0, 0, 1};
    //double d[] = {k1, k2, p1, p2};                                   

    int square_size = 32;
    int rows = 7;
    int cols = 4;

    //cv::Mat gray ;
    cv::Mat A(3, 3, CV_64FC1, m); // camera matrix
    cv::Mat distCoeffs(4, 1, CV_64FC1, d);
    cv::Mat rvecs;
    cv::Mat tvecs;
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoint;
    Size patternsize(7, 4); //interior number of corners

    for (int k = 0; k < cols; k++)
        for (int j = 0; j < rows; j++)
            objectPoints.push_back(cv::Point3f(float(k * square_size), float(j * square_size), 0));

    //######################################################
    //# 체스보드에서 코너를 검출한다.
    //######################################################

    //CALIB_CB_FAST_CHECK saves a lot of time on images
    //that do not contain any chessboard corners
    bool patternfound = findChessboardCorners(gray, patternsize, imagePoint //,
                                              //CALIB_CB_ADAPTIVE_THRESH +
                                              //CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK
    );

    //cout << "patternfound = " << endl << " "  << patternfound << endl << endl;

    if (patternfound)
    {

        cornerSubPix(gray, imagePoint, Size(11, 11), Size(-1, -1),
                     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        //cout << "corners = " << endl << " "  << imagePoint << endl << endl;

        //######################################################
        //# solvePNP 실행함
        //######################################################

        //if(imagePoint.size() != rows*cols)
        //    return;

        cv::solvePnP(objectPoints, imagePoint, A, distCoeffs, rvecs, tvecs);

        cout << "rvecs" << endl
             << " " << rvecs << endl;
        cout << "tvecs" << endl
             << " " << tvecs << endl
             << endl;

        Mat rotation_vector;
        Rodrigues(rvecs, rotation_vector);
        transpose(rotation_vector, rotation_vector);
        rotation_vector.convertTo(rotation_vector, CV_32FC1);
        tvecs.convertTo(tvecs, CV_32FC1);

        float a[1][3] = {{0, 0, 1}};
        Mat AA = Mat(1, 3, CV_32FC1, a);
        Mat nomal_vector;
        nomal_vector = AA * rotation_vector;
        //cout << "nomal_vector" << endl << " " << nomal_vector << endl << endl << endl;
        //aaaaaadfffff = (Scalar)nomal_vector;

        float *dfdf;

        dfdf = (float *)nomal_vector.data;
        aaaaaadfffff[0] = dfdf[0];
        aaaaaadfffff[1] = dfdf[1];
        aaaaaadfffff[2] = dfdf[2];

        dfdf = (float *)tvecs.data;
        aaaaaadfffff[3] = dfdf[0];
        aaaaaadfffff[4] = dfdf[1];
        aaaaaadfffff[5] = dfdf[2];
    }
}

Mat image_to_bit(Mat &standard_L, Mat &standard_H, Mat &img_A, Mat &img_B)
{
    return (((standard_L < img_A) & (standard_H < img_A))) &
           (~((standard_L < img_B) & (standard_H < img_B))) & 0b1;
}

void image_to_bit_stacking(Mat &stack, Mat &bit, int stage)
{
    stack = stack | (bit * (0b1 << stage));
}


void bit_to_cordination(
    Mat &horizontal, Mat &vertical, Mat &Map_hv, float focus,
    uint32_t h, uint32_t w, uint32_t h_im, uint32_t w_im)
{

    uint16_t *pt_horizontal = (uint16_t *)horizontal.data;
    uint16_t *pt_vertical = (uint16_t *)vertical.data;
    float *pt_Map_hv = (float *)Map_hv.data;

    uint32_t right_point_counter = 0;
    uint32_t counter = 0;
    uint32_t x_size = w_im;

    uint64_t a = 0, b = 0, c = 0;

    for (uint32_t y = 0; y < h; y++)
    {
        for (uint32_t x = 0; x < w; x++)
        {
            a = Gray2Bin[pt_vertical[counter]];
            b = Gray2Bin[pt_horizontal[counter]];
            if (a * b)
            {
                c = (a * x_size + b) * 3;
                pt_Map_hv[c] = x;
                pt_Map_hv[c + 1] = y;
                pt_Map_hv[c + 2] = focus;
                right_point_counter++;
            }
            counter++;
        }
    }

    printf("right_point_count %d \n", right_point_counter);
}

void cordination_to_point(Mat &Map_hv, float *plane_vectors, uint32_t h, uint32_t w)
{
    int a = 0;
    float *map = (float *)Map_hv.data;
    float *p2 = (float *)Map_hv.data;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (p2[a] * p2[a + 1] * p2[a + 2])
            {
                // U = ( N*(TRANSLATION-P1) ) / ( N*(P2-P1) )
                float U1 = (plane_vectors[0] * plane_vectors[3]) / (plane_vectors[0] * p2[a]);
                float U2 = (plane_vectors[1] * plane_vectors[4]) / (plane_vectors[1] * p2[a + 1]);
                float U3 = (plane_vectors[2] * plane_vectors[5]) / (plane_vectors[2] * p2[a + 2]);
                if (isnan(U1))
                    U1 = 0;
                if (isnan(U2))
                    U2 = 0;
                if (isnan(U3))
                    U3 = 0;
                float U = U1 + U2 + U3;

                // P = P1 + U * P2
                map[a] = U * p2[a];
                map[a + 1] = U * p2[a + 1];
                map[a + 2] = U * p2[a + 2];
            }
            a = a + 3;
        }
    }
}

//######################################################
//#
//######################################################

Mat projector_map_GRAY(int h, int w, int stage, int inv, int direction)
{

    Mat map(h, w, CV_8UC1);

    uchar *pt = map.data;

    int asd = 0;

    if ('h' == direction)
    {

        uint16_t Bin2Gray_converted[w];

        for (int x = 0; x < w; x++)
        {
            if (inv)
                Bin2Gray_converted[x] = ~(((Bin2Gray[x] >> stage) & 0b1) * 255);
            else
                Bin2Gray_converted[x] = (((Bin2Gray[x] >> stage) & 0b1) * 255);
        }

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                pt[asd] = Bin2Gray_converted[x], asd++;
            }
        }
    }

    if ('v' == direction)
    {

        uint16_t Bin2Gray_converted[h];

        for (int y = 0; y < h; y++)
        {
            if (inv)
                Bin2Gray_converted[y] = ~(((Bin2Gray[y] >> stage) & 0b1) * 255);
            else
                Bin2Gray_converted[y] = (((Bin2Gray[y] >> stage) & 0b1) * 255);
        }

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                pt[asd] = Bin2Gray_converted[y], asd++;
            }
        }
    }

    return map;
}
Mat projector_map_LAMP(int h, int w, int inv)
{

    int aa;

    if (inv)
    {
        aa = 0xFF;
    }
    else
    {
        aa = 0b0;
    }

    Mat map(h, w, CV_8UC1);

    uchar *pt = map.data;

    int asd = 0;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            pt[asd] = aa, asd++;
        }
    }

    return map;
}

//######################################################
//# operation function
//######################################################

void LSM(float *MAT_A, int a_y, int a_x, float *MAT_B, float *MAT_X)
{

    //float  MAT_A[15] = {  -1,1,2,  3,-1,1,  -1,3,4,  1,1,1,  1,1,1  };
    //float  MAT_B[5]  = {   1,1,1,1,1  };
    //int a_y=5, a_x=3;
    //float *MAT_X = (float *)malloc( sizeof(float)*a_y*a_x );
    //for ( int i=0; i<a_x;     i++ )    MAT_X[i]=0;

    float *eye = (float *)malloc(4 * a_x * a_x);
    float *AAt = (float *)malloc(4 * a_x * a_x);
    float *pinv = (float *)malloc(4 * a_y * a_x);

    for (int i = 0; i < a_y * a_x; i++)
        pinv[i] = 0;
    for (int i = 0; i < a_x * a_x; i++)
        eye[i] = 0, AAt[i] = 0;
    for (int i = 0; i < a_x * a_x; i += a_x + 1)
        eye[i] = 1;
    for (int i = 0; i < a_x; i++)
        MAT_X[i] = 0;

    /* ( At A ) */
    for (int i = 0; i < a_x; ++i)
        for (int j = 0; j < a_x; ++j)
            for (int k = 0; k < a_y; ++k)
                AAt[i * a_x + j] += MAT_A[k * a_x + i] * MAT_A[k * a_x + j];
    //// DATA
    //for (int i = 0; i < a_x; ++i) {
    //	for (int j = 0; j < a_x; ++j) cout << AAt[i*a_x+j] << " "; cout << "\n";
    //}

    /* inv( At A )  gause jordan inverse */
    for (int uuuu = 0; uuuu < a_x; uuuu++)
    {
        float buffer = AAt[uuuu * a_x + uuuu];
        for (int j = 0; j < a_x; j++)
        {
            AAt[uuuu * a_x + j] /= buffer;
            eye[uuuu * a_x + j] /= buffer;
        }
        uint32_t xxx = uuuu;
        for (int s = 1; s < a_x; s++)
        {
            ++xxx %= a_x, buffer = AAt[xxx * a_x + uuuu];
            for (int j = 0; j < a_x; j++)
            {
                AAt[xxx * a_x + j] -= AAt[uuuu * a_x + j] * buffer;
                eye[xxx * a_x + j] -= eye[uuuu * a_x + j] * buffer;
            }
        }
    }
    //// DATA
    //for (int i = 0; i < a_x; ++i) {
    //	for (int j = 0; j < a_x; ++j) cout << eye[i*3+j] << " "; cout << "\n";
    //}

    /* inv( At A ) At */
    for (int i = 0; i < a_x; i++)
        for (int j = 0; j < a_y; j++)
            for (int k = 0; k < a_x; k++)
                pinv[i * a_y + j] += eye[i * a_x + k] * MAT_A[j * a_x + k];
    //// DATA
    //for (int i = 0; i < a_x; ++i) {
    //	for (int j = 0; j < a_y; ++j) cout << pinv[i*a_y+j] << " "; cout << "\n";
    //}

    /* ( inv( At A ) At ) B */
    for (int j = 0; j < a_x; j++)
        for (int k = 0; k < a_y; k++)
            MAT_X[j] += pinv[j * a_y + k] * MAT_B[k];
    //// DATA
    //for (int j = 0; j < a_x; ++j) cout << MAT_X[j] << " "; cout << "\n";

    free(eye), free(AAt), free(pinv);
}

int is_triangle(float *xyz1, float *xyz2, float *xyz3)
{

    if ((xyz1[0] * xyz1[1] * xyz1[2]) == 0)
        return 0;
    if ((xyz2[0] * xyz2[1] * xyz2[2]) == 0)
        return 0;
    if ((xyz3[0] * xyz3[1] * xyz3[2]) == 0)
        return 0;

    float d1 = 10.0, d2 = 0.1;

    float l[2], m[2], n[2];

    l[0] = xyz1[0] - xyz2[0], l[1] = xyz1[1] - xyz2[1];
    m[0] = xyz2[0] - xyz3[0], m[1] = xyz2[1] - xyz3[1];
    n[0] = xyz3[0] - xyz1[0], n[1] = xyz3[1] - xyz1[1];

    float a = (l[0] * l[0]) + (l[1] * l[1]);
    float b = (n[0] * n[0]) + (n[1] * n[1]);
    float c = (m[0] * m[0]) + (m[1] * m[1]);

    if ((d1 > (a / b) > d2) == 0)
        return 0;
    if ((d1 > (b / c) > d2) == 0)
        return 0;
    if ((d1 > (c / a) > d2) == 0)
        return 0;

    return 1;
}

//######################################################
//# filter
//######################################################

float dfdfdfqqq = 0.01;

void right_point(Mat &image, float *weight_x, float *weight_y)
{

    float *aaaaa_x = (float *)malloc(4 * 1024 * 1024 * 3);
    float *aaaaa_y = (float *)malloc(4 * 1024 * 1024 * 3);
    float *bbbbb_x = (float *)malloc(4 * 1024 * 1024);
    float *bbbbb_y = (float *)malloc(4 * 1024 * 1024);

    int point_num_x = 0;
    int point_num_y = 0;

    float mean;

    float *ptr_pos1 = (float *)image.data;

    for (int a = 0; a < 1024; a++)
    {
        for (int b = 0; b < 1024; b++)
        {
            if (ptr_pos1[(a * 1024 + b) * 3] * ptr_pos1[((a * 1024 + b) * 3) + 1])
            {
                mean = ptr_pos1[(a * 1024 + b) * 3] - ((b * weight_x[0] + a * weight_x[1]) + weight_x[2]);
                if ((dfdfdfqqq > mean) || (mean > -dfdfdfqqq))
                {
                    aaaaa_x[point_num_x * 3] = (float)b;
                    aaaaa_x[point_num_x * 3 + 1] = (float)a, aaaaa_x[point_num_x * 3 + 2] = (float)1;
                    bbbbb_x[point_num_x] = ptr_pos1[(a * 1024 + b) * 3];
                    point_num_x++;
                }
                mean = ptr_pos1[(a * 1024 + b) * 3] - ((b * weight_y[0] + a * weight_y[1]) + weight_y[2]);
                if ((dfdfdfqqq > mean) || (mean > -dfdfdfqqq))
                {
                    aaaaa_y[point_num_y * 3] = (float)b;
                    aaaaa_y[point_num_y * 3 + 1] = (float)a, aaaaa_y[point_num_y * 3 + 2] = (float)1;
                    bbbbb_y[point_num_y] = ptr_pos1[(a * 1024 + b) * 3 + 1];
                    point_num_y++;
                }
            }
        }
    }

    printf("%d\n", point_num_x);

    float qqqq_x[3], qqqq_y[3];

    LSM(aaaaa_x, point_num_x, 3, bbbbb_x, &qqqq_x[0]);
    LSM(aaaaa_y, point_num_y, 3, bbbbb_y, &qqqq_y[0]);

    for (int a = 0; a < 1024; a++)
    {
        for (int b = 0; b < 1024; b++)
        {
            ptr_pos1[((a * 1024 + b) * 3)] = (qqqq_x[0] * b) + (qqqq_x[1] * a) + qqqq_x[2];
            ptr_pos1[((a * 1024 + b) * 3) + 1] = (qqqq_y[0] * b) + (qqqq_y[1] * a) + qqqq_y[2];
        }
    }

    printf("==> %f,%f,%f \n", qqqq_x[0], qqqq_x[1], qqqq_x[2]);
    printf("==> %f,%f,%f \n", qqqq_y[0], qqqq_y[1], qqqq_y[2]);

    free(aaaaa_x), free(bbbbb_x), free(aaaaa_y), free(bbbbb_y);
}

void ransac(Mat &aafsa)
{

#define sample_num 64

    // 시드값을 얻기 위한 random_device 생성.
    std::random_device rd;

    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis_x(0, 1023);
    std::uniform_int_distribution<int> dis_y(0, 1023);

    int aaaeqewr = 0;
    int counter_random = 0;

    int pos_XY_buffer[3][2];
    float pos_XY[sample_num][3][3];
    float pos_Z_X[sample_num][3];
    float pos_Z_Y[sample_num][3];

    // 무작위로 삼각형들을 뽑아냄
    while (1)
    {

        pos_XY_buffer[0][0] = dis_x(gen), pos_XY_buffer[0][1] = dis_y(gen);
        pos_XY_buffer[1][0] = dis_x(gen), pos_XY_buffer[1][1] = dis_y(gen);
        pos_XY_buffer[2][0] = dis_x(gen), pos_XY_buffer[2][1] = dis_y(gen);

        float *ptr_pos1 = (float *)aafsa.data;
        float *ptr_pos2 = (float *)aafsa.data;
        float *ptr_pos3 = (float *)aafsa.data;

        ptr_pos1 = &ptr_pos1[(((pos_XY_buffer[0][1] * 1024) + pos_XY_buffer[0][0]) * 3)];
        ptr_pos2 = &ptr_pos2[(((pos_XY_buffer[1][1] * 1024) + pos_XY_buffer[1][0]) * 3)];
        ptr_pos3 = &ptr_pos3[(((pos_XY_buffer[2][1] * 1024) + pos_XY_buffer[2][0]) * 3)];

        if (is_triangle(ptr_pos1, ptr_pos2, ptr_pos3))
        {

            pos_XY[aaaeqewr][0][0] = pos_XY_buffer[0][0],
            pos_XY[aaaeqewr][0][1] = pos_XY_buffer[0][1], pos_XY[aaaeqewr][0][2] = 1,

            pos_XY[aaaeqewr][1][0] = pos_XY_buffer[1][0],
            pos_XY[aaaeqewr][1][1] = pos_XY_buffer[1][1], pos_XY[aaaeqewr][1][2] = 1,

            pos_XY[aaaeqewr][2][0] = pos_XY_buffer[2][0],
            pos_XY[aaaeqewr][2][1] = pos_XY_buffer[2][1], pos_XY[aaaeqewr][2][2] = 1,

            ptr_pos1 = (float *)aafsa.data;
            ptr_pos2 = (float *)aafsa.data;
            ptr_pos3 = (float *)aafsa.data;

            pos_Z_X[aaaeqewr][0] = ptr_pos1[(((pos_XY_buffer[0][1] * 1024) + pos_XY_buffer[0][0]) * 3)];
            pos_Z_X[aaaeqewr][1] = ptr_pos2[(((pos_XY_buffer[1][1] * 1024) + pos_XY_buffer[1][0]) * 3)];
            pos_Z_X[aaaeqewr][2] = ptr_pos3[(((pos_XY_buffer[2][1] * 1024) + pos_XY_buffer[2][0]) * 3)];

            pos_Z_Y[aaaeqewr][0] = ptr_pos1[(((pos_XY_buffer[0][1] * 1024) + pos_XY_buffer[0][0]) * 3) + 1];
            pos_Z_Y[aaaeqewr][1] = ptr_pos2[(((pos_XY_buffer[1][1] * 1024) + pos_XY_buffer[1][0]) * 3) + 1];
            pos_Z_Y[aaaeqewr][2] = ptr_pos3[(((pos_XY_buffer[2][1] * 1024) + pos_XY_buffer[2][0]) * 3) + 1];

            aaaeqewr++;
        }

        if (aaaeqewr >= sample_num)
            break;
    }

    int counter_correct_point_most_x = 0;
    int counter_correct_point_most_y = 0;
    int counter_correct_point_x = 0;
    int counter_correct_point_y = 0;
    int counter_ssss_x = 0;
    int counter_ssss_y = 0;

    float SLMA_X[3], SLMA_prev_X[3];
    float SLMA_Y[3], SLMA_prev_Y[3];

    for (int qq = 0; qq < sample_num; qq++)
    {

        LSM(&pos_XY[qq][0][0], 3, 3, &pos_Z_X[qq][0], &SLMA_X[0]);
        LSM(&pos_XY[qq][0][0], 3, 3, &pos_Z_Y[qq][0], &SLMA_Y[0]);

        // 받은 이미지의 X 값을 훑으면서 일정범위 사이에 점이 있으면 카운트를 셈
        counter_correct_point_x = 0, counter_ssss_x = 0;
        counter_correct_point_y = 0, counter_ssss_y = 0;
        float *ptr_pos1 = (float *)aafsa.data;
        for (int a = 0; a < 1024; a++)
        {
            for (int b = 0; b < 1024; b++)
            {
                if (ptr_pos1[(a * 1024 + b) * 3] * ptr_pos1[(a * 1024 + b) * 3 + 1])
                {
                    float mean_x = ptr_pos1[(a * 1024 + b) * 3] - ((b * SLMA_X[0] + a * SLMA_X[1]) + SLMA_X[2]);
                    float mean_y = ptr_pos1[(a * 1024 + b) * 3 + 1] - ((b * SLMA_Y[0] + a * SLMA_Y[1]) + SLMA_Y[2]);
                    if ((dfdfdfqqq > mean_x) && (mean_x > -dfdfdfqqq))
                        counter_correct_point_x++;
                    if ((dfdfdfqqq > mean_y) && (mean_y > -dfdfdfqqq))
                        counter_correct_point_y++;
                }
                counter_ssss_x++, counter_ssss_y++;
            }
        }

        // 카운트가 지금까지 세었던 것 중에 가장 높으면 업데이트함
        if (counter_correct_point_x > counter_correct_point_most_x)
            counter_correct_point_most_x = counter_correct_point_x;
        SLMA_prev_X[0] = SLMA_X[0], SLMA_prev_X[1] = SLMA_X[1], SLMA_prev_X[2] = SLMA_X[2];
        if (counter_correct_point_y > counter_correct_point_most_y)
            counter_correct_point_most_y = counter_correct_point_y;
        SLMA_prev_Y[0] = SLMA_Y[0], SLMA_prev_Y[1] = SLMA_Y[1], SLMA_prev_Y[2] = SLMA_Y[2];

        printf(
            "%d %d ==> %f,%f,%f \n",
            counter_correct_point_x, qq,
            SLMA_X[0], SLMA_X[1], SLMA_X[2]);
    }

    right_point(aafsa, SLMA_prev_X, SLMA_prev_Y);
}

//######################################################
//#
//######################################################

void spliter(Mat &AAAA)
{
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
void capture(VideoCapture &A, Mat &B)
{
    A.grab(), A.grab(), A.grab(), A.grab(), A.grab(), A.grab(), A.retrieve(B); //A >> B;
    spliter(B);
}
void camera2(Mat &frame)
{
    gui_graycode_show(frame);
}

Mat buffersasasa(h, w, CV_16SC1);

int image_type = 0;  // background--graycode
int image_inv = 0;   // graycode
int image_dir = 0;   // graycode
int image_stage = 0; // graycode
int image_onoff = 0; // background
int ready_flag = 0;
void capture_algorithm(VideoCapture &cap)
{
    static Mat projector;
    //######################################################
    //#
    //######################################################
    if (ready_flag)
    {
        if (image_type == 1)
        {
            projector = projector_map_LAMP(1024, 1024, image_onoff);
            image_onoff = 0;
            camera2(projector), usleep(600000); //waitKey(adfadf);
            capture(cap, buffersasasa);
        }
        else if (image_type == 2)
        {
            projector = projector_map_GRAY(
                1024, 1024, image_stage, image_inv, image_dir);
            image_stage = 0, image_inv = 0, image_dir = 0;
            camera2(projector), usleep(600000); //waitKey(adfadf);
            capture(cap, buffersasasa);
        }
        else if (image_type == 3)
            capture(cap, buffersasasa);
        else if (image_type == 4)
            capture(cap, buffersasasa);
        else if (image_type == -1)
        {
            printf("camera setup \n");
        }
        image_type = 0, ready_flag = 0;
    }
}
void capture_background(Mat &aafsa, int color)
{
    image_onoff = color;
    image_type = 1, ready_flag = 1;
    while (ready_flag)
        usleep(10000);
    aafsa = buffersasasa;
}
void capture_graycode(Mat &aafsa, int stage, int invert, int dir)
{
    image_inv = invert, image_dir = dir, image_stage = stage;
    image_type = 2, ready_flag = 1;
    while (ready_flag)
        usleep(10000);
    aafsa = buffersasasa;
}
void capture_plane(Mat &aafsa)
{
    image_type = 3, ready_flag = 1;
    while (ready_flag)
        usleep(10000);
    aafsa = buffersasasa;
}
void capture_viewer(Mat &aafsa)
{
    image_type = 4, ready_flag = 1;
    while (ready_flag)
        usleep(10000);
    aafsa = buffersasasa;
    //aafsa.convertTo(aafsa, CV_8UC3);
}
void capture_setting(int a, int b)
{
    image_type = -1, ready_flag = 1;
    while (ready_flag)
        usleep(10000);
    printf("finish\n");
}

//######################################################
//# 파일 입출력
//######################################################

void aaaaaaa(Mat &qwerqer, int h, int w)
{

    //
    char a0[] = "ply\n";
    char a1[] = "format binary_little_endian 1.0\n";
    char a2[] = "element vertex 1000000         \n";
    char a3[] = "property float x\n";
    char a4[] = "property float y\n";
    char a5[] = "property float z\n";
    char a6[] = "end_header\n";

    //
    FILE *src;
    size_t nRead;

    // 헤더
    src = fopen("32323.ply", "wb");

    if (src != NULL)
    {
        printf("file open OK \n");
        fwrite(a0, 1, 4, src), fwrite(a1, 1, 32, src), fwrite(a2, 1, 32, src), fwrite(a3, 1, 17, src);
        fwrite(a4, 1, 17, src), fwrite(a5, 1, 17, src), fwrite(a6, 1, 11, src);
    }

    // 데이터
    float *qwerqerqwerqwer = (float *)qwerqer.data;

    float buffer[3072] = {0};
    float buffer_0 = 0;
    float buffer_1 = 0;
    float buffer_2 = 0;

    uint64_t count = 0;
    uint64_t count_buffer = 0;

    for (uint32_t y = 0; y < h; y++)
    {
        for (uint32_t x = 0; x < w; x++)
        {
            buffer_0 = qwerqerqwerqwer[count];
            buffer_1 = qwerqerqwerqwer[count + 1];
            buffer_2 = qwerqerqwerqwer[count + 2];
            count = count + 3;
            //if (buffer_0 * buffer_1 * buffer_2) {
            buffer[count_buffer] = buffer_0 / 100;
            buffer[count_buffer + 1] = buffer_1 / 100;
            buffer[count_buffer + 2] = buffer_2 / 100;
            count_buffer = count_buffer + 3;
            if (count_buffer > 3072)
                fwrite(buffer, sizeof(float), 3072, src), count_buffer = 0;
            //}
        }
    }

    if (count_buffer != 0)
        fwrite(buffer, sizeof(float), count_buffer, src), count_buffer = 0;

    fclose(src);
}
void aaaaaaa0(Mat &qwerqer, int h, int w)
{

    //
    char a0[] = "ply\n";
    char a1[] = "format binary_little_endian 1.0\n";
    char a2[] = "element vertex 1000000         \n";
    char a3[] = "property float x\n";
    char a4[] = "property float y\n";
    char a5[] = "property float z\n";
    char a6[] = "end_header\n";

    //
    FILE *src;
    size_t nRead;

    // 헤더
    src = fopen("323230.ply", "wb");

    if (src != NULL)
    {
        printf("file open OK \n");
        fwrite(a0, 1, 4, src), fwrite(a1, 1, 32, src), fwrite(a2, 1, 32, src), fwrite(a3, 1, 17, src);
        fwrite(a4, 1, 17, src), fwrite(a5, 1, 17, src), fwrite(a6, 1, 11, src);
    }

    // 데이터
    float buffer[3072] = {0};
    float buffer_0 = 0;
    float buffer_1 = 0;
    float buffer_2 = 0;

    uint64_t count = 0;
    uint64_t count_buffer = 0;

    float *qwerqerqwerqwer = (float *)qwerqer.data;

    for (uint32_t y = 0; y < h; y++)
    {
        for (uint32_t x = 0; x < w; x++)
        {
            buffer_0 = qwerqerqwerqwer[count];
            buffer_1 = qwerqerqwerqwer[count + 1];
            buffer_2 = qwerqerqwerqwer[count + 2];
            count = count + 3;
            //if ( buffer_0*buffer_1*buffer_2 ) {
            buffer[count_buffer] = (float)x / 50;
            buffer[count_buffer + 1] = (float)y / 50;
            buffer[count_buffer + 2] = buffer_0 / 100;
            count_buffer = count_buffer + 3;
            //}
            if (count_buffer > 3072)
                fwrite(buffer, sizeof(float), 3072, src), count_buffer = 0;
        }
    }

    if (count_buffer != 0)
        fwrite(buffer, sizeof(float), count_buffer, src), count_buffer = 0;

    fclose(src);
}
void aaaaaaa1(Mat &qwerqer, int h, int w)
{

    //
    char a0[] = "ply\n";
    char a1[] = "format binary_little_endian 1.0\n";
    char a2[] = "element vertex 1000000         \n";
    char a3[] = "property float x\n";
    char a4[] = "property float y\n";
    char a5[] = "property float z\n";
    char a6[] = "end_header\n";

    //
    FILE *src;
    size_t nRead;

    // 헤더
    src = fopen("323231.ply", "wb");

    if (src != NULL)
    {
        printf("file open OK \n");
        fwrite(a0, 1, 4, src), fwrite(a1, 1, 32, src), fwrite(a2, 1, 32, src), fwrite(a3, 1, 17, src);
        fwrite(a4, 1, 17, src), fwrite(a5, 1, 17, src), fwrite(a6, 1, 11, src);
    }

    // 데이터
    float buffer[3072] = {0};
    float buffer_0 = 0;
    float buffer_1 = 0;
    float buffer_2 = 0;

    uint64_t count = 0;
    uint64_t count_buffer = 0;

    float *qwerqerqwerqwer = (float *)qwerqer.data;

    for (uint32_t y = 0; y < h; y++)
    {
        for (uint32_t x = 0; x < w; x++)
        {
            buffer_0 = qwerqerqwerqwer[count];
            buffer_1 = qwerqerqwerqwer[count + 1];
            buffer_2 = qwerqerqwerqwer[count + 2];
            count = count + 3;
            //if ( buffer_0*buffer_1*buffer_2 ) {
            buffer[count_buffer] = (float)x / 50;
            buffer[count_buffer + 1] = (float)y / 50;
            buffer[count_buffer + 2] = buffer_1 / 100;
            count_buffer = count_buffer + 3;
            //}
            if (count_buffer > 3072)
                fwrite(buffer, sizeof(float), 3072, src), count_buffer = 0;
        }
    }

    if (count_buffer != 0)
        fwrite(buffer, sizeof(float), count_buffer, src), count_buffer = 0;

    fclose(src);
}

//######################################################
//#
//######################################################

int flag_process_run = 0, shutdown = 0;

GtkWidget *popwindow;

GtkWidget *drawing_area_graycode, *drawing_area_preview;
GdkPixbuf *pixbuf_graycode, *pixbuf_preview;
guchar *pixels_graycode, *pixels_preview;

void graycode_window_on()
{
    gtk_widget_show(popwindow);
}
void graycode_window_off()
{
    gtk_widget_hide(popwindow);
}

static void close_button()
{
    shutdown = 1;
}
static void move_button1(GtkWidget *button)
{
    flag_process_run = 1;
}
static void move_button2(GtkWidget *button)
{
    flag_process_run = 2;
}
static void move_button3(GtkWidget *button)
{
    flag_process_run = 3;
}
static void move_button4(GtkWidget *button)
{
    flag_process_run = 4;
}

static void slider_camera_control_black(GtkRange *range, gpointer win)
{
    thresholder_black = gtk_range_get_value(range);
    //printf("setting runs %d\n", thresholder_black);
}
static void slider_camera_control_white(GtkRange *range, gpointer win)
{
    thresholder_white = gtk_range_get_value(range);
    //printf("setting runs %d\n", thresholder_white);
}

int v4l_auto_exposure = 1;
int v4l_exposure_time_absolute = 500;
int v4l_contrast = 0;
int v4l_brightness = 50;
static void v4l_apply(GtkWidget *button)
{
    int ret;
    ret = system("v4l2-ctl --set-ctrl=auto_exposure=1");
    ret = system("v4l2-ctl --set-ctrl=exposure_time_absolute=1000");
    ret = system("v4l2-ctl --set-ctrl=contrast=0");
    ret = system("v4l2-ctl --set-ctrl=brightness=50");

    printf("v4l_apply\n");
}
static void v4l_set_exposure_time(GtkRange *range, gpointer win)
{
    v4l_exposure_time_absolute = gtk_range_get_value(range);
    printf("v4l_set_exposure_time\n");
}
static void v4l_set_brightness(GtkRange *range, gpointer win)
{
    v4l_brightness = gtk_range_get_value(range);
    printf("v4l_set_brightness\n");
}
static void v4l_set_contrast(GtkRange *range, gpointer win)
{
    v4l_contrast = gtk_range_get_value(range);
    printf("v4l_set_contrast\n");
}

void gui_preview_update(Mat &qwerqer)
{

    //printf("n_channels      : %d\n", gdk_pixbuf_get_n_channels(pixbuf)  );
    //printf("has_alpha       : %d\n", gdk_pixbuf_get_has_alpha(pixbuf)  );
    //printf("bits_per_sample : %d\n", gdk_pixbuf_get_bits_per_sample(pixbuf)  );
    //printf("pixels          : %d\n", gdk_pixbuf_get_pixels(pixbuf)  );
    //printf("%d", gdk_pixbuf_get_pixels_with_length(pixbuf)  );
    //printf("get_width       : %d\n", gdk_pixbuf_get_width(pixbuf)  );
    //printf("get_height      : %d\n", gdk_pixbuf_get_height(pixbuf)  );

    uint8_t *adfdf = (uint8_t *)qwerqer.data;

    for (int y = 0; y < 480; y++)
    {
        for (int x = 0; x < 640; x++)
        {
            int www = (y * 640 + x), qqq = www * 3;
            pixels_preview[qqq] = adfdf[www];
            pixels_preview[qqq + 1] = adfdf[www];
            pixels_preview[qqq + 2] = adfdf[www];
        }
    }

    for (int x = 0; x < 640; x++)
    {
        int qwerew, oooo, rtrtrt, rertete;

        rtrtrt = 120;
        qwerew = rtrtrt;
        rertete = pixels_preview[(rtrtrt * 640 + x) * 3];
        oooo = rtrtrt - (rertete >> 2) + 32;
        pixels_preview[(oooo * 640 + x) * 3] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 1] = 0;
        pixels_preview[(oooo * 640 + x) * 3 + 2] = 0;
        qwerew = (rtrtrt * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 255;
        pixels_preview[qwerew + 2] = 0;
        qwerew = ((rtrtrt + 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;
        qwerew = ((rtrtrt - 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;
        oooo = rtrtrt - (rertete >> 2) + 32 - (thresholder_black >> 2);
        pixels_preview[(oooo * 640 + x) * 3] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 1] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 2] = 0;
        oooo = rtrtrt - (rertete >> 2) + 32 + (thresholder_white >> 2);
        pixels_preview[(oooo * 640 + x) * 3] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 1] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 2] = 0;

        rtrtrt = 240;
        qwerew = rtrtrt;
        oooo = rtrtrt - (pixels_preview[(rtrtrt * 640 + x) * 3] >> 2) + 32;
        pixels_preview[(oooo * 640 + x) * 3] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 1] = 0;
        pixels_preview[(oooo * 640 + x) * 3 + 2] = 0;
        qwerew = (rtrtrt * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 255;
        pixels_preview[qwerew + 2] = 0;
        qwerew = ((rtrtrt + 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;
        qwerew = ((rtrtrt - 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;

        rtrtrt = 360;
        qwerew = rtrtrt;
        oooo = rtrtrt - (pixels_preview[(rtrtrt * 640 + x) * 3] >> 2) + 32;
        pixels_preview[(oooo * 640 + x) * 3] = 255;
        pixels_preview[(oooo * 640 + x) * 3 + 1] = 0;
        pixels_preview[(oooo * 640 + x) * 3 + 2] = 0;
        qwerew = (rtrtrt * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 255;
        pixels_preview[qwerew + 2] = 0;
        qwerew = ((rtrtrt + 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;
        qwerew = ((rtrtrt - 32) * 640 + x) * 3;
        pixels_preview[qwerew] = 0;
        pixels_preview[qwerew + 1] = 0;
        pixels_preview[qwerew + 2] = 255;
    }
    gdk_threads_enter();
    gtk_widget_queue_draw(drawing_area_preview);
    gdk_threads_leave();

    //printf("gui_preview_update image update \n");

}
void gui_graycode_show(Mat &qwerqer)
{

    uint8_t *adfdf = (uint8_t *)qwerqer.data;

    for (int y = 0; y < 1000; y++)
        for (int x = 0; x < 1000; x++)
            pixels_graycode[(y*1000+x)*3  ] = adfdf[y*1024+x],
            pixels_graycode[(y*1000+x)*3+1] = adfdf[y*1024+x],
            pixels_graycode[(y*1000+x)*3+2] = adfdf[y*1024+x];

    gdk_threads_enter();
    gtk_widget_queue_draw(drawing_area_graycode);
    gdk_threads_leave();

    //printf("gui_graycode_show image update \n");

}
void gui_main()
{

    /* GtkWidget is the storage type for widgets */
    GtkWidget *window;
    
    GtkWidget *fixed___0;
    GtkWidget *fixed___1;
    
    GtkWidget *button;
    GtkWidget *button1;
    GtkWidget *button2;
    GtkWidget *button3;
    GtkWidget *button4;
    GtkWidget *button_close;
    
    /*  */
    pixbuf_preview = gdk_pixbuf_new(GDK_COLORSPACE_RGB, 0, 8, 640, 480);
    pixbuf_graycode = gdk_pixbuf_new(GDK_COLORSPACE_RGB, 0, 8, 1000, 1000);
    pixels_preview = gdk_pixbuf_get_pixels(pixbuf_preview);
    pixels_graycode = gdk_pixbuf_get_pixels(pixbuf_graycode);

    /*  */
    gdk_threads_init();

    /* Initialise GTK */
    gtk_init(0, NULL);//gtk_init (&argc, &argv);

    /* Create a new window */
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    popwindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Fixed Container");
    gtk_window_set_default_size(GTK_WINDOW(window), 640 + 320 + 160, 480 + 240 + 120);
    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    //gtk_window_set_default_size(GTK_WINDOW(popwindow), 640 + 320 + 160, 480 + 240 + 120);
    gtk_window_fullscreen(GTK_WINDOW(popwindow));
    //gtk_window_set_keep_above(GTK_WINDOW(window), TRUE);
    //gtk_window_fullscreen(GTK_WINDOW(window));
    //gtk_window_move(GTK_WINDOW(window), 1, -11);
    //gtk_window_set_default_size(GTK_WINDOW(popwindow), 640, 480);
    //gtk_window_set_keep_above(GTK_WINDOW(popwindow), TRUE);
    //gtk_window_set_modal(GTK_WINDOW(popwindow), TRUE);
    //gtk_window_set_decorated(GTK_WINDOW(popwindow), FALSE);
    //gtk_window_set_resizable(GTK_WINDOW(popwindow), FALSE);
    //gtk_window_set_position(GTK_WINDOW(popwindow), (GtkWindowPosition)GTK_WIN_POS_CENTER);
    //gtk_window_set_transient_for(GTK_WINDOW(popwindow), GTK_WINDOW(window));

    /* Here we connect the "destroy" event to a signal handler */
    g_signal_connect(window, "destroy", G_CALLBACK(close_button), NULL);
    gtk_container_set_border_width(GTK_CONTAINER(window), 3);
    gtk_widget_show(window);
    //gtk_widget_show(popwindow);

    /* Create a Fixed Container */
    fixed___0 = gtk_fixed_new();
    fixed___1 = gtk_fixed_new();
    gtk_container_add(GTK_CONTAINER(window), fixed___0);
    gtk_container_add(GTK_CONTAINER(popwindow), fixed___1);
    gtk_widget_show(fixed___0);
    gtk_widget_show(fixed___1);

    /*  */
    button1 = gtk_button_new_with_label("projector side");
    gtk_fixed_put(GTK_FIXED(fixed___0), button1, 10, 10);
    gtk_widget_show(button1);
    
    button2 = gtk_button_new_with_label("oboject side");
    gtk_fixed_put(GTK_FIXED(fixed___0), button2, 10, 60);
    gtk_widget_show(button2);
    
    button3 = gtk_button_new_with_label("scan");
    gtk_fixed_put(GTK_FIXED(fixed___0), button3, 10, 110);
    gtk_widget_show(button3);
    
    button4 = gtk_button_new_with_label("data save");
    gtk_fixed_put(GTK_FIXED(fixed___0), button4, 10, 160);
    gtk_widget_show(button4);

    button_close = gtk_button_new_with_label("closer");
    gtk_fixed_put(GTK_FIXED(fixed___0), button_close, 10, 210);
    gtk_widget_show(button_close);
    
    drawing_area_preview = gtk_image_new_from_pixbuf(pixbuf_preview);
    gtk_fixed_put(GTK_FIXED(fixed___0), drawing_area_preview, 130, 10);
    gtk_widget_show(drawing_area_preview);

    // 카메라 제어
    GtkWidget *slider_parameter_black;
    slider_parameter_black = gtk_hscale_new_with_range ( 0 , 300 , 10 );
    gtk_widget_set_size_request(slider_parameter_black, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), slider_parameter_black, 0, 500);
    gtk_widget_show(slider_parameter_black);
    GtkWidget *slider_parameter_white;
    slider_parameter_white = gtk_hscale_new_with_range ( 0 , 300 , 10 );
    gtk_widget_set_size_request(slider_parameter_white, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), slider_parameter_white, 0, 570);
    gtk_widget_show(slider_parameter_white);

    GtkWidget *v4l_setting_apply;
    v4l_setting_apply = gtk_button_new_with_label("apply");
    gtk_widget_set_size_request(slider_parameter_black, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), v4l_setting_apply, 10, 260);
    gtk_widget_show(v4l_setting_apply);

    //value, lower, upper, step_increment, page_increment, page_size
    GtkWidget *v4l_setting_exp_time_abs;
    GtkObject *v4l_setting_exp_time_abs_obj;
    v4l_setting_exp_time_abs_obj = gtk_adjustment_new (
        1000, 100, 10000, 10, 100, 4);
    v4l_setting_exp_time_abs = gtk_hscale_new ( 
        GTK_ADJUSTMENT (v4l_setting_exp_time_abs_obj) );
    gtk_widget_set_size_request(v4l_setting_exp_time_abs, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), v4l_setting_exp_time_abs, 100, 570);
    gtk_widget_show(v4l_setting_exp_time_abs);

    GtkWidget *v4l_setting_contrast;
    GtkObject *v4l_setting_contrast_obj;
    v4l_setting_contrast_obj = gtk_adjustment_new (
        0, -100, 100, 1, 3, 10);
    v4l_setting_contrast = gtk_hscale_new ( 
        GTK_ADJUSTMENT (v4l_setting_contrast_obj) );
    gtk_widget_set_size_request(v4l_setting_contrast, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), v4l_setting_contrast, 250, 570);
    gtk_widget_show(v4l_setting_contrast);

    GtkWidget *v4l_setting_brightness;
    GtkObject *v4l_setting_brightness_obj;
    v4l_setting_brightness_obj = gtk_adjustment_new (
        50, 0, 100, 1, 30, 2);
    v4l_setting_brightness = gtk_hscale_new ( 
        GTK_ADJUSTMENT (v4l_setting_brightness_obj) );
    gtk_widget_set_size_request(v4l_setting_brightness, 100, 60);
    gtk_fixed_put(GTK_FIXED(fixed___0), v4l_setting_brightness, 400, 570);
    gtk_widget_show(v4l_setting_brightness);

    // 프리뷰
    drawing_area_graycode = gtk_image_new_from_pixbuf(pixbuf_graycode);
    gtk_fixed_put(GTK_FIXED(fixed___1), drawing_area_graycode, 0, 0);
    gtk_widget_show(drawing_area_graycode);

    /* connect button signal */
    g_signal_connect(button1, "clicked", G_CALLBACK(move_button1), NULL);
    g_signal_connect(button2, "clicked", G_CALLBACK(move_button2), NULL);
    g_signal_connect(button3, "clicked", G_CALLBACK(move_button3), NULL);
    g_signal_connect(button4, "clicked", G_CALLBACK(move_button4), NULL);
    g_signal_connect(button_close, "clicked", G_CALLBACK(close_button), NULL);
    
    g_signal_connect(
        slider_parameter_black, "value-changed", 
        G_CALLBACK(slider_camera_control_black), NULL);
    g_signal_connect(
        slider_parameter_white, "value-changed", 
        G_CALLBACK(slider_camera_control_white), NULL);

    g_signal_connect(v4l_setting_apply, "clicked", G_CALLBACK(v4l_apply), NULL);

    //g_signal_connect(
    //    button2, "clicked", G_CALLBACK(slider_camera_control), NULL);

    /* Enter the event loop */
    gtk_main();

}

//######################################################
//#
//######################################################

Mat AAA, BBB, CCC;

void graycode_map(Mat &aafsa, int scann_calib_switch = 0)
{

    float plane_vectors[] = {0, 0, 1.0, 0, 0, 600.0};

    if (scann_calib_switch)
    {
        //capture_plane( aafsa );
        //image_to_plane( aafsa, plane_vectors );
        //printf("%f, %f, %f,    %f, %f, %f \n\n",
        //    plane_vectors[0], plane_vectors[1], plane_vectors[2],
        //    plane_vectors[3], plane_vectors[4], plane_vectors[5]
        //);
    }

    Mat black(h, w, CV_16SC1);
    Mat white(h, w, CV_16SC1);
    Mat buffer(h, w, CV_16SC1);
    Mat buffer_inv(h, w, CV_16SC1);
    Mat bit_h(h, w, CV_16UC1);
    Mat bit_v(h, w, CV_16UC1);
    Mat bit_buffer(h, w, CV_16UC1);

    Mat bit_hvsdas(1024, 1024, CV_32FC3);

    black = 0, white = 0;
    buffer = 0, buffer_inv = 0;
    bit_h = 0, bit_v = 0;
    bit_buffer = 0, bit_hvsdas = 0;

    capture_background(black, 0);
    capture_background(white, 1);

    black.convertTo(black, CV_16SC1);
    white.convertTo(white, CV_16SC1);

    black += thresholder_black;
    white -= thresholder_white;

    for (int stage = 0; stage < 10; stage++)
    {
        /* graycode_h */
        capture_graycode(buffer, stage, 0, 'h');
        capture_graycode(buffer_inv, stage, 1, 'h');
        buffer.convertTo(buffer, CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);
        bit_buffer = image_to_bit(black, white, buffer, buffer_inv);
        bit_buffer.convertTo(bit_buffer, CV_16UC1);
        image_to_bit_stacking(bit_h, bit_buffer, stage);
        printf("%d\n", stage);
    }

    for (int stage = 0; stage < 10; stage++)
    {
        /* graycode_v */
        capture_graycode(buffer, stage, 0, 'v');
        capture_graycode(buffer_inv, stage, 1, 'v');
        buffer.convertTo(buffer, CV_16SC1);
        buffer_inv.convertTo(buffer_inv, CV_16SC1);
        bit_buffer = image_to_bit(black, white, buffer, buffer_inv);
        bit_buffer.convertTo(bit_buffer, CV_16UC1);
        image_to_bit_stacking(bit_v, bit_buffer, stage);
        printf("%d\n", stage);
    }
    printf("image_to_bit_stackinged\n");

    //######################################################
    //#
    //######################################################

    bit_to_cordination(bit_h, bit_v, bit_hvsdas, 600, h, w, 1024, 1024);
    printf("bit_to_cordination\n");

    //Mat ssbit_h = bit_h*(1<<6);
    //camera2(ssbit_h);
    //waitKey(6000);
    //Mat ssbit_v = bit_v*(1<<6);
    //camera2(ssbit_v);
    //waitKey(6000);
    //bit_hasdas = bit_hasdas*100;
    //camera2(bit_hasdas);
    //waitKey(6000);
    //bit_vasdas = bit_vasdas*100;
    //camera2(bit_vasdas);
    //waitKey(6000);
    //adsadasdas = bit_hvsdas/600;
    //camera2(adsadasdas);
    //waitKey(6000);

    aafsa = bit_hvsdas;

    if (scann_calib_switch)
    {
        //ransac(aafsa);
        //cordination_to_point(  aafsa,  plane_vectors,  1024,1024  );
    }

    black.release(), white.release();
    buffer.release(), buffer_inv.release();
    bit_h.release(), bit_v.release();
    bit_buffer.release();
    bit_hvsdas.release();


}

void triangulation(Mat &_p1, Mat &_p2, Mat &postion, float focus, int h, int w)
{
    int a = 0;
    float *image_map = (float *)postion.data;
    float *map = (float *)postion.data;
    float *p1 = (float *)_p1.data;
    float *p2 = (float *)_p2.data;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            //image_map[a] = 0, image_map[a+1] = 0, image_map[a+2] = 0;
            if (image_map[a] * image_map[a + 1] * image_map[a + 2])
            {
                // theta 구하기
                float theta = -(image_map[a] / focus);
                // rad 구하기
                float rad = atan(theta);
                // 회전변환
                float x = (1 * cos(rad)); //- (0*cos);
                float y = (1 * sin(rad)); //+ (0*sin);
                float w = sqrt(x * x) + sqrt(y * y);
                float nomal_vector_x = x / w;
                float nomal_vector_y = 0;
                float nomal_vector_z = y / w;
                // U = ( N*(TRANSLATION-P1) ) / ( N*(P2-P1) )
                float U1 = (nomal_vector_x * (-p1[a])) / (nomal_vector_x * (p2[a] - p1[a]));
                float U2 = (nomal_vector_y * (-p1[a + 1])) / (nomal_vector_y * (p2[a + 1] - p1[a + 1]));
                float U3 = (nomal_vector_z * (-p1[a + 2])) / (nomal_vector_z * (p2[a + 2] - p1[a + 2]));
                float U = U1 + U2 + U3;
                // P = P1 + U * P2
                map[a] = p1[a] + (U * (p2[a]));
                map[a + 1] = p1[a + 1] + (U * (p2[a + 1]));
                map[a + 2] = p1[a + 2] + (U * (p2[a + 2]));
            }
            a += 3;
        }
    }

    //    theta = -np.true_divide(  yx[0:,0:,0],  focus  )
    //    #print("theta map [yx => xyz]: ",yx.shape)
    //    rad = np.arctan(theta)
    //    #print("rad map [yx => xyz]: ",yx.shape)
    //    sin = np.sin(rad)
    //    cos = np.cos(rad)
    //    xxxxxxxx = (plane_map[0:,0:,0] * cos) - (plane_map[0:,0:,2] * sin)
    //    yyyyyyyy = (plane_map[0:,0:,0] * sin) + (plane_map[0:,0:,2] * cos)
    //    adadsdas = np.sqrt(xxxxxxxx*xxxxxxxx) + np.sqrt(yyyyyyyy*yyyyyyyy)

}

//void processing()
//{
//
//    //cv::Mat p1(  cv::Size(1024,  1024),  CV_32FC3,  Scalar(   0,   0,   0  )  );
//    //cv::Mat p2(  cv::Size(1024,  1024),  CV_32FC3,  Scalar(  10,  10,  10  )  );
//    //float plane_vectors[] = {    0, 0, 1.0,    0, 0, 600.0    };
//    //cordination_to_point(  gray,  plane_vectors,  1024,1024  );
//
//    Mat gray1;
//    printf("graycode_map start \n");
//
//    graycode_window_on();
//    graycode_map(gray1, 1);
//    graycode_window_off();
//    printf("cordination_to_point finish \n");
//
//    aaaaaaa(gray1, projector_map_shape_y, projector_map_shape_x);
//    aaaaaaa0(gray1, projector_map_shape_y, projector_map_shape_x);
//    aaaaaaa1(gray1, projector_map_shape_y, projector_map_shape_x);
//    printf("owari \n");
//
//}

void calib_projector_side()
{
    printf("calib_projector_side start \n");
    graycode_window_on();
    graycode_map(AAA, 1);
    graycode_window_off();
    printf("calib_projector_side owari \n");
}
void calib_object_side()
{
    printf("calib_object_side start \n");
    graycode_window_on();
    graycode_map(BBB, 1);
    graycode_window_off();
    printf("calib_object_side owari \n");
}
void scan_object()
{
    printf("scan_object start \n");
    graycode_map(CCC, 0);
    triangulation(AAA,BBB,CCC,600,1024,1024);
    printf("scan_object end \n");
}
void scan_data_save()
{
    printf("scan data save start \n");
    aaaaaaa(AAA, projector_map_shape_y, projector_map_shape_x);
    aaaaaaa0(AAA, projector_map_shape_y, projector_map_shape_x);
    aaaaaaa1(AAA, projector_map_shape_y, projector_map_shape_x);
    printf("scan data save end \n");
}

//######################################################
//#
//######################################################

int capture_algorithm_state = 0;

void thread_processing()
{
    capture_algorithm_state = 1;
    int counter = 0;
    sleep(3);
    while (1)
    {
        if (shutdown)
        {   
            capture_algorithm_state = 0;
            gtk_main_quit();
            break;
        }
        usleep(10000);
        if (flag_process_run)
        {
            if (flag_process_run == 1)
                calib_projector_side();
            else if (flag_process_run == 2)
                calib_object_side();
            else if (flag_process_run == 3)
                scan_object();
            else if (flag_process_run == 4)
                scan_data_save();
            flag_process_run = 0;
        }
        if (counter++ > 20)
        {
            static Mat retry;
            capture_viewer(retry), gui_preview_update(retry), counter = 0;
        }
        printf("%d\n", counter);
    }
}
void thread_camera()
{
    VideoCapture cap(0 );
    if (!cap.isOpened())
        printf("카메라를 열수 없습니다. \n");
    cap.set(CAP_PROP_FRAME_WIDTH, w);
    cap.set(CAP_PROP_FRAME_HEIGHT, h);
    cap.set(CAP_PROP_FPS, 13);
    //cap.set(CAP_PROP_EXPOSURE, 0.1 );
    //cap.set(CV_CAP_PROP_AUTO_EXPOSURE,0.25);
    //cap.set(CAP_PROP_XI_AUTO_WB,0);
    //cap.set(CAP_PROP_ISO_SPEED, 100);
    sleep(2);
    while (1)
    {
        if (shutdown&(~capture_algorithm_state))
        {
            break;
        }
        usleep(10000);
        capture_algorithm(cap);
    }
}
void thread_main()
{
    //######################################################
    //#
    //######################################################
    graycode_lookup_table(bit, Gray2Bin, Bin2Gray);
    //######################################################
    //#
    //######################################################
    while (1)
    {
        gui_main(), printf("gui_main finish\n");
        break;
    }
}

//######################################################
//#
//######################################################

int main(int, char **)
{
    thread hi0(thread_main);
    thread hi1(thread_camera);
    thread hi2(thread_processing);
    hi0.join();
    hi1.join();
    hi2.join();
    return 0;
}

/*
*
* User Controls
*                      brightness 0x00980900 (int)    : min=0 max=100 step=1 default=50 value=50 flags=slider
*                        contrast 0x00980901 (int)    : min=-100 max=100 step=1 default=0 value=0 flags=slider
*                      saturation 0x00980902 (int)    : min=-100 max=100 step=1 default=0 value=0 flags=slider
*                     red_balance 0x0098090e (int)    : min=1 max=7999 step=1 default=1000 value=1000 flags=slider
*                    blue_balance 0x0098090f (int)    : min=1 max=7999 step=1 default=1000 value=1000 flags=slider
*                 horizontal_flip 0x00980914 (bool)   : default=0 value=0
*                   vertical_flip 0x00980915 (bool)   : default=0 value=0
*            power_line_frequency 0x00980918 (menu)   : min=0 max=3 default=1 value=1
*                       sharpness 0x0098091b (int)    : min=-100 max=100 step=1 default=0 value=0 flags=slider
*                   color_effects 0x0098091f (menu)   : min=0 max=15 default=0 value=0
*                          rotate 0x00980922 (int)    : min=0 max=360 step=90 default=0 value=0 flags=modify-layout
*              color_effects_cbcr 0x0098092a (int)    : min=0 max=65535 step=1 default=32896 value=32896
* 
* Codec Controls
*              video_bitrate_mode 0x009909ce (menu)   : min=0 max=1 default=0 value=0 flags=update
*                   video_bitrate 0x009909cf (int)    : min=25000 max=25000000 step=25000 default=10000000 value=10000000
*          repeat_sequence_header 0x009909e2 (bool)   : default=0 value=0
*             h264_i_frame_period 0x00990a66 (int)    : min=0 max=2147483647 step=1 default=60 value=60
*                      h264_level 0x00990a67 (menu)   : min=0 max=11 default=11 value=11
*                    h264_profile 0x00990a6b (menu)   : min=0 max=4 default=4 value=4
* 
* Camera Controls
*                   auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=0 value=0
*          exposure_time_absolute 0x009a0902 (int)    : min=1 max=10000 step=1 default=1000 value=1000
*      exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=0
*              auto_exposure_bias 0x009a0913 (intmenu): min=0 max=24 default=12 value=12
*       white_balance_auto_preset 0x009a0914 (menu)   : min=0 max=10 default=1 value=1
*             image_stabilization 0x009a0916 (bool)   : default=0 value=0
*                 iso_sensitivity 0x009a0917 (intmenu): min=0 max=4 default=0 value=0
*            iso_sensitivity_auto 0x009a0918 (menu)   : min=0 max=1 default=1 value=1
*          exposure_metering_mode 0x009a0919 (menu)   : min=0 max=2 default=0 value=0
*                      scene_mode 0x009a091a (menu)   : min=0 max=13 default=0 value=0
* 
* JPEG Compression Controls
*             compression_quality 0x009d0903 (int)    : min=1 max=100 step=1 default=30 value=30
* 
* v4l2-ctl --set-ctrl=brightness=0
*
*/
