#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#define PI 3.14159265f
#define MAX_R 120
#define OCTAVES 4 
#define S 2
#define SIGMA_ANT 0.5
#define SIGMA_PRE sqrt( 2-4*SIGMA_ANT*SIGMA_ANT )
#define SIGMA sqrt(2)

using namespace cv ;
using namespace std ;

class Point
{
public:
	int x, y ;
} ;

extern float gaussian_weight[MAX_R][MAX_R] ;
void sift( Mat &img ) ;
void compute_gaussian_weight( double sigma ) ;
void gaussianBlur( Mat &src, Mat &dst, int r, float g_weight[][MAX_R] ) ;
void GenerateDoG( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2] ) ;
void DetectFeatures( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2],  bool **final_feature ) ;
bool check_local_maximal( Mat *DoG, int i, int j ) ;
bool is_low_contrast_or_edge( Mat &DoG, int i, int j ) ;
#endif
