#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#include <vector>
#define PI 3.14159265f
#define MAX_R 120
#define OCTAVES 4
#define S 2
#define SIGMA_ANT 0.5
//#define SIGMA_PRE sqrt( 2-4*SIGMA_ANT*SIGMA_ANT )
#define SIGMA_PRE 1.6
#define SIGMA sqrt(2)
#define BINS 36

using namespace cv ;
using namespace std ;

class KEY
{
public:
	int x, y, octave, interval ;
	float orient, mag ;
	KEY(){} 
	KEY( int xi, int yi, float ori, float m, int oct, int inter )
	{ 
		x = xi ; y = yi ; orient = ori ; mag = m ; 
		octave = oct ;
		interval = inter ;
	}
} ;
class DESCRIPT
{
public:
	int x, y ;
	vector<float> feature ;
	DESCRIPT() {}
	DESCRIPT( int x1, int y1, vector<double> &f )
	{
		x = x1 ;
		y = y1 ;
		feature = f ;
	}
} ;


extern float gaussian_weight[MAX_R][MAX_R] ;
extern Mat ***isfeature ;
void sift( Mat &img ) ;
void compute_gaussian_weight( double sigma ) ;
void gaussianBlur( Mat &src, Mat &dst, int r, float g_weight[][MAX_R] ) ;
void GenerateDoG( Mat &img_src, Mat Layers[][S+3], Mat DoGs[][S+2], double Sigmas[][S+3] ) ;
void DetectFeatures( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2], bool **final_feature ) ;
bool check_local_maximal( Mat *DoG, int i, int j ) ;
bool is_low_contrast_or_edge( Mat &DoG, int i, int j ) ;
void ComputeOrientation( Mat Layers[][S+3], double Sigmas[][S+3], vector<KEY> &keypoint_list ) ;
void GenerateFeatures(  Mat Layers[][S+3], vector<KEY> &keypoint_list, vector<DESCRIPT> &descriptor ) ;
#endif
