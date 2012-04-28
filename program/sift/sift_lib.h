#ifndef _SIFT_H_
#define _SIFT_H_
#include <opencv2/opencv.hpp>
#include <vector>
#define PI 3.14159265f
#define MAX_R 120
#define OCTAVES 4
#define S 3
#define SIGMA_ANT 0.5
//#define SIGMA_PRE sqrt( 2-4*SIGMA_ANT*SIGMA_ANT )
#define SIGMA_PRE 1.6
#define SIGMA sqrt(2)
#define BINS 36
#define WINDOW_SIZE 16
#define DEC_BINS 8
#define F_THRESHOLD 0.2f

using namespace cv ;
using namespace std ;

class KEY
{
public:
	/*x_d, y_d: coordinate in the extracyed layer */
	int x_l, y_l ;
	int x, y, octave, interval ;
	float orient, mag ;
	KEY(){} 
	KEY( int x1, int y1, int x2, int y2, float ori, float m, int oct, int inter )
	{ 
		x = x1 ; y = y1 ; orient = ori ; mag = m ; 
		x_l = x2 ;
		y_l = y2 ;
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
	DESCRIPT( int x1, int y1, vector<float> &f )
	{
		x = x1 ;
		y = y1 ;
		feature = f ;
	}
} ;


extern float gaussian_weight[MAX_R][MAX_R] ;
extern Mat ***isfeature ;
void sift( vector<DESCRIPT> &descriptor, Mat &img ) ;
void compute_gaussian_weight( double sigma ) ;
void gaussianBlur( Mat &src, Mat &dst, int r, float g_weight[][MAX_R] ) ;
void GenerateDoG( Mat &img_src, Mat Layers[][S+3], Mat DoGs[][S+2], double Sigmas[][S+3] ) ;
void DetectFeatures( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2], bool **final_feature ) ;
bool check_local_maximal( Mat *DoG, int i, int j ) ;
bool is_low_contrast_or_edge( Mat &DoG, int i, int j ) ;
void ComputeOrientation( Mat Layers[][S+3], double Sigmas[][S+3], vector<KEY> &keypoint_list ) ;
void GenerateFeatures(  Mat Layers[][S+3], vector<KEY> &keypoint_list, vector<DESCRIPT> &descriptor ) ;
void BuildGaussianTable( Mat &table, int size, float sigma ) ;
#endif
