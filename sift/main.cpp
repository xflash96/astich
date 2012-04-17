#include <iostream>
#include <opencv2/opencv.hpp>
#include "sift_lib.h"

using namespace cv ;
using namespace std ;

int main( int argc, char *argv[] )
{
	Mat img_src = imread( argv[1], 1 ) ;
	Mat img ;
	sift( img_src ) ;
	//resize( img_src, img, Size(0,0), 2, 2 ) ;
	//imwrite( argv[2], img ) ;
	/*
	Mat img, img1 ;
	cvtColor(img_src,img,CV_RGB2GRAY);
	//img.convertTo( img_src, CV_32FC1, 1.0/255) ;
	for( int j=0 ; j<img.cols ; j++ )
	{
		//cerr << (img.at<float>(100,j)) <<endl ;
		(img.at<uchar>(100,j)) = 200 ;
		//(img.at<Vec3f>(10,j)).val[0] = 1;
		//(img.at<Vec3f>(10,j)).val[1] =1 ;
		//(img.at<Vec3f>(10,j)).val[2] = 1 ;
	}
	//cerr << "test" << endl ;
	img.convertTo( img1, CV_32FC1, 1 ) ;
	cerr << (int)(img.at<uchar>(100,0)) << endl ;
	img1.at<float>(100,0 ) = 200.5 ;
	cerr << img1.at<float>(100,0 ) << endl ;
	img1.convertTo( img, CV_8UC1 ) ;
	*/
}
