#include <iostream>
#include<cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "sift_lib.h"

using namespace cv ;
using namespace std ;

int main( int argc, char *argv[] )
{
	Mat img_src = imread( argv[1], 1 ) ;
	vector<DESCRIPT> fv1 ;
	sift( fv1, img_src ) ;

	Mat img = Mat::zeros( img_src.rows, img_src.cols*2, CV_8UC3 ) ;
	//cvtColor( img_src, img_src, CV_RGB2GRAY ) ;
	for( int i=0 ; i<img_src.rows ; i++ )
		for( int j=0 ; j<img_src.cols ; j++ )
			img.at<Vec3b>(i,j) = img_src.at<Vec3b>(i,j) ;
			//img.at<char>(i,j) = img_src.at<char>(i,j) ;
	cerr << img_src.rows << endl ;

	img_src = imread( argv[2], 1 ) ;
	vector<DESCRIPT> fv2 ;
	sift( fv2, img_src ) ;
	//cvtColor( img_src, img_src, CV_RGB2GRAY ) ;
	for( int i=0 ; i<img_src.rows ; i++ )
		for( int j=0 ; j<img_src.cols ; j++ )
			img.at<Vec3b>(i, j+img_src.cols ) = img_src.at<Vec3b>(i,j) ;
			//img.at<char>(i+img_src.rows,j) = img_src.at<char>(i,j) ;
	cerr << img_src.rows << endl ;

	int f_size = (WINDOW_SIZE/4)*(WINDOW_SIZE/4)*DEC_BINS ;
	IplImage qImg;
	qImg = IplImage(img);
	//img.convertTo( img, CV_8UC3, 1) ;

	for( int i=0 ; i<fv1.size() ; i++ )
	{
		//find closest and second closest
		int idx[2] ;
		double tmp_dis, dis[2] ;
		dis[0] = dis[1] = 10000000000 ; 
		for( int j = 0 ; j<fv2.size() ; j++ )
		{
			tmp_dis = 0 ;
			for( int k=0 ; k<f_size ; k++ )
				tmp_dis += ( fv1[i].feature[k]-fv2[j].feature[k])*( fv1[i].feature[k]-fv2[j].feature[k]) ;
			tmp_dis = sqrt( tmp_dis ) ;
			if( tmp_dis < dis[0] )
			{
				dis[0] = tmp_dis ;
				idx[0] = j ;
			}
			else if( tmp_dis < dis[1] )
			{
				dis[1] = tmp_dis ;
				idx[1] = j ;
			}
			//if( fv1[i].x == 262 && fv2[j].x == 131 )
			//	cout << "131 dis " << tmp_dis << endl;
		}
		//if( fv1[i].x == 262 )
		//	cout << "262 dis " << dis[0] << " " << dis[1] << endl;
		if( dis[0]/dis[1]<0.8 && dis[0]<0.7 )
		{
			cerr << dis[0] << endl ;
			//printf( "(%d,%d), (%d,%d), (%d, %d)\n", fv1[i].x, fv1[i].y, fv2[ idx[0] ].x, fv2[ idx[0] ].y, fv1[i].x+fv2[ idx[0] ].x, fv1[i].y+fv2[ idx[0] ].y ) ;
			printf( "(%d,%d), (%d,%d)\n", fv1[i].x, fv1[i].y, fv2[ idx[0] ].x, fv2[ idx[0] ].y ) ;
			cvLine(&qImg, cvPoint( fv1[i].y , fv1[i].x), cvPoint(fv2[ idx[0] ].y+img_src.cols,  fv2[ idx[0] ].x  ), CV_RGB(0,255,0), 1);
		}
	}
	imwrite( argv[3], img ) ;

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
