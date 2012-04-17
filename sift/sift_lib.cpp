#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <cstring>
#include <algorithm>
#include "sift_lib.h"

using namespace cv;
using namespace std;

float gaussian_weight[MAX_R][MAX_R] ;
void sift( Mat &img_src )
{
	bool **isfeature ;
	Mat imgGray ;
	Mat DoGs[OCTAVES][S+2], Layers[OCTAVES][S+3] ;

	isfeature = new bool*[img_src.rows] ;
	for( int i=0 ; i<img_src.rows ; i++ )
	{
		isfeature[i] = new bool[img_src.cols] ;
		memset( isfeature[i], 0, img_src.cols ) ;
	}

	cvtColor(img_src, imgGray, CV_RGB2GRAY);
	GenerateDoG( imgGray, Layers, DoGs ) ;
	DetectFeatures( imgGray, Layers, DoGs, isfeature ) ;

	cerr << "Output\n" ;
	Mat img1 ;
	cvtColor(imgGray, img1, CV_GRAY2RGB);
	int f_n  = 0 ;
	for( int i=0 ; i<img1.rows ; i++ )
		for( int j=0 ; j<img1.cols ; j++ )
			if( isfeature[i][j] )
			{
				f_n++ ;
				img1.at<Vec3b>(i,j)[0] = (uchar)255 ;
				img1.at<Vec3b>(i,j)[1] = (uchar)255 ;
				img1.at<Vec3b>(i,j)[2] = (uchar)0 ;
			}
	cerr << "Feature num: " << f_n<< endl ;
	imwrite( "/home/student/97/b97018/htdocs/f.png", img1 ) ;

	cerr << "Free Memory\n" ;
	for( int i=0 ; i<img1.rows ; i++ )
		delete[] isfeature[i] ;
	delete[] isfeature ;
}
void compute_gaussian_weight( double sigma )
{
	double sigma_sqr ;
	double w ;
	int r ;
	sigma_sqr = 2*sigma*sigma ;
	r = int(2*sigma+1e-7) ;
	for( int i=0 ; i<=r ; i++ )
		for( int j=0 ; j<=r ; j++ )
		{
			w = 1.0/(PI*sigma_sqr)*exp( -(double)(i*i+j*j)/sigma_sqr ) ;
			gaussian_weight[ r+i ][ r+j ] = (float)w ;
			gaussian_weight[ r+i ][ r-j ] = (float)w ;
			gaussian_weight[ r-i ][ r+j ] = (float)w ;
			gaussian_weight[ r-i ][ r-j ] = (float)w ;
		}
}
void gaussianBlur( Mat &src, Mat &dst, int r, float g_weight[][MAX_R] )
{
	int i_from, j_from, i_to, j_to ;
	float total_w, pixel ;

	for( int i=0 ; i<src.rows ; i++ )
		for( int j=0 ; j<src.cols ; j++ )
		{
			i_from = max( -r, -i ) ;
			j_from = max( -r, -j ) ;
			i_to = min( r, src.rows-i-1 ) ;
			j_to = min( r, src.cols-j-1 ) ;
			
			total_w = 0 ;
			pixel = 0 ;
			for( int k1=i_from ; k1<=i_to ; k1++ )
				for( int k2=j_from ; k2<=j_to ; k2++ )
				{
					total_w += g_weight[ k1 ][ k2 ] ;
					pixel += src.at<float>( i+k1, j+k2 )*g_weight[ k1 ][ k2 ] ;
				}
			dst.at<float>(i,j) = pixel/total_w ;
		}
}
void GenerateDoG( Mat &img_src, Mat Layers[][S+3], Mat DoGs[][S+2] ) 
{
	cerr << "Generate DoGs\n" ;
	int r ;
	double sigma, sigma_f, initial_sigma ;
	Mat img, img_tmp ;


	img_src.convertTo( img, CV_32FC1, 1) ;
	img_tmp = img ;
	r = (int)( 2*SIGMA_ANT+1e-7 ) ;
	compute_gaussian_weight( SIGMA_ANT ) ;
	gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	resize( img_tmp, img, Size(0,0), 2, 2 ) ;

	///preblur	
	img_tmp = img ;
	r = (int)( 2*SIGMA_PRE+1e-7 ) ;
	compute_gaussian_weight( SIGMA_PRE ) ;
	gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	img = img_tmp ;

	initial_sigma = sqrt( (2*SIGMA_ANT)*(2*SIGMA_ANT) + SIGMA_PRE*SIGMA_PRE );
	Layers[0][0] = img ;
	for( int k=0 ; k<OCTAVES ; k++ )
	{
		cerr << "octave " << k << endl ;
		sigma = initial_sigma ;
		for( int i=1 ; i<S+3 ; i++ )
		{
			cerr << "s " << i << endl ;
			sigma_f = sqrt( pow(2, 2.0/S) - 1)*sigma ;
			sigma *= pow( 2, 1.0/S ) ;
			r = (int)(2*sigma_f) ;
			Layers[k][i]= Mat::zeros( Layers[k][i-1].rows, Layers[k][i-1].cols, CV_32FC1 )  ;
			compute_gaussian_weight( sigma_f ) ;
			gaussianBlur( Layers[k][i-1], Layers[k][i], r, (float (*)[MAX_R])&gaussian_weight[r][r] ) ;
		}
		cerr << "generate DoGs\n" ;
		for( int i=0 ; i<S+2 ; i++ )
			DoGs[k][i] = Layers[k][i+1]-Layers[k][i] ;
		//img_tmp = img ;
		//r = (int)( 2*SIGMA_PRE+1e-7 ) ;
		//compute_gaussian_weight( SIGMA_PRE ) ;
		//gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
		//img = img_tmp ;
		resize( Layers[k][0], Layers[k+1][0], Size(0,0), 0.5, 0.5 ) ;
	}
}

void DetectFeatures( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2],  bool **final_feature )
{
	cerr << "Detect Features\n" ;
	bool **isfeature ;
	float scale ;
	isfeature = new bool*[ Layers[0][0].rows ] ;
	for( int i=0 ; i<Layers[0][0].rows ; i++ )
		isfeature[i] = new bool[ Layers[0][0].cols ] ;

	for( int k=0 ; k<OCTAVES ; k++ )
	{
		/*initialize*/
		for( int i=0 ; i<Layers[k][0].rows ; i++ )
			memset( isfeature[i], 0, Layers[k][0].cols ) ;
		for( int idx=1 ; idx<S+1 ; idx++ )
		{
			for( int i=1 ; i<DoGs[k][idx].rows-1 ; i++ )
				for( int j=1 ; j<DoGs[k][idx].cols-1 ; j++ )
				{
					if( isfeature[i][j] || !check_local_maximal( &DoGs[k][idx-1], i, j) )
						continue ;
					if( is_low_contrast_or_edge( DoGs[k][idx], i, j ) ) 
						continue ;
					isfeature[i][j] = 1 ;
				}
		}
		scale = (float)img.rows/(float)Layers[k][0].rows ;
		for( int i=1 ; i<DoGs[k][0].rows-1 ; i++ )
			for( int j=1 ; j<DoGs[k][0].cols-1 ; j++ )
				if( isfeature[i][j] )
					final_feature[ (int)((float)i*scale) ][ (int)((float)j*scale) ] = 1 ;
					
	}

	for( int i=0 ; i<Layers[0][0].rows ; i++ )
		delete[] isfeature[i] ;
	delete[] isfeature ;
}

bool check_local_maximal( Mat *DoG, int i, int j )
{
	bool check ;
	check = true ;
	/*detect min*/
	for( int k1=-1 ; k1<=1 && check; k1++ )	
		for( int k2=-1 ; k2<=1 ; k2++ )	
		{
			if( DoG[0].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
			if( DoG[2].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
			if( (k1 != 0||k2 != 0) &&DoG[1].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
		}
	if( check )
		return true ;
	check = true ;
	/*detect max*/
	for( int k1=-1 ; k1<=1 && check; k1++ )	
		for( int k2=-1 ; k2<=1 ; k2++ )	
		{
			if( DoG[0].at<float>(i+k1, j+k2)>= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
			if( DoG[2].at<float>(i+k1, j+k2)>= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
			if( (k1 != 0 || k2 != 0 ) && DoG[1].at<float>(i+k1, j+k2) >= DoG[1].at<float>(i, j) )
			{
				check = false ;
				break ;
			}
		}
	return check ;
}

bool is_low_contrast_or_edge( Mat &DoG, int i, int j ) 
{
	float local_max ;
	float G[2] ;
	float H[2][2], det ;
	G[0] = (float)( ( DoG.at<float>(i+1,j) - DoG.at<float>(i-1,j) )*0.5 ) ;
	G[1] = (float)( ( DoG.at<float>(i,j+1) - DoG.at<float>(i,j-1) )*0.5 ) ;
	H[0][0] =  DoG.at<float>(i+1,j) - 2*DoG.at<float>(i,j) + DoG.at<float>(i-1,j) ;
	H[1][1] =  DoG.at<float>(i,j+1) - 2*DoG.at<float>(i,j) + DoG.at<float>(i,j-1) ;
	H[0][1] = (float)( ( DoG.at<float>(i-1,j-1) - DoG.at<float>(i-1,j) - DoG.at<float>(i,j-1) + 
	                  DoG.at<float>(i+1,j+1))*0.25 ) ;
	H[1][0] = H[0][1] ;
	local_max = (float)0.5/( H[1][0]*H[1][0]-H[0][0]*H[1][1] )*
		    ( G[0]*( G[0]*H[1][1]-G[1]*H[0][1] ) +  G[1]*( -G[0]*H[0][1]+G[1]*H[0][0] ) ) ;

	if( abs( DoG.at<float>(i,j)+local_max )<0.03*255 )
		return true ;
	det = ( H[0][0]*H[1][1]-H[0][1]*H[0][1] ) ;
	if(  det < 0 || (H[0][0]+H[1][1])*(H[0][0]+H[1][1])/det >= 12.1  )
		return true ;
	return false ;
}
