#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <cstring>
#include <algorithm>
#include <vector>
#include "sift_lib.h"

using namespace cv;
using namespace std;

float gaussian_weight[MAX_R][MAX_R] ;
Mat ***isfeature ;
void sift( Mat &img_src )
{
	bool **final_feature ;
	Mat imgGray ;
	Mat DoGs[OCTAVES+1][S+2], Layers[OCTAVES+1][S+3] ;
	double Sigmas[ OCTAVES+1 ][ S+3 ] ;
	vector<KEY> keypoint_list ;
	vector<DESCRIPT> descriptor ;

	isfeature = new Mat**[ OCTAVES ] ;
	for( int i=0 ; i<OCTAVES ; i++ )
		isfeature[i] = new Mat*[ S+2 ] ;
	final_feature = new bool*[ img_src.rows ] ;
	for( int i=0 ; i<img_src.rows ; i++ )
	{
		final_feature[i] = new bool[ img_src.cols ] ;
		memset( final_feature[i], 0, img_src.cols ) ;
	}

	cvtColor(img_src, imgGray, CV_RGB2GRAY);
	GenerateDoG( imgGray, Layers, DoGs, Sigmas ) ;
	DetectFeatures( imgGray, Layers, DoGs, final_feature ) ;
	ComputeOrientation( Layers, Sigmas, keypoint_list ) ;
	GenerateFeatures( Layers, keypoint_list, descriptor ) ;


	cerr << "Output\n" ;
	Mat img1 ;
	cvtColor(imgGray, img1, CV_GRAY2RGB);
	int f_n  = 0 ;
	for( int i=0 ; i<img1.rows ; i++ )
		for( int j=0 ; j<img1.cols ; j++ )
			if( final_feature[i][j] )
			{
				f_n++ ;
				img1.at<Vec3b>(i,j)[0] = (uchar)255 ;
				img1.at<Vec3b>(i,j)[1] = (uchar)255 ;
				img1.at<Vec3b>(i,j)[2] = (uchar)0 ;
			}
	cerr << "Feature num: " << f_n<< endl ;
	imwrite( "/home/student/97/b97018/htdocs/f.png", img1 ) ;

	cerr << "Free Memory\n" ;
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
void GenerateDoG( Mat &img_src, Mat Layers[][S+3], Mat DoGs[][S+2], double Sigmas[][S+3] ) 
{
	cerr << "Generate DoGs\n" ;
	//int r ;
	double sigma, sigma_f, initial_sigma ;
	Mat img, img_tmp ;


	img_src.convertTo( img, CV_32FC1, 1) ;
	img_tmp = img ;
	//r = (int)( 2*SIGMA_ANT+1e-7 ) ;
	//compute_gaussian_weight( SIGMA_ANT ) ;
	//gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	GaussianBlur( img, img_tmp, Size(0,0), SIGMA_ANT, 0 ) ;
	resize( img_tmp, img, Size(0,0), 2, 2 ) ;

	///preblur	
	img_tmp = img ;
	//r = (int)( 2*SIGMA_PRE+1e-7 ) ;
	//compute_gaussian_weight( SIGMA_PRE ) ;
	//gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	GaussianBlur( img, img_tmp, Size(0,0), SIGMA_PRE, 0 ) ;
	img = img_tmp ;

	initial_sigma = sqrt( (2*SIGMA_ANT)*(2*SIGMA_ANT) + SIGMA_PRE*SIGMA_PRE );
	Layers[0][0] = img ;
	Sigmas[0][0] = initial_sigma*0.5 ;
	for( int k=0 ; k<OCTAVES ; k++ )
	{
		cerr << "octave " << k << endl ;
		sigma = initial_sigma ;
		for( int i=1 ; i<S+3 ; i++ )
		{
			cerr << "s " << i << endl ;
			sigma_f = sqrt( pow(2, 2.0/S) - 1)*sigma ;
			sigma *= pow( 2, 1.0/S ) ;
			Sigmas[k][i] = sigma * 0.5 *pow( 2.0f, (float)i ) ;
			Layers[k][i]= Mat::zeros( Layers[k][i-1].rows, Layers[k][i-1].cols, CV_32FC1 )  ;
			isfeature[k][ i-1 ] = new Mat( Mat::zeros(Layers[k][i-1].rows, Layers[k][i-1].cols, CV_8UC1) );
			//r = (int)(2*sigma_f) ;
			//compute_gaussian_weight( sigma_f ) ;
			//gaussianBlur( Layers[k][i-1], Layers[k][i], r, (float (*)[MAX_R])&gaussian_weight[r][r] ) ;
			GaussianBlur( Layers[k][i-1], Layers[k][i], Size(0,0), sigma_f, 0 ) ;
		}
		cerr << "generate DoGs\n" ;
		for( int i=0 ; i<S+2 ; i++ )
			DoGs[k][i] = Layers[k][i+1]-Layers[k][i] ;
		if( k<OCTAVES-1 )
		{
			resize( Layers[k][0], Layers[k+1][0], Size(0,0), 0.5, 0.5 ) ;
			Sigmas[k+1][0] = Sigmas[k][S] ;
		}
	}
}

void DetectFeatures( Mat &img, Mat Layers[][S+3], Mat DoGs[][S+2],  bool **final_feature )
{
	cerr << "Detect Features\n" ;
	float scale ;
	for( int k=0 ; k<OCTAVES ; k++ )
	{
		/*initialize*/
		scale = (float)img.rows/(float)Layers[k][0].rows ;
		for( int idx=1 ; idx<S+1 ; idx++ )
		{
			for( int i=1 ; i<DoGs[k][idx].rows-1 ; i++ )
				for( int j=1 ; j<DoGs[k][idx].cols-1 ; j++ )
				{
					if( !check_local_maximal( &DoGs[k][idx-1], i, j) )
						continue ;
					if( is_low_contrast_or_edge( DoGs[k][idx], i, j ) ) 
						continue ;
					isfeature[k][ idx-1 ]->at<char>( i, j  ) = 1 ;
					final_feature[ (int)((float)i*scale) ][ (int)((float)j*scale) ] = 1 ;
				}
		}
	}

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
void ComputeOrientation( Mat Layers[][S+3], double Sigmas[][S+3], vector<KEY> &keypoint_list ) 
{
	cerr << "Compute Orientation" << endl ;
	Mat Orient[ OCTAVES ][ S+2 ] ;
	Mat Magnitude[ OCTAVES ][ S+2 ] ;
	Mat X = Mat( 3, 3, CV_32FC1 ) ;
	Mat Y = Mat( 3, 1, CV_32FC1 ) ;
	Mat A ;
	Mat magBlur ;
	double max_peak_w, hist[BINS] ;
	float dx, dy, ori ;
	int window_size, scale, degree ;
	scale = 1 ;
	for( int k=0 ; k<OCTAVES ; k++, scale *= 2 )
	{
		cerr << k << endl ;
		for( int idx=1 ; idx<S+1 ; idx++ )
		{
			Orient[k][idx-1] = Mat::zeros( Layers[k][0].rows, Layers[k][0].cols, CV_32FC1 ) ;
			Magnitude[k][idx-1] = Mat::zeros( Layers[k][0].rows, Layers[k][0].cols, CV_32FC1 ) ;
			for( int x=1 ; x<Layers[k][0].rows-1 ; x++ )
				for( int y=1 ; y<Layers[k][0].cols-1 ; y++ )
				{
					dx = Layers[k][idx].at<float>( x+1, y)-Layers[k][idx].at<float>( x-1, y);
					dy = Layers[k][idx].at<float>( x, y+1)-Layers[k][idx].at<float>( x, y-1);
					Magnitude[k][idx-1].at<float>( x, y )=sqrt( dx*dx+dy*dy ) ;
					Orient[k][idx-1].at<float>( x, y ) = atan( dy/dx ) ;
				}
			GaussianBlur( Magnitude[k][idx-1], magBlur, Size(0,0), Sigmas[k][idx]*1.5, 0 ) ;
			window_size = (int)(Sigmas[k][idx]*1.5*9) ;
			for( int x=0 ; x<Layers[k][0].rows ; x++ )
				for( int y=0 ; y<Layers[k][0].cols ; y++ )
					if( isfeature[k][idx-1]->at<char>(x,y) == 1 )
					{
						memset( hist, 0, BINS*sizeof(double) ) ;
						for( int x1 = -window_size ; x1<=window_size ; x1++ )
							for( int y1 = -window_size ; y1<=window_size ; y1++ )
							{
								if( x+x1<0 || y+y1<0 ||
								    x+x1>=Layers[k][0].rows || 
								    y+y1>=Layers[k][0].cols )
									continue ;
								ori = Orient[k][idx-1].at<float>( x1+x, y1+y ) ;
								ori += PI ;
								degree = int(ori/PI*180) ;
								hist[ (int)(degree/( 360/BINS) ) ] += magBlur.at<float>( x1+x, y1+y ) ;
							}
						max_peak_w = hist[0] ;
						for( int i=1 ; i<BINS ; i++ )
							if( hist[i]>max_peak_w )
							{
								max_peak_w = hist[i] ;
							}
						for( int i=0 ; i<BINS ; i++ )
						{
							if( hist[i]>0.8*max_peak_w)
							{
								for( int j=0 ; j<3 ; j++ )
								{
									X.at<float>( j, 0 ) = (float)((i-j-1)*(i-j-1)) ;
									X.at<float>( j, 1 ) = (float)(i-j-1) ;
									X.at<float>( j, 2 ) = 1.0f;
									Y.at<float>( j, 0 ) = (float)hist[ (i-j-1+BINS)%BINS ] ;
								}
								A = (X.inv())*( Y ) ;
								ori = -A.at<float>(1,0)/( 2*A.at<float>(0,0) ) ;
								if( abs(ori) > 2*BINS )
									ori = (float)i ;
								while( ori < 0 )
									ori += BINS ;
								while( ori >= BINS )
									ori -= BINS ;
								keypoint_list.push_back( KEY( x*scale/2, 
								                              y*scale/2,
											      ori,
											      (float)hist[i],
											      k, idx) ) ;
							}
						}
					}

		}
	}
}

void GenerateFeatures(  Mat Layers[][S+3], vector<KEY> &keypoint_list, vector<DESCRIPT> &descriptor ) 
{
	cerr << "Generate features\n" ;
	//create interpolated image
	MAT interpolated_mag[ OCTAVES ][ S+3 ] ;
	MAT interpolated_ori[ OCTAVES ][ S+3 ] ;
	int width, height ;
	for( int k=0 ; k<OCTAVES ; k++ )
	{
		width = Layers[k][0].rows ;
		height = Layers[k][1].cols ;
		for( int idx=1 ; idx<S+1 ; idx++ )
		{
			
		}
	}
}
