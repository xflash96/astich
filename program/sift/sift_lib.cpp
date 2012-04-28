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
void sift( vector<DESCRIPT> &descriptor, Mat &img_src )
{
	bool **final_feature ;
	Mat imgGray ;
	Mat DoGs[OCTAVES+1][S+2], Layers[OCTAVES+1][S+3] ;
	double Sigmas[ OCTAVES+1 ][ S+3 ] ;
	vector<KEY> keypoint_list ;
	descriptor.clear() ;
	//vector<DESCRIPT> descriptor ;

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
	imwrite( "/home/student/97/b97018/htdocs/f2.png", img1 ) ;

	IplImage qImg;
	qImg = IplImage(img1);
	for( int i=0 ; i<keypoint_list.size() ; i++ )
	{
		int x=keypoint_list[i].x ;
		int y=keypoint_list[i].y ;
		f_n++ ;
		//img1.at<Vec3b>(x,y)[0] = (uchar)255 ;
		//img1.at<Vec3b>(x,y)[1] = (uchar)255 ;
		//img1.at<Vec3b>(x,y)[2] = (uchar)0 ;
		cvLine(&qImg, cvPoint(y, x), cvPoint(y, x), CV_RGB(255,255,0), 3);
		cvLine(&qImg, cvPoint(y, x), cvPoint(y+10*cos( keypoint_list[i].orient ), x+10*sin( keypoint_list[i].orient )  ), CV_RGB(255,255,0), 1);
	}
	cerr << "Feature num: " << f_n<< endl ;
	imwrite( "/home/student/97/b97018/htdocs/f1.png", img1 ) ;
	
	

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
	for( int i=0 ; i<img.rows ; i++ )
		for( int j=0 ; j<img.cols ; j++ )
			img.at<float>(i,j) /= 255 ;
	//r = (int)( 2*SIGMA_ANT+1e-7 ) ;
	//compute_gaussian_weight( SIGMA_ANT ) ;
	//gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	GaussianBlur( img, img, Size(0,0), SIGMA_ANT, 0 ) ;
	//resize( img, img, Size(0,0), 2, 2 ) ;
	pyrUp( img, img ) ;

	///preblur	
	//r = (int)( 2*SIGMA_PRE+1e-7 ) ;
	//compute_gaussian_weight( SIGMA_PRE ) ;
	//gaussianBlur( img, img_tmp, r, (float (*)[MAX_R])&gaussian_weight[r][r]) ; 
	GaussianBlur( img, img, Size(0,0), SIGMA_PRE, 0 ) ;

	//initial_sigma = sqrt( (2*SIGMA_ANT)*(2*SIGMA_ANT) + SIGMA_PRE*SIGMA_PRE );
	initial_sigma = sqrt(2);
	Layers[0][0] = img ;
	Sigmas[0][0] = initial_sigma*0.5 ;
	for( int k=0 ; k<OCTAVES ; k++ )
	{
		cerr << "octave " << k << endl ;
		sigma = initial_sigma ;
		for( int i=1 ; i<S+3 ; i++ )
		{
			sigma_f = sqrt( pow(2, 2.0/S) - 1)*sigma ;
			sigma *= pow( 2, 1.0/S ) ;
			Sigmas[k][i] = sigma * 0.5*pow( 2.0f, (float)k ) ;
			Layers[k][i]= Mat::zeros( Layers[k][i-1].rows, Layers[k][i-1].cols, CV_32FC1 )  ;
			isfeature[k][ i-1 ] = new Mat( Mat::zeros(Layers[k][i-1].rows, Layers[k][i-1].cols, CV_8UC1) );
			//r = (int)(2*sigma_f) ;
			//compute_gaussian_weight( sigma_f ) ;
			//gaussianBlur( Layers[k][i-1], Layers[k][i], r, (float (*)[MAX_R])&gaussian_weight[r][r] ) ;
			GaussianBlur( Layers[k][i-1], Layers[k][i], Size(0,0), sigma_f, 0 ) ;
			DoGs[k][i-1] = Layers[k][i]-Layers[k][i-1] ;
		}
		if( k<OCTAVES-1 )
		{
			//resize( Layers[k][0], Layers[k+1][0], Size(0,0), 0.5, 0.5 ) ;
			pyrDown( Layers[k][0], Layers[k+1][0]  )  ;
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
	/*detect min*/
	check = true ;
	for( int k1=-1 ; k1<=1 && check; k1++ )	
		for( int k2=-1 ; k2<=1 && check ; k2++ )	
		{
			if( DoG[0].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
				check = false ;
			if( DoG[2].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
				check = false ;
			if( (k1 != 0||k2 != 0) &&DoG[1].at<float>(i+k1, j+k2)<= DoG[1].at<float>(i, j) )
				check = false ;
		}
	if( check )
		return true ;
	/*detect max*/
	check = true ;
	for( int k1=-1 ; k1<=1 && check; k1++ )	
		for( int k2=-1 ; k2<=1 && check ; k2++ )	
		{
			if( DoG[0].at<float>(i+k1, j+k2)>= DoG[1].at<float>(i, j) )
				check = false ;
			if( DoG[2].at<float>(i+k1, j+k2)>= DoG[1].at<float>(i, j) )
				check = false ;
			if( (k1 != 0 || k2 != 0 ) && DoG[1].at<float>(i+k1, j+k2) >= DoG[1].at<float>(i, j) )
				check = false ;
		}
	return check ;
}

bool is_low_contrast_or_edge( Mat &DoG, int i, int j ) 
{
	float local_max ;
	float G[2] ;
	float H[2][2], det, tra ;
	G[0] = (float)( ( DoG.at<float>(i+1,j) - DoG.at<float>(i-1,j) )*0.5 ) ;
	G[1] = (float)( ( DoG.at<float>(i,j+1) - DoG.at<float>(i,j-1) )*0.5 ) ;
	H[0][0] =  DoG.at<float>(i+1,j) - 2*DoG.at<float>(i,j) + DoG.at<float>(i-1,j) ;
	H[1][1] =  DoG.at<float>(i,j+1) - 2*DoG.at<float>(i,j) + DoG.at<float>(i,j-1) ;
	H[0][1] = (float)( ( DoG.at<float>(i-1,j-1) - DoG.at<float>(i-1,j) - DoG.at<float>(i,j-1) + 
	                  DoG.at<float>(i+1,j+1))*0.25 ) ;
	H[1][0] = H[0][1] ;
	local_max = (float)0.5/( H[1][0]*H[1][0]-H[0][0]*H[1][1] )*
		    ( G[0]*( G[0]*H[1][1]-G[1]*H[0][1] ) +  G[1]*( -G[0]*H[0][1]+G[1]*H[0][0] ) ) ;

	if( abs( DoG.at<float>(i,j)+local_max )<0.03 )
		return true ;

	det = ( H[0][0]*H[1][1]-H[0][1]*H[0][1] ) ;
	tra = H[0][0]+H[1][1] ;
	if(  det < 0 || tra*tra/det > 12.1  )
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
					/*dy:x cooridinate dx: y coordinate*/
					Orient[k][idx-1].at<float>( x, y ) = atan2( dx, dy ) ;
					if( Orient[k][idx-1].at<float>( x, y )<0 )
						Orient[k][idx-1].at<float>( x, y ) += PI*2+1e-7 ;
				}

			GaussianBlur( Magnitude[k][idx-1], magBlur, Size(0,0), Sigmas[k][idx]*1.5, 0 ) ;
			window_size = (int)(Sigmas[k][idx]*1.5*9)/2 ;
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
								degree = int(ori/PI*180) ;
								hist[ (int)(degree/( 360/BINS) ) ] += magBlur.at<float>( x1+x, y1+y ) ;
							}
						max_peak_w = hist[0] ;
						for( int i=1 ; i<BINS ; i++ )
							if( hist[i]>max_peak_w )
								max_peak_w = hist[i] ;
						for( int i=0 ; i<BINS ; i++ )
						{
							if( hist[i]>0.8*max_peak_w)
							{
								for( int j=0 ; j<3 ; j++ )
								{
									X.at<float>( j, 0 ) = (float)((i+j-1)*(i+j-1)) ;
									X.at<float>( j, 1 ) = (float)(i+j-1) ;
									X.at<float>( j, 2 ) = 1.0f;
									Y.at<float>( j, 0 ) = (float)hist[ (i+j-1+BINS)%BINS ] ;
								}
								if( Y.at<float>( 0, 0 )>=Y.at<float>( 1, 0 )||
								    Y.at<float>( 2, 0 )>=Y.at<float>( 1, 0 ) )
									continue ;
								A = (X.inv())*( Y ) ;
								ori = -A.at<float>(1,0)/( 2*A.at<float>(0,0) ) ;
								if( abs(ori) > 2*BINS )
									ori = (float)i ;
								while( ori < 0 )
									ori += BINS ;
								while( ori >= BINS )
									ori -= BINS ;
								ori = ori*( 2*PI/BINS ) ;
								keypoint_list.push_back( KEY( x*scale/2, 
								                              y*scale/2,
											      x, y,
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
	Mat interpolated_mag[ OCTAVES ][ S+3 ] ;
	Mat interpolated_ori[ OCTAVES ][ S+3 ] ;
	Mat GaussianTable, WeightTable ;
	int rows, cols, xi, yi, x_l, y_l, h_window, octave, interval, f_size ;
	float dx, dy, degree, bins, tmp ;
	float hist[DEC_BINS] ;
	vector<float> feature ;

	f_size = (WINDOW_SIZE/4)*(WINDOW_SIZE/4)*DEC_BINS ;
	feature = vector<float>(f_size) ;

	for( int k=0 ; k<OCTAVES ; k++ )
	{
		rows = Layers[k][0].rows ;
		cols = Layers[k][0].cols ;
		for( int idx=1 ; idx<S+1 ; idx++ )
		{
			interpolated_mag[k][idx-1] = Mat::zeros( rows+1, cols+1, CV_32FC1 ) ;
			interpolated_ori[k][idx-1] = Mat::zeros( rows+1, cols+1, CV_32FC1 ) ;
			for( float x=1.5f ; x<rows-1.5+1e-7 ; x ++ )
				for( float y=1.5f ; y<cols-1.5+1e-7 ; y++ )
				{
					dx = ( Layers[k][idx].at<float>( (int)(x+1.5+1e-7), y )+
					       Layers[k][idx].at<float>( (int)(x+0.5+1e-7), y )-
					       Layers[k][idx].at<float>( (int)(x-1.5+1e-7), y )-
					       Layers[k][idx].at<float>( (int)(x-0.5+1e-7), y ) )*0.5f ;
					       
					dy = ( Layers[k][idx].at<float>( x, (int)(y+1.5+1e-7) )+
					       Layers[k][idx].at<float>( x, (int)(y+0.5+1e-7) )-
					       Layers[k][idx].at<float>( x, (int)(y-1.5+1e-7) )-
					       Layers[k][idx].at<float>( x, (int)(y-0.5+1e-7) ) )*0.5f ;
					xi = (int)( x+1+1e-7 ) ;
					yi = (int)( y+1+1e-7 ) ;
					interpolated_mag[k][idx-1].at<float>( xi, yi ) = sqrt( dx*dx+dy*dy ) ;
					interpolated_ori[k][idx-1].at<float>( xi, yi ) = atan2( dx, dy ) ;
					if( interpolated_ori[k][idx-1].at<float>( xi, yi )<0 )
						interpolated_ori[k][idx-1].at<float>( xi, yi ) += 2*PI+1e-7f ;
				}
		}
	}
	BuildGaussianTable( GaussianTable, WINDOW_SIZE, WINDOW_SIZE/2 ) ;
	WeightTable = Mat::zeros( WINDOW_SIZE, WINDOW_SIZE, CV_32FC1 ) ;
	h_window = WINDOW_SIZE/2 ;
	for( int idx=0 ; idx<keypoint_list.size() ; idx++ )
	{
		int fx, fy ;
		fx = keypoint_list[idx].x ;
		fy = keypoint_list[idx].y ;
		x_l = keypoint_list[idx].x_l ;
		y_l = keypoint_list[idx].y_l ;
		octave = keypoint_list[idx].octave ;
		interval = keypoint_list[idx].interval ;

		rows = Layers[ octave ][0].rows ;
		cols = Layers[ octave ][0].cols ;

		for( int i=0 ; i<WINDOW_SIZE ; i++ )
			for( int j=0 ; j<WINDOW_SIZE ; j++ )
			{
				if( x_l+i-h_window+1<0 || y_l+j-h_window+1<0 || 
				    x_l+i-h_window+1>rows || y_l+j-h_window+1>cols  )
					WeightTable.at<float>(i,j) = 0 ;
				else
				{
					WeightTable.at<float>(i,j) = GaussianTable.at<float>(i,j)*
								     interpolated_mag[ octave ][ interval-1 ].at<float>( x_l+i-h_window+1, y_l+j-h_window+1 ) ;

					//cout << i << " " << j << " "<< GaussianTable.at<float>(i,j) << " " << interpolated_mag[ octave ][ interval-1 ].at<float>( x_l+i-h_window+1, y_l+j-h_window+1 ) <<endl ;

				}
			}
		for( int i=0 ; i<WINDOW_SIZE/4 ; i++ )
			for( int j=0 ; j<WINDOW_SIZE/4 ; j++ )
			{
				memset( hist, 0, DEC_BINS*sizeof(float) ) ;
				int x_s, x_e, y_s, y_e ;
				x_s = x_l-h_window+1 + i*WINDOW_SIZE/4 ;
				x_e = x_l-h_window+1 + (i+1)*WINDOW_SIZE/4-1 ;
				y_s = y_l-h_window+1 + j*WINDOW_SIZE/4 ;
				y_e = y_l-h_window+1 + (j+1)*WINDOW_SIZE/4-1 ;
				for( int x=x_s ; x <= x_e ; x++ )
					for( int y=y_s ; y <= y_e ; y++ )
					{
						if( x<0 || y<0 || x> rows || y > cols )
							continue ;
						degree = interpolated_ori[ octave ][ interval-1 ].at<float>(x, y)-keypoint_list[idx].orient ;
						if( degree<0 )
							degree += 2*PI ;
						degree = degree/PI*180 ;
						bins = (degree/(360/DEC_BINS)) ;
						hist[ (int)(bins) ] += ( 1-abs( bins-(int)(bins)-0.5 ) )*WeightTable.at<float>( x-(x_l-h_window+1), y-( y_l-h_window+1 ) ) ;
					}
				for( int k=0 ; k<DEC_BINS ; k++ )
				{
					feature[ (i*WINDOW_SIZE/4+j)*DEC_BINS+k ] = hist[k] ;
				}
			}
		tmp = 0 ;
		for( int i=0 ; i<f_size ; i++ )
			tmp += feature[i]*feature[i] ;
		tmp = sqrt(tmp) ;
		for( int i=0 ; i<f_size ; i++ )
		{
			feature[i] /= tmp ;
			if( feature[i]>F_THRESHOLD )
				feature[i] = F_THRESHOLD ;
		}
		tmp = 0 ;
		for( int i=0 ; i<f_size ; i++ )
			tmp += feature[i]*feature[i] ;
		tmp = sqrt(tmp) ;
		for( int i=0 ; i<f_size ; i++ )
			feature[i] /= tmp ;
		descriptor.push_back( DESCRIPT( fx, fy, feature ) ) ;
	}
}
void BuildGaussianTable( Mat &table, int size, float sigma )
{
	float cen ; 
	double tmp, normal ;
	
	table = Mat::zeros( size, size, CV_32FC1 ) ;
	cen = size/2 + 0.5f ;

	normal = 0 ;
	for( int i=1 ; i<=size ; i++ )
		for( int j=1 ; j<=size ; j++ )
		{
			tmp = 1.0/( 2*PI*sigma*sigma )*
			      exp( -( (i-cen)*(i-cen)+(j-cen)*(j-cen) )/(2.0*sigma*sigma) ) ;
			table.at<float>(i-1, j-1) = tmp ;
			normal += tmp ;
		}
	for( int i=0 ; i<size ; i++ )
		for( int j=0 ; j<size ; j++ )
			table.at<float>(i,j) /= normal ;
}
