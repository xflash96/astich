#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <ctime>
#include "morph_lib.h"

using namespace cv ;
using namespace std ;

Qfloat const PI=4*atan(1);
Mat CrossDissoving( Mat &img_src, Mat &img_dst, Qfloat t )
{
	Mat mix = (1-t)*img_src+t*img_dst ;
	return mix ;
}

Mat Warping( Mat &img_src, PARA &para, Mat line_src, Mat line_dst, int rows, int cols )
{

	int n ;
	Qfloat u, v, w ;
	Qfloat unit ;
	Mat morph, W;
	Vec2f X, _P, _Q, P, Q, QmP, _QmP, _X ;
	Vec3f pixel ;
	Vec2f u_param, v_param, x_param;

	morph = Mat::zeros( rows, cols, CV_32FC3 ) ;
	W = Mat::zeros( rows, cols, CV_32FC1 ) ;
	n = line_src.cols;
	unit = sqrt( rows*rows+cols*cols ) ;
	/* precalc line dist
	*/

	const Vec2f* line_dst_0 = line_dst.ptr<Vec2f>(0);
	const Vec2f* line_dst_1 = line_dst.ptr<Vec2f>(1);
	const Vec2f* line_src_0 = line_src.ptr<Vec2f>(0);
	const Vec2f* line_src_1 = line_src.ptr<Vec2f>(1);
	for( int l=0 ; l<n ; l++ )
	{
		P = line_dst_0[l] ;
		Q = line_dst_1[l];
		_P = line_src_0[l] ;
		_Q = line_src_1[l] ;

		P.val[0] *= rows ;
		P.val[1] *= cols ;
		Q.val[0] *= rows ;
		Q.val[1] *= cols ;
		_P.val[0] *= img_src.rows ;
		_P.val[1] *= img_src.cols ;
		_Q.val[0] *= img_src.rows ;
		_Q.val[1] *= img_src.cols ;

		QmP = Q-P ;
		_QmP = _Q-_P ;
		u_param = QmP*(1/( QmP.val[0]*QmP.val[0]+QmP.val[1]*QmP.val[1] )) ;
		v_param = Vec2f( QmP.val[1], -QmP.val[0] )*(1/norm(QmP)) ;
		x_param = 1/norm(_QmP)*Vec2f(_QmP.val[1], -_QmP.val[0]);

		for( int i=0 ; i<rows ; i++ ){
			Vec3f* morph_i = morph.ptr<Vec3f>(i);
			float* W_i = W.ptr<float>(i);
			for( int j=0 ; j<cols ; j++ )
			{
				//X = Vec2f( i/(Qfloat)rows, j/(Qfloat)cols ) ;
				X = Vec2f( i, j ) ;

				u = (X-P).dot( u_param ) ;
				v = (X-P).dot( v_param ) ;

				_X = _P + u*(_QmP) + v*x_param ;

				int _x = (int) _X.val[0], _y = (int) _X.val[1];

				if(_x>=0 && _y>=0 && _x<img_src.rows && _y<img_src.cols)
				{
					w = 1/( countDisToSegment( P, Q, X, abs(v) ) );
					if( para.type == 0 )
						pixel = img_src.at<Vec3f>(_x, _y);
					else if( para.type == 1 )
						pixel = BilinearInterpolation( img_src, _X.val[0], _X.val[1], 1 );
					else if( para.type == 2 )
						pixel = GaussianKernelInterpolation( img_src, _X.val[0], _X.val[1], 1 );
					morph_i[j] += w*pixel ; 
					W_i[j] += w ;
				}
			}
		}
	}
	for( int i=0; i<rows; i++)
		for( int j=0; j<cols; j++)
			morph.at<Vec3f>(i, j) = 1/W.at<float>(i,j)*morph.at<Vec3f>(i, j) ;
	return morph ;
}

Vec3f BilinearInterpolation( Mat &img_src, Qfloat x, Qfloat y, Qfloat sigma )
{
	Qfloat a, b, w, W  ;
	int quan_x, quan_y ;
	Vec3f pixel = Vec3f(0, 0, 0) ;
	W = 0 ;
	a = 1-x+int(x) ;
	b = 1-y+int(y) ;
	for( int i=0 ; i<=1 ; i++, a=1-a )
		for( int j=0 ; j<=1 ; j++, b=1-b )
		{
			quan_x = int( x+i+1e-7 ) ;
			quan_y = int( y+j+1e-7 ) ;
			if(quan_x<0 || quan_y<0 || quan_x>=img_src.rows || quan_y>=img_src.cols)
				continue ;
			w = a*b ;
			if( w>1e-7 )
			{
				pixel = pixel+w*img_src.at<Vec3f>(quan_x, quan_y) ;
				W += w ;
			}
		}
	if( W < 1e-7 )
		pixel = Vec3f(-1, -1, -1) ;
	else
		pixel = 1/W*pixel ;
	return pixel ;
}

Vec3f GaussianInterpolation( Mat &img_src, Qfloat x, Qfloat y, Qfloat sigma )
{
	Qfloat w, W, dis ;
	int quan_x, quan_y, rows, cols ;
	rows = img_src.rows ;
	cols = img_src.cols ;
	Vec3f pixel = Vec3f(0, 0, 0) ;
	W = 0 ;
	for( int i=-2 ; i<=2 ; i++ )
		for( int j=-2 ; j<=2 ; j++ )
		{
			quan_x = (int)( x+i+1e-7 ) ;
			quan_y = (int)( y+j+1e-7 ) ;
			dis = ( (quan_x-x)*(quan_x-x)+(quan_y-y)*(quan_y-y) )/sigma ;
			if( sqrt(dis)>2+1e-7 )
				continue ;
			if( quan_x<0 || quan_x>=rows || quan_y<0 || quan_y >= cols  )
				continue ;
			else
			{
				w = 1/( 2*PI*sigma)*exp( -0.5*dis ) ;
				W += w ;
				pixel = pixel+w*img_src.at<Vec3f>(quan_x, quan_y) ;
			}
		}
	if( W < 1e-7 )
		pixel = Vec3f(-1, -1, -1) ;
	else
		pixel = 1/W*pixel ;
	return pixel ;

}

Qfloat _GaussianKernel[] = {
	0.007306882745280776, 0.032747176537766653, 0.05399096651318806, 0.032747176537766653, 0.007306882745280776,
	0.032747176537766653, 0.146762663173739930, 0.24197072451914337, 0.146762663173739930, 0.032747176537766653,
	0.053990966513188060, 0.241970724519143370, 0.39894228040143270, 0.241970724519143370, 0.053990966513188060,
	0.032747176537766653, 0.146762663173739930, 0.24197072451914337, 0.146762663173739930, 0.032747176537766653,
	0.007306882745280776, 0.032747176537766653, 0.05399096651318806, 0.032747176537766653, 0.007306882745280776,
};

Vec3f GaussianKernelInterpolation( Mat &img_src, Qfloat x, Qfloat y, Qfloat sigma )
{
	Qfloat w, W, dis ;
	int quan_x, quan_y, rows, cols ;
	rows = img_src.rows ;
	cols = img_src.cols ;
	Vec3f pixel = Vec3f(0, 0, 0) ;
	W = 0 ;
	for( int i=-2 ; i<=2 ; i++ )
		for( int j=-2 ; j<=2 ; j++ )
		{
			quan_x = (int)( x+i+1e-7 ) ;
			quan_y = (int)( y+j+1e-7 ) ;
			//dis = ( (quan_x-x)*(quan_x-x)+(quan_y-y)*(quan_y-y) )/sigma ;
			//if( sqrt(dis)>2+1e-7 )
			//	continue ;
			if( quan_x<0 || quan_x>=rows || quan_y<0 || quan_y >= cols  )
				continue ;
			else
			{
				int idx = (i+2)*5+j+2;
				w = _GaussianKernel[idx];
				W += w ;
				pixel = pixel+w*img_src.at<Vec3f>(quan_x, quan_y) ;
			}
		}
	if( abs( W ) < 1e-7 )
		Vec3f pixel = Vec3f(-1, -1, -1) ;
	else
		pixel = 1/W*pixel ;
	return pixel ;

}

Qfloat countDisToSegment( Vec2f P, Vec2f Q, Vec2f X, Qfloat min_dis_to_line )
{
#if 0
	Vec2f QmP = Q-P ;
	if( (X-P).dot( QmP ) * ( X-Q ).dot(QmP) < 0 )
		return min_dis_to_line*min_dis_to_line ;
	else
#endif
	return min((X-P).dot(X-P), (X-Q).dot(X-Q));
//	return min( norm( X-P ), norm(X-Q) ) ;
}
