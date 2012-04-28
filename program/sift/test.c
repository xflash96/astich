#include<stdio.h>

int test( int a[][5][5] )
{
	int i, j ;
	for( i=-1 ; i<=1 ; i++ )
		for( j=-1; j<=1 ; j++ )
			a[0][i][j] = 1 ;
	printf( "%d\n", a ) ;
	printf( "%d\n", &a[0][0] ) ;
}
int main()
{
	int a[5][5][5] = {0} ;
	test( &a[1][2][2] ) ;
	//printf( "%d\n", &a[2][2] ) ;
	int i, j ;
	for( i=0 ; i<5 ; i++ )
	{
		for( j=0; j<5 ; j++ )
		{
			printf( "%d ", a[1][i][j] ) ;
		}
		printf( "\n" ) ;
	}
}
