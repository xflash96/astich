/* from http://www.opencv.org.cn/index.php */
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	string imagename = "lena.jpg";
	Mat img = imread(imagename);

	if(img.empty())
	{
		return -1;
	}

	namedWindow("image", 1);
	imshow("image", img);
	waitKey();
	return 0;
}
