/* from http://www.opencv.org.cn/index.php */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

char **IMG_FNAME;

void INFO(const char *tmpl, ...)
{
	va_list ap;
	va_start(ap, tmpl);
	fprintf(stderr, "\t");
	vfprintf(stderr, tmpl, ap);
	fprintf(stderr, "\n");
	va_end(ap);
}

int randint(int lower, int upper)
{
	return lower+(int)(rand()*1./RAND_MAX*(upper-lower));
}


void getFeatures(Mat &img, vector<KeyPoint>& keypoints, Mat& descrs)
{
	SiftFeatureDetector detector(0.08,0.08);
	detector.detect (img, keypoints);
	
	SiftDescriptorExtractor extractor;
	extractor.compute (img, keypoints, descrs);
}

void showFeature(Mat& src, Mat& dst, int src_img_idx, int dst_img_idx)
{
	Mat src_img = imread(IMG_FNAME[src_img_idx]);
	Mat dst_img = imread(IMG_FNAME[dst_img_idx]);
	Size size(src_img.cols+dst_img.cols, MAX(src_img.rows,dst_img.rows));
	Mat canvas = Mat(size, CV_MAKETYPE(src_img.depth(), 3));
	Mat offset = Mat(2,1,CV_32FC1);
	Mat src_canvas = canvas(Rect(0,0,src_img.cols,src_img.rows));
	Mat dst_canvas = canvas(Rect(src_img.cols,0,dst_img.cols,dst_img.rows));
	src_img.copyTo(src_canvas);
	dst_img.copyTo(dst_canvas);

	for (int i=0; i<(size_t)src.rows; i++) {
		Point st, ed;
		Scalar color = Scalar( randint(0,255),
			randint(0,255), randint(0,255));
		float x = src.at<float>(i,0);
		float y = src.at<float>(i,1);
		st.x = (int)x;
		st.y = (int)y;
		circle(src_canvas, st, 4, color, 4);

		x = dst.at<float>(i,0);
		y = dst.at<float>(i,1);
		ed.x = (int)x;
		ed.y = (int)y;
		circle(dst_canvas, ed, 4, color, 4);

		ed.x += dst_img.cols;

		line(canvas, st, ed, color);
	}
	imshow("canvas", canvas);
	waitKey(100);
}

/* Get All features from features_fname.
 * If the file does not exist, get them from every img_fnames
 * and build the feature file.
 */
void getAllFeatures(int n_imgs, char** img_fnames, const char *features_fname, vector<Mat>& keypoints, vector<Mat>& descrs)
{
	vector<KeyPoint> frame_keypoints;
	Mat frame_descrs;
	vector<Point2f> frame_keypoints_xy;
	Mat img;

	if (features_fname && -1!=access(features_fname, F_OK)) {
		FileStorage fs(features_fname, FileStorage::READ);
		fs["keypoints"] >> keypoints;
		fs["descrs"] >> descrs;
		fs.release();
		return;
	}

	FileStorage fs(features_fname, FileStorage::WRITE);
	for (int i=0; i<n_imgs; i++) {
		INFO("reading %d frame", i);
		img = imread(img_fnames[i]);

		frame_keypoints.clear();
		frame_keypoints_xy.clear();
		getFeatures(img, frame_keypoints, frame_descrs);
		INFO("frame_keypoints.size = %d", frame_keypoints.size());

		for (size_t j=0; j<frame_keypoints.size(); j++)
			frame_keypoints_xy.push_back (frame_keypoints[j].pt);
		Mat frame_keypoints_mat(frame_keypoints_xy, true);

		keypoints.push_back(frame_keypoints_mat);
		descrs.push_back(frame_descrs);
	}
	fs << "keypoints" << "[" << keypoints << "]";
	fs << "descrs" << "[" << descrs << "]";
	fs.release();
}

bool size_icmp(const vector<DMatch>& i, const vector<DMatch>& j)
{
	return i.size()>j.size();
}

vector<vector<DMatch> >* getConnectedImgFeat(int n_imgs, vector<Mat> &descrs, FlannBasedMatcher& matcher, int i, int n_feats, int n_connected_imgs)
{
	vector<vector<DMatch> > *img_matches;
	img_matches = new vector<vector<DMatch> >(n_imgs);
	vector<vector<DMatch> > feat_matches;

	INFO("pairing %d frame", i);
	matcher.knnMatch (descrs[i], feat_matches, n_feats);

	// Group by imgIdx
	for (size_t j=0; j<feat_matches.size(); j++) {
		vector<DMatch>& candid = feat_matches[j];
		for (size_t k=0; k<candid.size(); k++) { // skip self
			int idx = candid[k].imgIdx;
			assert(idx >= 0 && idx <n_imgs);
			if (idx == i) // skip same frame
				continue;
			(*img_matches)[idx].push_back(candid[k]);
		}
	}

	sort(*img_matches, size_icmp);
	if (n_connected_imgs >= n_imgs)
		n_connected_imgs = n_imgs-1;

	img_matches->erase(img_matches->begin()+n_connected_imgs,
			  img_matches->end());
	for (int i=0; i<(int)img_matches->size(); i++) {
		INFO("%d->%d: %d", i, (*img_matches)[i][0].imgIdx, (int)(*img_matches)[i].size());
	}
	return img_matches;
}

vector<int> random_sample_idx(int max_idx, int n)
{
	assert(n>=0 && max_idx>=0);
	vector<int> samples;
	while(samples.size()!=(size_t)n) {
		int idx = randint(0, max_idx);
		// if idx already in samples, break
		size_t i;
		for (i=0; i<samples.size(); i++)
			if (samples[i]==idx) break;
		if (i!=samples.size())
			break;

		//cerr << idx << " ";
		samples.push_back(idx);
	}
//	cerr << endl;
	return samples;
}

void matches_idx_to_mat(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, vector<int>& idxes, Mat& src, Mat& dst)
{
	src = Mat((int)idxes.size(), 3, CV_32FC1);
	dst = Mat((int)idxes.size(), 2, CV_32FC1);
	for (int i=0; i<(int)idxes.size(); i++) {
		DMatch &dm = matches[idxes[i]];
		int src_feat_idx = dm.queryIdx;
		int dst_img_idx = dm.imgIdx; 
		int dst_feat_idx = dm.trainIdx;

		
		Point2f pt;
		pt = keypoints[src_img_idx].at<Point2f>(src_feat_idx, 0);
		src.at<float>(i, 0) = pt.x;
		src.at<float>(i, 1) = pt.y;
		src.at<float>(i, 2) = 1;
		pt = keypoints[dst_img_idx].at<Point2f>(dst_feat_idx, 0);
		dst.at<float>(i, 0) = pt.x;
		dst.at<float>(i, 1) = pt.y;
	}
}


/* src*H = dst
 */
Mat calc_homography(Mat& src, Mat &dst)
{
	//cout << "src" << src << endl;
	//cout << "dst" << dst << endl;
	return (src.t()*src).inv()*src.t()*dst;
}

void homo_ransac(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, float good_err, int good_n, int max_iter, vector<DMatch>& inliers, Mat homo)
{
	Mat best_model;
	double best_err = 1e41;
	vector<int> best_inliers;
	bool result = false;
	// some lazy way to construct whole mat
	vector<int> all_idx((int)matches.size());
	INFO("matches# = %d", matches.size());
	INFO("idx_size = %d", matches.size());
	for (size_t i=0; i<all_idx.size(); i++) {
		all_idx[i] = (int)i;
	}
	Mat all_src, all_dst;
	matches_idx_to_mat(src_img_idx, keypoints, matches, all_idx,
			all_src, all_dst);
	//showFeature(all_src, all_dst, src_img_idx, matches[0].imgIdx);
	cerr << "img_no " << src_img_idx << " " << matches[0].imgIdx << endl;

	for (int i=0; i<max_iter; i++) {
		vector<int> samples_idx;
		samples_idx = random_sample_idx ((int)matches.size(), 3);
		Mat src, dst;
		matches_idx_to_mat (src_img_idx, keypoints,
				matches, samples_idx, src, dst);

		Mat model = calc_homography(src, dst);

		Mat diff = all_src*model-all_dst;
		for (size_t j=0; j<(size_t)diff.rows; j++) {
			float x = diff.at<float>((int)j,0);
			float y = diff.at<float>((int)j,1);
			float err = x*x+y*y;
			if (err<good_err) {
				samples_idx.push_back((int)j);
			}
		}
		if (samples_idx.size() > (size_t)good_n) {
			matches_idx_to_mat (src_img_idx, keypoints,
					matches, samples_idx, src, dst);
			model = calc_homography(src, dst);
			diff = src*model-dst; //FIXME
			float total_err = 0;
			for (size_t j=0; j<(size_t)diff.rows; j++) {
				float x = diff.at<float>((int)j,0);
				float y = diff.at<float>((int)j,1);
				total_err += x*x+y*y;
			}
			int size = (int)samples_idx.size();
			total_err /= (float)(size*size);
			total_err += 20.f/(float)(size*size);
			if (total_err < best_err) {
				INFO("best err = %lf, #= %d", best_err, size);
				best_err = total_err;
				best_model = model;
				best_inliers = samples_idx;
				result = true;
			}
		}
	}
	for (size_t i=0; i<best_inliers.size(); i++) {
		inliers.push_back(matches[best_inliers[i]]);
	}
	if (result) {
		Mat src, dst;
		matches_idx_to_mat (src_img_idx, keypoints,
				matches, best_inliers, src, dst);
		showFeature(src, dst, src_img_idx, matches[0].imgIdx);
		cout << best_model << endl;
		homo = best_model;
	}
}

bool is_model_valid(int n_inlier, int n_matches)
{
	return n_inlier > 5.9+0.22*n_matches;
}

vector<vector<DMatch> >* validate_matches(int src_img_idx, vector<Mat>& keypoints, vector<vector<DMatch> >& matches)
{
	float good_err = 1e2;
	int good_n= 10;
	int max_iter = 10000;

	vector<vector<DMatch> > *valid_matches;
	valid_matches = new vector<vector<DMatch> >();
	for (size_t i=0; i<matches.size(); i++) {
		vector<DMatch> inliers, outliers;
		Mat homo;
		homo_ransac (src_img_idx, keypoints, matches[i], 
				good_err, good_n, max_iter,
				inliers, homo);
		if (is_model_valid((int)inliers.size(), (int)matches[i].size())) {
			valid_matches->push_back(inliers);
			INFO ("### validated ###");
		} else {
			INFO ("### out ###");
		}
	}
	return valid_matches;
}

void astich(int n_imgs, char** img_fnames)
{
	int n_feats = 4+1;
	int n_connected_imgs = 6;
	FlannBasedMatcher matcher;
	vector<Mat> keypoints;
	vector<Mat> descrs;

	srand(0);

	getAllFeatures (n_imgs, img_fnames, "feats.yml", keypoints, descrs);
	
	matcher.add (descrs);
	matcher.train ();

	vector<vector<DMatch> > matches;

	for (int i=0; i<n_imgs; i++) {
		vector<vector<DMatch> > *img_matches = getConnectedImgFeat(
			n_imgs, descrs, matcher, i, n_feats, n_connected_imgs);
		vector<vector<DMatch> > *valid_matches = validate_matches (i, keypoints, *img_matches);
		valid_matches = valid_matches;
	}


	return;
}

int main(int argc, char **argv)
{
	if (2>argc) {
		printf(	"%s:\tAutomatic Stiching Tool"
			"\t%s [PICTURES...]",
			argv[0], argv[0]);
		exit(0);
	}
	char **img_fnames = (char**)calloc(argc, sizeof(char*));
	IMG_FNAME = img_fnames; //FIXME
	int n_imgs = 0;
//	argc = 3;
	for (int i=1; i<argc; i++) {
		char *s = argv[i];
		if (0==strcmp(s, "")){
		}else{
			img_fnames[n_imgs] = strdup(argv[i]);
			INFO("%d img = %s", n_imgs, argv[i]);
			n_imgs++;
		}
	}

	astich(n_imgs, img_fnames);

	for (int i=0; i<n_imgs; i++)
		free(img_fnames[i]);
	free(img_fnames);

	return 0;
}
