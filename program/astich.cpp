/* from http://www.opencv.org.cn/index.php */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

char **IMG_FNAME;
const float OO = 1e23f;
float CURV = (float)M_PI/10;
float IMG_WIDTH;
float IMG_HEIGHT;
Rect_<float> zero_border(OO, OO, 0, 0);

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

void CylinderProj(float &theta, float &h, float f, float x, float y)
{
	theta = atan2(x, f);
	h = y/sqrt(x*x+f*f);
}

void InvCylinderProj(float &x, float &y, float theta, float h, float f)
{
	x = f*tan(theta);
	y = h*sqrt(x*x+f*f);
}

void CylinderWrap(Mat &dst, Mat &src, Mat &mask)
{
	float curv = CURV;
	float f = (float)(src.cols/2./tan(curv));
	float s = (float)(src.cols/2. / curv);
	int width = src.cols, height = (int)(1.*src.rows/f*s);

	//cerr << height << " ; " << width << endl;
	dst = Mat(height, width, CV_8UC3);
	mask = Mat(height, width, CV_8UC1);
	
	int oi = -width/2, oj = -height/2;
	int ox = src.cols/2, oy = src.rows/2;
	for (int i=0; i<width; i++) {
		for (int j=0; j<height; j++) {
			float x, y;
			InvCylinderProj(x,y,(float)(i+oi)/s,(float)(j+oj)/s,f);
			int ix=(int)x+ox, iy=(int)y+oy;
	//		cerr << i << "," << j << " " << x << "," << y << endl;
			if (0<=ix && ix<src.cols && 0<=iy && iy<src.rows) {
				dst.at<Vec3b>(j,i) = src.at<Vec3b>(iy,ix);
				mask.at<uchar>(j,i) = 255;
			} else {
				mask.at<uchar>(j,i) = 0;
			}
		}
	}
}

void readAndProjImg(int idx, Mat& img, Mat& mask)
{
	if (-1==access(IMG_FNAME[idx], F_OK)) {
		INFO("File %s Not Found", IMG_FNAME[idx]);
		exit(0);
	}
	Mat raw_img = imread(IMG_FNAME[idx], 1);
	CylinderWrap(img, raw_img, mask);
	IMG_WIDTH = (float)img.cols;
	IMG_HEIGHT = (float)img.rows;
	return;
}

void getFeatures(Mat &img, Mat &mask, vector<KeyPoint>& keypoints, Mat& descrs)
{
	SiftFeatureDetector detector;
	detector.detect (img, keypoints, mask);
	
	SiftDescriptorExtractor extractor;
	extractor.compute (img, keypoints, descrs);
}

void showFeature(Mat& src, Mat& dst, int src_img_idx, int dst_img_idx)
{
	Mat src_img, src_mask; 
	Mat dst_img, dst_mask; 
	readAndProjImg(src_img_idx, src_img, src_mask);
	readAndProjImg(dst_img_idx, dst_img, dst_mask);
	Size size(src_img.cols+dst_img.cols, MAX(src_img.rows,dst_img.rows));
	Mat canvas = Mat(size, CV_MAKETYPE(src_img.depth(), 3));
	canvas = Scalar(0,0,0);
	Mat offset = Mat(2,1,CV_32FC1);
	Mat src_canvas = canvas(Rect(0,0,src_img.cols,src_img.rows));
	Mat dst_canvas = canvas(Rect(src_img.cols,0,dst_img.cols,dst_img.rows));
	src_img.copyTo(src_canvas, src_mask);
	dst_img.copyTo(dst_canvas, dst_mask);

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

		//line(canvas, st, ed, color);
	}
	namedWindow("canvas", CV_WINDOW_AUTOSIZE);
	imshow("canvas", canvas);
	waitKey(100);
	//destroyWindow("canvas");
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
	Mat img, mask;

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
		readAndProjImg(i, img, mask);

		frame_keypoints.clear();
		frame_keypoints_xy.clear();
		getFeatures(img, mask, frame_keypoints, frame_descrs);
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

template <class _T>
bool size_icmp(const vector<_T>& i, const vector<_T>& j)
{
	return i.size()>j.size();
}

vector<vector<DMatch> >* getConnectedImgFeat(int n_imgs, vector<Mat> &descrs, FlannBasedMatcher& matcher, int src_idx, int n_feats, int n_connected_imgs)
{
	vector<vector<DMatch> > *img_matches;
	img_matches = new vector<vector<DMatch> >(n_imgs);
	vector<vector<DMatch> > feat_matches;

	INFO("pairing %d frame", src_idx);
	matcher.knnMatch (descrs[src_idx], feat_matches, n_feats);

	// Group by imgIdx
	for (size_t j=0; j<feat_matches.size(); j++) {
		vector<DMatch>& candid = feat_matches[j];
		for (size_t k=0; k<candid.size(); k++) { // skip self
			int idx = candid[k].imgIdx;
			assert(idx >= 0 && idx <n_imgs);
			if (idx == src_idx) // skip same frame
				continue;
			img_matches->at(idx).push_back(candid[k]);
		}
	}

	sort(*img_matches, size_icmp<DMatch>);
	if (n_connected_imgs >= n_imgs)
		n_connected_imgs = n_imgs-1;

	img_matches->erase(img_matches->begin()+n_connected_imgs,
			  img_matches->end());
	INFO("");
	for (int j=0; j<(int)img_matches->size(); j++) {
		if (0==(int)img_matches->at(j).size()) {
			img_matches->erase(img_matches->begin()+j, img_matches->end());
			break;
		}
		INFO("%d->%d: %d", src_idx, img_matches->at(j)[0].imgIdx, (int)(*img_matches)[j].size());
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

void matches_idx_to_mat(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, vector<int>& idxes, Mat& src, Mat& dst, bool dst_homo=false)
{
	src = Mat((int)idxes.size(), 3, CV_32FC1);
	if (dst_homo) {
		dst = Mat((int)idxes.size(), 3, CV_32FC1);
	} else {
		dst = Mat((int)idxes.size(), 2, CV_32FC1);
	}
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
		if (dst_homo) {
			dst.at<float>(i, 2) = 1;
		}
	}
}

// Lazy method
void matches_to_mat(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, Mat& src, Mat& dst, bool dst_homo=false)
{
	vector<int> all_idxes;
	for (size_t i=0; i<matches.size(); i++) {
		all_idxes.push_back((int)i);
	}
	matches_idx_to_mat(src_img_idx, keypoints, matches, all_idxes,
			src, dst, dst_homo);
}

void update_rect(Mat &pt, Rect_<float> &rect)
{
	float x = pt.at<float>(0,0);
	float y = pt.at<float>(0,1);

	if (rect.x>x) {
		rect.x = x;
	}
	if (rect.y>y) {
		rect.y = y;
	}
	if (rect.width < x-rect.x) {
		rect.width = x-rect.x;
	}
	if (rect.height < y-rect.y) {
		rect.height = y-rect.y;
	}
}

bool inside_rect(float x, float y, Rect_<float> rect)
{
	return rect.x<=x && x<=rect.x+rect.width
		&& rect.y<=y && y<=rect.y+rect.height;
}

/* src*H = dst
 */
Mat calc_homography(Mat& src, Mat &dst)
{
	//cout << "src" << src << endl;
	//cout << "dst" << dst << endl;
	return (src.t()*src).inv()*src.t()*dst;
}

bool is_model_valid(int n_inlier, int n_region)
{
	return n_inlier > 5.9+0.22*n_region;
}

bool homo_ransac(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, float good_err, int good_n, int max_iter, vector<DMatch>& inliers, Mat homo)
{
	Mat best_model;
	float best_err = OO;
	vector<int> best_inliers;
	bool result = false;
	//INFO("matches# = %d", matches.size());
	Mat all_src, all_dst;
	matches_to_mat(src_img_idx, keypoints, matches, all_src, all_dst);
	//showFeature(all_src, all_dst, src_img_idx, matches[0].imgIdx);

	for (int i=0; i<max_iter; i++) {
		vector<int> samples_idx;
		samples_idx = random_sample_idx ((int)matches.size(), 3);
		Mat src, dst;
		matches_idx_to_mat (src_img_idx, keypoints,
				matches, samples_idx, src, dst);

		Mat model = calc_homography(src, dst);

		Mat diff = all_src*model-all_dst;
		Rect_<float> match_region = zero_border;
		for (int j=0; j<diff.rows; j++) {
			float x = diff.at<float>(j,0);
			float y = diff.at<float>(j,1);
			float err = x*x+y*y;
			if (err<good_err) {
				samples_idx.push_back(j);

				Mat pt = Mat(1, 2, CV_32FC1);
				pt.at<float>(0,0) = all_dst.at<float>(j,0);
				pt.at<float>(0,1) = all_dst.at<float>(j,1);
				update_rect(pt, match_region);
			}
		}
		int n_region = 0;
		for (int j=0; j<diff.rows; j++) {
			float x = all_dst.at<float>(j,0);
			float y = all_dst.at<float>(j,1);
			if (inside_rect(x, y, match_region)) {
				n_region++;
			}
		}
		int n_valid = (int)samples_idx.size();
		if (n_valid >= good_n && is_model_valid(n_valid, n_region)) {
			matches_idx_to_mat (src_img_idx, keypoints,
					matches, samples_idx, src, dst);
			model = calc_homography(src, dst);
			diff = src*model-dst;
			float total_err = 0;
			for (size_t j=0; j<(size_t)diff.rows; j++) {
				float x = diff.at<float>((int)j,0);
				float y = diff.at<float>((int)j,1);
				total_err += (float)(abs(x)+abs(y));
			}
			total_err /= (float)(n_valid);
			total_err += 40.f/(float)(n_valid);
			if (total_err < best_err) {
				best_err = total_err;
				best_model = model;
				best_inliers = samples_idx;
				result = true;
				//cerr << n_region << " " << n_valid << endl;
			}
		}
	}
	if (result) {
		for (size_t i=0; i<best_inliers.size(); i++) {
			inliers.push_back(matches[best_inliers[i]]);
		}

		Mat src, dst;
		matches_idx_to_mat (src_img_idx, keypoints,
				matches, best_inliers, src, dst);
		Mat model = calc_homography(src, dst);
		showFeature(src, dst, src_img_idx, matches[0].imgIdx);
		waitKey(10);

		cerr << "img_no " << src_img_idx 
			<< " " << matches[0].imgIdx << endl;
		INFO("best err = %lf, #= %d", best_err, best_inliers.size());
		homo = best_model;
	}
	return result;
}

vector<vector<DMatch> >* validate_neighbors(int src_img_idx, vector<Mat>& keypoints, vector<vector<DMatch> >& neighbors)
{
	float good_err = 200;
	int good_n= 50;
	int ransac_max_iter = 2000;

	vector<vector<DMatch> > *valid_matches;
	valid_matches = new vector<vector<DMatch> >();
	for (size_t i=0; i<neighbors.size(); i++) {
		vector<DMatch> inliers;
		Mat homo;
		bool result = homo_ransac (src_img_idx, keypoints, neighbors[i], 
				good_err, good_n, ransac_max_iter,
				inliers, homo);
		if (result) {
			//INFO ("### validated ###");
			valid_matches->push_back(inliers);
		} else {
			//INFO ("### out ###");
		}
	}
	return valid_matches;
}

struct MatchPair {
	int src_img_idx, dst_img_idx;
	Mat src_pt, dst_pt;
};

void flood_fill(vector<vector<MatchPair> >& graph, int idx, vector<bool>& visited, vector<int>& group, int &max_idx)
{
	int max_edges = 0;
	deque<int> q;

	visited[idx] = true;
	q.push_back(idx);
	while (!q.empty()) {
		idx = q[0];
		group.push_back(idx);
		if (max_edges<(int)graph[idx].size()) {
			max_edges = (int)graph[idx].size();
			max_idx = idx;
		}

		q.pop_front();
		for (size_t i=0; i<graph[idx].size(); i++) {
			int dst = graph[idx][i].dst_img_idx;
			if (!visited[dst]) {
				visited[dst] = true;
				q.push_back(dst);
			}
		}
	}
}

void make_undirected_graph(vector<vector<MatchPair> >& graph)
{
	for (size_t i=0; i<graph.size(); i++) {
		vector<MatchPair> &edges = graph[i];
		int src_img_idx = (int)i;
		for (size_t j=0; j<edges.size(); j++) {
			MatchPair &node = edges[j];
			int dst_img_idx = node.dst_img_idx;
			int k;
			vector<MatchPair> &dst_edges = graph[dst_img_idx];
			for (k=0; k<(int)dst_edges.size(); k++) {
				if (src_img_idx==dst_edges[k].dst_img_idx)
					break;
			}
			if (k==(int)dst_edges.size()) {
				MatchPair p;
				p.src_img_idx = dst_img_idx;
				p.dst_img_idx = src_img_idx;
				p.src_pt = node.dst_pt;
				p.dst_pt = node.src_pt;
				dst_edges.push_back(p);
			}
		}
	}
}

float update_Hi_pq(vector<vector<MatchPair> > &graph, int idx, vector<Mat> &homos, int p, int q)
{
	Mat &src_homo = homos[idx];
	vector<MatchPair> &edges = graph[idx];
	double nu=0, de=0, err=0;
	int n=0;
	for (size_t i=0; i<edges.size(); i++) {
		MatchPair &pair = edges[i];
		Mat &src = pair.src_pt;
		Mat &dst = pair.dst_pt;
		Mat &dst_homo = homos[pair.dst_img_idx];
		Mat diff = src*src_homo - dst*dst_homo;
		//cerr << "diff " << diff.rowRange(0, 10) << endl;
		for (int j=0; j<diff.rows; j++) {
			float d = diff.at<float>(j,q);
			float x = src.at<float>(j,p);
			nu += 1.*d*x;
			de += 1.*x*x;
			err += 1.*d*d;
		}
		n += diff.rows;
	}
	if (abs(de)>1e-5) {
		if (p!=2) {
			float c, of, c0 = 1e3;
			if (q==0) {
				c = IMG_WIDTH*IMG_WIDTH*c0;
			} else {
				c = IMG_HEIGHT*IMG_HEIGHT*c0;
			}
			if (p==q) {
				of = 1;
			} else {
				of = 0;
			}
			nu -= of*(float)n;
			de += c*(float)n;
		}
		src_homo.at<float>(p,q) -= (float)(nu/de);
	}
	return (float)err;
}

/* update the elements of H_i
 * 
 */
float solve_Hi_once(vector<vector<MatchPair> > &graph, int idx, vector<Mat> &homos)
{
	float err = 0;
	for (int p=0; p<3; p++) {
		for (int q=0; q<2; q++) {
			err += update_Hi_pq(graph, idx, homos, p, q);
		}
	}
	return err;
}


/* Bundle Adjustment using
 * Least Square Coordinate Descent
 *
 * Denote
 * 	* src_{i|m} as the matched query points of match m
 * 	* dst_{j|m} as the matched train points of match m
 * 	* H_i as the homographic transform i
 * 	* H_{j} as the homographic transform j
 * we minimize
 * 	\sum_{\forall matches m} (src_{i|m}*H_{i} - dst_{j|m}*H_{j})^2 
 * on
 * 	H_{i}
 * with a fixed staring point
 * 	H_{base} = [1 0; 0 1; 0 0]
 *
 * We suppose the images are geographically consistent on a plane.
 *
 */
void bundle_adjustment(vector<vector<MatchPair> >& graph, vector<Mat>& homos, vector<vector<int> >& groups, int max_iter)
{
	float _base_homo[] = {1,0, 0,1, 0,0};
	Mat base_homo = Mat(3, 2, CV_32FC1, _base_homo);
	homos = vector<Mat>(graph.size());
	for(size_t i=0; i<homos.size(); i++) {
		homos[i] = base_homo.clone();
	}

	make_undirected_graph(graph);

	// Find groups
	vector<bool> visited(graph.size(), false);
	for (int i=0; i<(int)graph.size(); i++) {
		if (0 == graph[i].size()) {
			continue;
		}
		int idx = graph[i][0].src_img_idx;
		if (visited[idx]) {
			continue;
		}
		vector<int> group;
		int base_idx;

		flood_fill(graph, idx, visited, group, base_idx);
		for (size_t i=0; i<group.size(); i++) {
			visited[group[i]] = false;
		}

		// BFS from group base
		group.clear();
		flood_fill(graph, base_idx, visited, group, base_idx);
		groups.push_back(group);
	}

	float err=0;
	for (int i=0; i<max_iter; i++) {
		for (size_t j=0; j<groups.size(); j++) {
			vector<int> &group = groups[j];
			for (size_t k=1; k<group.size(); k++) {
				int idx = group[k];
				err = solve_Hi_once(graph, idx, homos);
			}
		}
	}
	INFO("err: %f", err);
}

void get_homo_img_range(Mat &img, Mat &homo, Rect_<float> &rect)
{
	Mat pt;
	
	float _down_left[] = {0, 0, 1};
	pt = Mat(1, 3, CV_32FC1, _down_left);
	pt *= homo;
	update_rect(pt, rect);

	float _down_right[] = {(float)img.cols, 0, 1};
	pt = Mat(1, 3, CV_32FC1, _down_right);
	pt *= homo;
	update_rect(pt, rect);

	float _up_left[] = {0, (float)img.rows, 1};
	pt = Mat(1, 3, CV_32FC1, _up_left);
	pt *= homo;
	update_rect(pt, rect);

	float _up_right[] = {(float)img.cols, (float)img.rows, 1};
	pt = Mat(1, 3, CV_32FC1, _up_right);
	pt *= homo;
	update_rect(pt, rect);
}

void backward_homo_wrap(Mat &dst, Mat &src, Mat &mask, Mat &homo, Rect_<float> &dst_border)
{
	Rect_<float> border = zero_border;
	get_homo_img_range(src, homo, border);
	INFO("%f %f %f %f", border.x, border.y, border.width, border.height);
	int width = (int)border.width, height = (int)border.height;
	Mat inv_homo = Mat(homo, Rect(0,0,2,2)).inv();
	float ox = homo.at<float>(2,0);
	float oy = homo.at<float>(2,1);
	for (int i=0; i<width; i++) {
		for (int j=0; j<height; j++) {
			int nx = (int)(1.*i+border.x-dst_border.x);
			int ny = (int)(1.*j+border.y-dst_border.y);
			Mat pt = Mat(1,2, CV_32FC1);
			pt.at<float>(0,0) = (float)i+border.x-ox;
			pt.at<float>(0,1) = (float)j+border.y-oy;
			pt *= inv_homo;
			int ix = (int)pt.at<float>(0,0);
			int iy = (int)pt.at<float>(0,1);
			if (0<=ix && ix<src.cols && 0<=iy && iy<src.rows
				&& 0<=nx && nx< dst.cols
				&& 0<=ny && ny< dst.rows
				&& mask.at<uchar>(iy,ix)>0) {
				float *dst_v = dst.at<Vec3f>(ny,nx).val;
				float *src_v = src.at<Vec3f>(iy,ix).val;
				dst_v[0] = src_v[0];
				dst_v[1] = src_v[1];
				dst_v[2] = src_v[2];
			}
		}
	}
}

void blending(vector<Mat>& homos, vector<vector<int> >& groups)
{
	for (size_t i=0; i<groups.size(); i++) {
		vector<int> &group = groups[i];
		// Get canvas size
		Rect_<float> border = zero_border;
		for (size_t j=0; j<group.size(); j++) {
			int idx = group[j];
			Mat img, mask;
			readAndProjImg(idx, img, mask);
			Mat homo = homos[idx];
			cerr << homo << endl;

			get_homo_img_range(img, homo, border);
		}

		INFO("width = %f, height = %f", border.width, border.height);
		INFO("ox = %f, oy = %f", border.x, border.y);

		Mat out = Mat(border.size(), CV_32FC3);
		out = Scalar(255,0,0);

		for (size_t j=0; j<group.size(); j++) {
			int idx = group[j];
			INFO("rendering %d", idx);
			Mat raw_img, img, mask;
			readAndProjImg(idx, raw_img, mask);
			raw_img.convertTo(img, CV_32FC3, 1/255.);
			Mat homo = homos[idx];
			cerr << homo << endl;

			backward_homo_wrap(out, img, mask, homo, border);

			namedWindow("result", CV_WINDOW_NORMAL);
			imshow("result", out);
			waitKey(0);
		}
		out *= 255;
		char fname[255];
		sprintf(fname, "result%d.jpg", (int)i);
		imwrite(fname, out);
	}
}

void astich(int n_imgs, char** img_fnames)
{
	int n_feats = 4+1;
	int n_connected_imgs = 6;
	int bundle_max_iter = 500;
	FlannBasedMatcher matcher;
	vector<Mat> keypoints;
	vector<Mat> descrs;

	srand(0);

	getAllFeatures (n_imgs, img_fnames, "feats.yml", keypoints, descrs);
	
	matcher.add (descrs);
	matcher.train ();

	vector<vector<MatchPair> > graph;

	for (int i=0; i<n_imgs; i++) {
		vector<vector<DMatch> > *neighbors = getConnectedImgFeat(
			n_imgs, descrs, matcher, i, n_feats, n_connected_imgs);

		vector<vector<DMatch> > *valid_neighbors 
			= validate_neighbors (i, keypoints, *neighbors);

		MatchPair node;
		vector<MatchPair> node_edges;
		for (int j=0; j<(int)valid_neighbors->size(); j++) {
			matches_to_mat(i, keypoints, valid_neighbors->at(j), node.src_pt, node.dst_pt, true);
			node.src_img_idx = i;
			node.dst_img_idx = valid_neighbors->at(j).at(0).imgIdx;
			node_edges.push_back(node);
		}
		graph.push_back(node_edges);
	}

	vector<Mat> homos;
	vector<vector<int> > groups;
	bundle_adjustment(graph, homos, groups, bundle_max_iter);
	
	blending(homos, groups);


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
	//argc = 3;
	for (int i=1; i<argc; i++) {
		char *s = argv[i];
		if (0==strcmp(s, "-f")){
			float de;
			sscanf(argv[i+1], "%f", &de);
			CURV = (float)M_PI/de;
			i += 1;
		}else{
			img_fnames[n_imgs] = strdup(argv[i]);
			//Mat img, w;
			//readAndProjImg(n_imgs, img, w);
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
