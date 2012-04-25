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

template <class _T>
bool size_icmp(const vector<_T>& i, const vector<_T>& j)
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

	sort(*img_matches, size_icmp<DMatch>);
	if (n_connected_imgs >= n_imgs)
		n_connected_imgs = n_imgs-1;

	img_matches->erase(img_matches->begin()+n_connected_imgs,
			  img_matches->end());
	for (int i=0; i<(int)img_matches->size(); i++) {
		if (0==(int)(*img_matches)[i].size()) {
			img_matches->erase(img_matches->begin()+i, img_matches->end());
			break;
		}
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
	matches_idx_to_mat(src_img_idx, keypoints, matches, all_idxes, src, dst, dst_homo);
}

/* src*H = dst
 */
Mat calc_homography(Mat& src, Mat &dst)
{
	//cout << "src" << src << endl;
	//cout << "dst" << dst << endl;
	return (src.t()*src).inv()*src.t()*dst;
}

bool homo_ransac(int src_img_idx, vector<Mat>& keypoints, vector<DMatch>& matches, float good_err, int good_n, int max_iter, vector<DMatch>& inliers, Mat homo)
{
	Mat best_model;
	double best_err = 1e41;
	vector<int> best_inliers;
	bool result = false;
	// some lazy way to construct whole mat
	vector<int> all_idx((int)matches.size());
	INFO("matches# = %d", matches.size());
	INFO("idx_size = %d", matches.size());
	Mat all_src, all_dst;
	matches_to_mat(src_img_idx, keypoints, matches, all_src, all_dst);
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
			total_err += 40.f/(float)(size*size);
			if (total_err < best_err) {
				best_err = total_err;
				best_model = model;
				best_inliers = samples_idx;
				result = true;
				INFO("best err = %lf, #= %d", best_err, size);
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
	return result;
}

bool is_model_valid(int n_inlier, int n_matches)
{
	return n_inlier > 5.9+0.22*n_matches;
//	FIXME calculate ``overlaped area''
}

vector<vector<DMatch> >* validate_neighbors(int src_img_idx, vector<Mat>& keypoints, vector<vector<DMatch> >& neighbors)
{
	float good_err = 1e3;
	int good_n= 10;
	int max_iter = 10000;

	vector<vector<DMatch> > *valid_matches;
	valid_matches = new vector<vector<DMatch> >();
	for (size_t i=0; i<neighbors.size(); i++) {
		vector<DMatch> inliers, outliers;
		Mat homo;
		bool result = homo_ransac (src_img_idx, keypoints, neighbors[i], 
				good_err, good_n, max_iter,
				inliers, homo);
		if (!result) {
			continue;
		}
		if (is_model_valid((int)inliers.size(), (int)neighbors[i].size())) {
			valid_matches->push_back(inliers);
			INFO ("### validated ###");
		} else {
			INFO ("### out ###");
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
	}
	if (abs(de)>1e-5) {
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

	for (int i=0; i<max_iter; i++) {
		for (size_t j=0; j<groups.size(); j++) {
			vector<int> &group = groups[j];
			for (size_t k=1; k<group.size(); k++) {
				int idx = group[k];
				float err = solve_Hi_once(graph, idx, homos);
				INFO("err: %f", err);
			}
		}
	}
}

void update_min_max(float* v, Mat& homo, float &min_x, float &min_y, float &max_x, float &max_y)
{
	Mat pt = Mat(1, 3, CV_32FC1, v);
	pt *= homo;
	cerr << pt << endl;
	float x = pt.at<float>(0,0);
	float y = pt.at<float>(0,1);

	if (min_x>x) {
		min_x = x;
	}
	if (min_y>y) {
		min_y = y;
	}
	if (max_x<x) {
		max_x = x;
	}
	if (max_y<y) {
		max_y = y;
	}
}

void stroke(Mat& m, Mat &w, int x, int y, Vec3f v)
{
	for (int i=-1; i<=1; i++) {
		for (int j=-1; j<=1; j++) {
			float weight = 1;//-0.25f*(float)abs(i)-0.25f*(float)abs(j);
			m.at<Vec3f>(x+i,y+j) = v*weight;
//			w.at<float>(x+i,y+j) += weight;
		}
	}
}

void blending(vector<Mat>& homos, vector<vector<int> >& groups)
{
	for (size_t i=0; i<groups.size(); i++) {
		vector<int> &group = groups[i];
		// Get canvas size
		float min_x = 1e11f, min_y = 1e11f;
		float max_x = -1e11f, max_y = -1e11f;
		for (size_t j=0; j<group.size(); j++) {
			int idx = group[j];
			Mat img = imread(IMG_FNAME[idx], 1);
			Mat homo = homos[idx];

			float _down_left[] = {0, 0, 1};
			update_min_max(_down_left, homo, min_x, min_y, max_x, max_y);
			float _down_right[] = {(float)img.cols, 0, 1};
			update_min_max(_down_right, homo, min_x, min_y, max_x, max_y);
			float _up_left[] = {0, (float)img.rows, 1};
			update_min_max(_up_left, homo, min_x, min_y, max_x, max_y);
			float _up_right[] = {(float)img.cols, (float)img.rows, 1};
			update_min_max(_up_right, homo, min_x, min_y, max_x, max_y);
		}
		min_x-=1, min_y-=1;
		max_x+=1, max_y+=1;
		float ox=-min_x, oy = -min_y;
		int width = (int)(max_x-min_x)+1, height = (int)(max_y-min_y)+1;

		Mat out = Mat(height, width, CV_32FC3);
		Mat w = Mat(height, width, CV_32FC1);

		INFO("width = %d, height = %d", width, height);

		for (size_t j=0; j<group.size(); j++) {
			int idx = group[j];
			INFO("rendering %d", idx);
			Mat img_orig = imread(IMG_FNAME[idx], 1);
			Mat img;
			img_orig.convertTo(img, CV_32FC3, 1/255.);
			Mat homo = homos[idx];

			for (int m=0; m<img.cols; m++) {
				for (int n=0;  n<img.rows; n++) {
					Mat pt = Mat(1,3,CV_32FC1);
					pt.at<float>(0,0) = (float)m;
					pt.at<float>(0,1) = (float)n;
					pt.at<float>(0,2) = (float)1;
					pt *= homo;
					int x = (int)(pt.at<float>(0,0)+ox);
					int y = (int)(pt.at<float>(0,1)+oy);
					stroke(out, w, y, x, img.at<Vec3f>(n,m));
				}
			}

#if 0
			Mat oout = out.clone();
			for (int m=0; m<out.cols; m++) {
				for (int n=0; n<out.rows; n++) {
					float weight = w.at<float>(n,m);
					if (abs(weight) > 1e-3) {
						Vec3f v = oout.at<Vec3f>(n,m);
						v.val[0] /= weight;
						v.val[1] /= weight;
						v.val[2] /= weight;
						oout.at<Vec3f>(n,m) = v;
					}
				}
			}
			imshow("oresult", oout);
			imshow("w", w);
			waitKey(0);
#endif
			imshow("result", out);
			waitKey(0);
		}
		/*
		for (int m=0; m<out.cols; m++) {
			for (int n=0; n<out.rows; n++) {
				float weight = w.at<float>(n,m);
				if (abs(weight) > 1e-3) {
					Vec3f v = out.at<Vec3f>(n,m);
					v.val[0] /= weight;
					v.val[1] /= weight;
					v.val[2] /= weight;
					out.at<Vec3f>(n,m) = v;
				}
			}
		}
		*/
		out *= 255;
		char fname[255];
		sprintf(fname, "result%d.jpg", i);
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
