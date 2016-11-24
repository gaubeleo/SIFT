#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <direct.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\nonfree\features2d.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

#define SIGMA 1.6
#define OCTAVE_COUNT 4
#define STEP_COUNT 3

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

#define NORM_VALUE 0.1

//////////////////////////////////////////////////////////////////////////////////

void cvSIFT(const Mat &img)
{
	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(img, keypoints);

	Mat output;
	drawKeypoints(img, keypoints, output, Scalar(0., 0., 255.), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("OPENCV_SIFT", output);
	waitKey();
	destroyAllWindows();
}

//////////////////////////////////////////////////////////////////////////////////
// Little Helpers

void printMat(const Mat &img){
	assert(img.type() == CV_64FC1);

	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			if (x == 0)
				cout << "[ ";
			cout << row[x];
			if (x != img.cols - 1)
				cout << ", ";
			else
				cout << " ]" << endl;
		}
	}
	cout << endl;
}

double sumMat(const Mat &img){
	assert(img.type() == CV_64FC1);

	double sum = 0.;
	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			sum += row[x];
		}
	}
	return sum;
}

Mat convertTo8bit(const Mat &img, bool color)
{
	assert(img.type() == CV_64FC1);
	Mat newImg, temp;
	img.copyTo(temp);
	temp *= 255.;
	temp.convertTo(newImg, CV_8UC1);

	if (color){
		Mat colorImg;
		vector<Mat> channels;
		channels.push_back(newImg);
		channels.push_back(newImg);
		channels.push_back(newImg);
		merge(channels, colorImg);
		return colorImg;
	}
	return newImg;
}

bool isIdentical(const Mat &img0, const Mat &img1){
	assert(img0.type() == CV_64FC1 && img1.type() == CV_64FC1);
	// throw error if Mats are not of same size instead of just false!
	assert(img0.rows == img1.rows && img0.cols == img1.cols);

	for (int y = 0; y < img0.rows; y++){
		const double *row0 = img0.ptr<double>(y);
		const double *row1 = img1.ptr<double>(y);
		for (int x = 0; x < img0.cols; x++){
			if (row0[x] != row1[x])
				return false;
		}
	}
	return true;
}

double absolute(double x){
	return x >= 0 ? x : -x;
}

bool isMin(double value, double y0, double y1, double y2)
{
	if (value >= y0)
		return false;
	if (value >= y1)
		return false;
	if (value >= y2)
		return false;
	return true;
}

bool isMax(double value, double y0, double y1, double y2)
{
	if (value <= y0)
		return false;
	if (value <= y1)
		return false;
	if (value <= y2)
		return false;
	return true;
}

double getMax(const Mat &img, bool ignoreCenter){
	assert(img.rows > 0 && img.cols > 0);

	double maxValue = DBL_MIN;
	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			if (ignoreCenter && y * 2 + 1 == img.rows && x * 2 + 1 == img.cols)
				continue;
			if (maxValue == DBL_MIN || row[x] > maxValue)
				maxValue = row[x];
		}
	}
	assert(maxValue != DBL_MIN);
	return maxValue;
}

double getMin(const Mat &img, bool ignoreCenter){
	assert(img.rows > 0 && img.cols > 0);

	double minValue = DBL_MAX;
	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			if (ignoreCenter && y * 2 + 1 == img.rows && x * 2 + 1 == img.cols)
				continue;
			if (minValue == DBL_MAX || row[x] < minValue)
				minValue = row[x];
		}
	}
	assert(minValue != DBL_MAX);
	return minValue;
}

Mat normalizeImage(const Mat &img){
	assert(img.type() == CV_64FC1);

	Mat normalizedImg;
	img.copyTo(normalizedImg);

	double maxValue = getMax(img, false);
	normalizedImg /= maxValue;
	//normalizedImg /= NORM_VALUE;

	return normalizedImg;
}

Mat absoluteImage(const Mat &img)
{
	assert(img.type() == CV_64FC1);

	Mat absoluteImg(img.rows, img.cols, img.type());
	for (int y = 0; y < img.rows; y++){
		double *rowAbs = absoluteImg.ptr<double>(y);
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			rowAbs[x] = absolute(row[x]);
		}
	}

	return absoluteImg;
}

Mat doubleImage(const Mat &img)
{
	assert(img.type() == CV_64FC1);

	Mat doubledImg;
	resize(img, doubledImg, Size(), 2, 2);

	return doubledImg;
}

//////////////////////////////////////////////////////////////////////////////////
// basic Image input/output

void showImg(const Mat &img, string title)
{
	imshow(title, img);
	waitKey();
	destroyAllWindows();
}

void showImg(const Mat &img)
{
	showImg(img, "SIFT");
}

// Load an image from a given path using OpenCV 
// use IMREAD_COLOR or IMREAD_GRAYSCALE as flags
Mat loadImg(const string directory, const string filename, int flags){
	string fullFilename = string(directory + "\\" + filename);
	Mat image;
	image = imread(fullFilename, flags);

	if (!image.data){
		cout << "image file " << fullFilename << " could not be opened" << endl;
		getchar();
		exit(-1);
	}

	return image;
}

// saves an image to the specified path
bool saveImg(const string directory, const string filename, const Mat &img){
	string fullFilename = string(directory + "\\" + filename);
	struct stat sb;

	if (!(stat(directory.c_str(), &sb) == 0 && sb.st_mode == S_IFDIR)){
		_mkdir(directory.c_str());
	}

	cout << "successfully written '" << fullFilename << "' to file!" << endl;

	return imwrite(fullFilename, img);
}

//////////////////////////////////////////////////////////////////////////////////
// Filters

// apply a filter on a single patch of an image, thus kernel size and value size need to be of same dimensions
// the calculated value gets stored at the location of pixel
void _filter(double* pixel, const Mat &values, const Mat &kernel){
	assert(values.channels() == 1);
	assert(values.size() == kernel.size());
	assert(kernel.type() == CV_64FC1);
	assert(values.type() == CV_64FC1);

	double value = 0;
	for (int y = 0; y < kernel.rows; y++){
		const double *rowKernel = kernel.ptr<double>(y);
		const double *rowValues = values.ptr<double>(y);
		for (int x = 0; x < kernel.cols; x++){
			value += rowKernel[x] * (double)rowValues[x];
		}
	}
	*pixel = value;
}

// filters an entire image (in CV_64FC1 format) using a given kernel
Mat filter(const Mat &img, const Mat &kernel){
	assert(kernel.rows % 2 == 1 && kernel.cols % 2 == 1);
	assert(kernel.type() == CV_64FC1);
	assert(img.type() == CV_64FC1);

	int yOffset = (kernel.rows - 1) / 2;
	int xOffset = (kernel.cols - 1) / 2;

	Mat filteredImg(img.rows, img.cols, CV_64FC1, Scalar(0.));

	for (int y = 0; y < img.rows; y++){
		//copy border from original Image
		if (y < yOffset || y >= img.rows - yOffset){
			img.row(y).copyTo(filteredImg.row(y));
			continue;
		}

		double *row = filteredImg.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			//copy border from original Image
			if (x < xOffset || x >= img.cols - xOffset){
				row[x] = img.at<double>(y, x);
				continue;
			}

			const Mat tmp = img(Rect(x - xOffset, y - yOffset, kernel.cols, kernel.rows));
			_filter(&row[x], tmp, kernel);
		}
	}
	return filteredImg;
}

void _maxFilter(double* pixel, const Mat &values)
{
	double value = getMax(values, false);
	*pixel = value;
}

Mat maxFilter(const Mat &img, int kernelHeight, int kernelWidth)
{
	assert(kernelHeight % 2 == 1 && kernelWidth % 2 == 1);
	assert(img.type() == CV_64FC1);

	int yOffset = (kernelHeight - 1) / 2;
	int xOffset = (kernelWidth - 1) / 2;

	Mat filteredImg(img.rows, img.cols, CV_64FC1, Scalar(0.));

	for (int y = 0; y < img.rows; y++){
		double *row = filteredImg.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			int x0 = max(x - xOffset, 0);
			int y0 = max(y - yOffset, 0);
			int w = min(img.cols-x0, kernelWidth);
			int h = min(img.rows-y0, kernelHeight);
			Rect rect(x0, y0, w, h);
			const Mat tmp = img(rect);
			_maxFilter(&row[x], tmp);
		}
	}
	return filteredImg;
}

void _minFilter(double* pixel, const Mat &values)
{
	double value = getMin(values, false);
	*pixel = value;
}

Mat minFilter(const Mat &img, int kernelHeight, int kernelWidth)
{
	assert(kernelHeight % 2 == 1 && kernelWidth % 2 == 1);
	assert(img.type() == CV_64FC1);

	int yOffset = (kernelHeight - 1) / 2;
	int xOffset = (kernelWidth - 1) / 2;

	Mat filteredImg(img.rows, img.cols, CV_64FC1, Scalar(0.));

	for (int y = 0; y < img.rows; y++){
		double *row = filteredImg.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			int x0 = max(x - xOffset, 0);
			int y0 = max(y - yOffset, 0);
			int w = min(img.cols - x0, kernelWidth);
			int h = min(img.rows - y0, kernelHeight);
			Rect rect(x0, y0, w, h);
			const Mat tmp = img(rect);
			_minFilter(&row[x], tmp);
		}
	}
	return filteredImg;
}

/////////////////////////////////////////////////////////////////////////////

// scales an image up by a given factor
Mat upscale(const Mat &img, int factor){
	assert(img.channels() == 1);
	assert(img.type() == CV_64FC1);

	Mat scaledImg(img.rows * factor, img.cols * factor, img.type());

	if (factor == 1){
		img.copyTo(scaledImg);
		return scaledImg;
	}

	for (int y = 0; y < scaledImg.rows; y++){
		double *rowScale = scaledImg.ptr<double>(y);
		const double *rowOrg = img.ptr<double>(y / factor);

		for (int x = 0; x < scaledImg.cols; x++){
			rowScale[x] = rowOrg[x / factor];
		}
	}

	return scaledImg;
}

// scales an image down by a given factor
Mat downscale(const Mat &img, int factor){
	assert(img.channels() == 1);
	assert(img.type() == CV_64FC1);

	Mat scaledImg(img.rows / factor, img.cols / factor, img.type());

	if (factor == 1){
		img.copyTo(scaledImg);
		return scaledImg;
	}

	for (int y = 0; y < scaledImg.rows; y++){
		double *rowScale = scaledImg.ptr<double>(y);
		const double *rowOrg = img.ptr<double>(y*factor);

		for (int x = 0; x < scaledImg.cols; x++){
			rowScale[x] = rowOrg[x * factor];
		}
	}

	return scaledImg;
}

// returns a single GaussianKernel for a specific sigma
Mat createGaussianKernel(double sigma){
	assert(FILTER_WIDTH % 2 == 1 && FILTER_HEIGHT % 2 == 1);

	Mat kernel(FILTER_HEIGHT, FILTER_WIDTH, CV_64FC1);

	int yOffset = (FILTER_HEIGHT - 1) / 2;
	int xOffset = (FILTER_WIDTH - 1) / 2;

	double value, sum = 0.;
	for (int y = -yOffset; y <= yOffset; y++){
		double *row = kernel.ptr<double>(y + yOffset);
		for (int x = -xOffset; x <= xOffset; x++){
			value = (1. / (2.*M_PI*sigma*sigma)) * pow(M_E, (-(x*x + y*y) / (2 * sigma*sigma)));
			row[x + xOffset] = value;
			sum += value;
		}
	}
	kernel /= sum;

	return kernel;
}

// returns GaussianKernels for various sigma 
vector<Mat> createGaussianKernels(){
	cout << "\tcreating GaussianKernels " << endl;
	vector<Mat> kernels;
	Mat kernel;
	for (int s = 0; s < STEP_COUNT + 3; s++){
		double k = pow(2, (double)s / (double)STEP_COUNT);
		kernel = createGaussianKernel(k*SIGMA);
		kernels.push_back(kernel);
	}
	return kernels;
}

// returns GaussianPyramid [OCTAVECOUNT][STEPCOUNT+2](various image sizes betweeen octaves, but same within single octave)
vector<vector<Mat>> createGaussianPyramid(const Mat &img){
	cout << "creating Gaussian Pyramid: " << endl;
	vector<vector<Mat>> gp(OCTAVE_COUNT);
	
	Mat grayImg;
	if (img.channels() == 1)
		img.copyTo(grayImg);
	else if (img.channels() == 3){
		cv::cvtColor(img, grayImg, CV_BGR2GRAY);
	}
	else {
		cout << "image channel count not supported: " << img.channels() << endl;
		exit(-1);
	}
	grayImg.convertTo(grayImg, CV_64FC1);
	grayImg /= 255.;

	//grayImg = doubleImage(grayImg);
	GaussianBlur(grayImg, grayImg, Size(), 0.5); // 1. if image_doubling was used

	//const vector<Mat> kernels = createGaussianKernels();

	cout << "\tfiltering the image on different GaussianKernels" << endl;
	Mat octaveImg;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		vector<Mat> &octave = gp.at(o);
		octave.resize(STEP_COUNT + 3);
		for (int s = 0; s < STEP_COUNT + 3; s++){
			Mat &bluredImg = octave.at(s);
			if (s == 0){
				if (o == 0)
					octaveImg = grayImg;
				else
					octaveImg = downscale(gp.at(o-1).at(STEP_COUNT), 2);
				bluredImg = octaveImg;
			}
			else{
				//printMat(kernels.at(s));
				//filter2D(octaveImg, bluredImg, -1, kernels.at(s));
				double k = pow(2, (double)s / (double)STEP_COUNT);
				GaussianBlur(octaveImg, bluredImg, Size(), k*SIGMA);
				//bluredImg = filter(octaveImg, kernels.at(s));	
				//filter2D(octaveImg, temp, -1, kernels.at(s-1));
				//isIdentical(temp, bluredImg);
			}
		}
	}
	return gp;
}

void showGaussianPyramid(const vector<vector<Mat>> &gp, bool scaleUp){
	cout << "displaying Gaussian Pyramid:" << endl;
	assert(gp.size() == OCTAVE_COUNT);

	Mat img;
	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		const vector<Mat> octave = gp.at(o);

		assert(octave.size() == STEP_COUNT + 3);

		for (unsigned s = 0; s < octave.size(); s++){
			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;
			octave.at(s).copyTo(img);
			if (scaleUp)
				img = upscale(img, 1 << o);

			imshow("Gaussian Pyramid", img);
			waitKey();
			destroyAllWindows();
		}
	}
}

Mat calcDifference(const Mat &img0, const Mat &img1){
	assert(img0.rows == img1.rows && img0.cols == img1.cols);
	assert(img0.type() == img1.type() && img0.type() == CV_64FC1);

	Mat difference(img0.rows, img0.cols, CV_64FC1);

	for (int y = 0; y < difference.rows; y++){
		double *rowDiff = difference.ptr<double>(y);
		const double *row0 = img0.ptr<double>(y);
		const double *row1 = img1.ptr<double>(y);
		for (int x = 0; x < difference.cols; x++){
			rowDiff[x] = row0[x] - row1[x];
		}
	}
	return difference;
}

vector<vector<Mat>> createDifferenceOfGaussians(const vector<vector<Mat>> &gp){
	cout << "creating Difference of Gaussians: " << endl;
	vector<vector<Mat>> DoG;

	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		//cout << "\toctave: " << o << endl;
		vector<Mat> octaveDoG;
		const vector<Mat> octaveGP = gp.at(o);
		for (unsigned s = 0; s < octaveGP.size() - 1; s++){
			//cout << "\t\tstep: " << s << endl;
			//cout << isIdentical(octaveGP.at(s), octaveGP.at(s + 1)) << endl;
			Mat difference = calcDifference(octaveGP.at(s), octaveGP.at(s + 1));
			octaveDoG.push_back(difference);
		}
		DoG.push_back(octaveDoG);
	}
	return DoG;
}

// could be same method as showGaussianPyramid except for assertions
void showDifferenceOfGaussians(const vector<vector<Mat>> &DoG, bool scaleUp, bool normalize){
	cout << "displaying Difference of Gaussians:" << endl;
	assert(DoG.size() == OCTAVE_COUNT);

	Mat img;
	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		const vector<Mat> octave = DoG.at(o);

		assert(octave.size() == STEP_COUNT + 2);

		for (unsigned s = 0; s < octave.size(); s++){
			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;

			octave.at(s).copyTo(img);
			if (scaleUp)
				img = upscale(img, 1 << o);
			if (normalize)
				img = normalizeImage(img);
				
			imshow("DoG", img);
			waitKey();
			destroyAllWindows();
		}
	}
}

void filterExtrema(const vector<vector<Mat>>& DoG, vector<vector<Mat>>* minima, vector<vector<Mat>>* maxima)
{
	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		double scale = (1 << o);

		const vector<Mat> octaveDoG = DoG.at(o);

		vector<Mat> octaveMin;
		vector<Mat> octaveMax;

		for (unsigned s = 0; s < octaveDoG.size(); s++)
		{
			const Mat diff = octaveDoG.at(s);

			Mat minImg = minFilter(diff, 3, 3);
			Mat maxImg = maxFilter(diff, 3, 3);

			octaveMin.push_back(minImg);
			octaveMax.push_back(maxImg);
		}

		minima->push_back(octaveMin);
		maxima->push_back(octaveMax);
	}
}

// calculate all Minima and Maximas in DoG
vector<vector<vector<KeyPoint>>> calcExtrema(const vector<vector<Mat>>& DoG, const vector<vector<Mat>> &minima, const vector<vector<Mat>> &maxima)
{
	cout << "calculating Extrema in DoG: " << endl;
	vector<vector<vector<KeyPoint>>> extrema(OCTAVE_COUNT);

	KeyPoint extremum;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		cout << "\toctave: " << o << endl;
		int scale = (1 << o);

		const vector<Mat> octaveDoG = DoG.at(o);
		const vector<Mat> octaveMin = minima.at(o);
		const vector<Mat> octaveMax = maxima.at(o);

		vector<vector<KeyPoint>> &octaveExtrema = extrema.at(o);
		octaveExtrema.resize(STEP_COUNT);

		for (unsigned s = 1; s < octaveDoG.size() - 1; s++)
		{
			cout << "\t\tstep: " << s << endl;
			double k = pow(2, (double)s / (double)STEP_COUNT);

			const Mat diff = octaveDoG.at(s);

			const Mat min0 = octaveMin.at(s-1);
			const Mat min2 = octaveMin.at(s+1);

			const Mat max0 = octaveMax.at(s-1);
			const Mat max2 = octaveMax.at(s+1);

			vector<KeyPoint> &stepExtrema = octaveExtrema.at(s-1);

			int m = 0, mm = 0;
			for (int y = 1; y < diff.rows-1; y++)
			{
				const double *rowDiff = diff.ptr<double>(y);

				const double *rowMin0 = min0.ptr<double>(y);
				const double *rowMin2 = min2.ptr<double>(y);

				const double *rowMax0 = max0.ptr<double>(y);
				const double *rowMax2 = max2.ptr<double>(y);

				for (int x = 1; x < diff.cols-1; x++)
				{
					double diffValue = rowDiff[x];

					const Rect rect(x-1, y-1, 3, 3);
					const Mat temp = diff(rect);
					double minValue1 = getMin(temp, true);
					double maxValue1 = getMax(temp, true);

					/*if (diffValue > maxValue1 && diffValue > rowMax0[x])
					{
						cout << diffValue << ", " << rowMax0[x] << endl;
					}*/

					if (isMin(diffValue, rowMin0[x], minValue1, rowMin2[x]))
					{
						mm++;
						double sigma = scale * k * SIGMA;
						//cout << x << ", " << scale << ", " << diff.rows << endl;
						extremum = KeyPoint(float(x*scale + scale/2), float(y*scale + scale/2), 3.*scale, -1, 0, o, -1);
						stepExtrema.push_back(extremum);
					}
					else if ((isMax(diffValue, rowMax0[x], maxValue1, rowMax2[x])))
					{
						m++;
						extremum = KeyPoint(x*scale + scale / 2, y*scale + scale / 2, 8 * scale, -1, 0, o, 1);
						stepExtrema.push_back(extremum);
					}
				}
			}
			cout << m<< ", " << mm << endl;
		}
	}
	return extrema;
}

Mat drawKeypoints(const Mat &img, const vector<KeyPoint> keypoints){
	Mat keypointImg;
	if (img.channels() == 3 && img.type() != CV_64FC1)
		keypointImg = img;
	else if (img.type() == CV_64FC1)
		keypointImg = convertTo8bit(img, true);
	else
		assert(false);

	drawKeypoints(keypointImg, keypoints, keypointImg, Scalar(0., 0., 255.), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	return keypointImg;
}

Mat drawKeypoints(const Mat &img, const vector<vector<vector<KeyPoint>>> keypoints)
{
	Mat output;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		const vector<vector<KeyPoint>> octaveKP = keypoints.at(o);
		for (unsigned s = 0; s < octaveKP.size(); s++){
			const vector<KeyPoint> &KP = octaveKP.at(s);
			output = drawKeypoints(img, KP);
		}
	}

	return output;
}

// could be same method as showGaussianPyramid except for assertions
void showDoGkeypoints(const vector<vector<Mat>> &DoG, const vector<vector<vector<KeyPoint>>> keypoints){
	cout << "displaying Difference of Gaussians:" << endl;
	assert(DoG.size() == OCTAVE_COUNT);

	Mat img, keypointImg;
	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		const vector<Mat> octave = DoG.at(o);
		const vector<vector<KeyPoint>> octaveKP = keypoints.at(o);

		assert(octave.size() == STEP_COUNT + 2);
		assert(octaveKP.size() == STEP_COUNT);

		for (unsigned s = 0; s < octave.size(); s++){

			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;

			octave.at(s).copyTo(img);
			img = absoluteImage(img);
			img = normalizeImage(img);
			img = upscale(img, 1 << o);

			if (s == 0 || s == STEP_COUNT + 1);
				//showImg(img, "DOG keypoints");
			else{
				const vector<KeyPoint> &KP = octaveKP.at(s - 1);
				keypointImg = drawKeypoints(img, KP);
				showImg(keypointImg, "DOG keypoints");
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	//Mat img = loadImg("src", "lenna.jpg", IMREAD_COLOR);
	Mat img = loadImg("src", "Testimage_gradients.jpg", IMREAD_COLOR);
	//Mat img = loadImg("src", "photographer.jpg", IMREAD_COLOR);
	//Mat img = loadImg("src", "line.png", IMREAD_COLOR);

	//cvSIFT(img);

	//resize(img, img, Size(256, 256));

	vector<vector<Mat>> gp = createGaussianPyramid(img);
	//showGaussianPyramid(gp, true);

	vector<vector<Mat>> DoG = createDifferenceOfGaussians(gp);
	//showDifferenceOfGaussians(DoG, true, true);

	vector<vector<Mat>> minima;
	vector<vector<Mat>> maxima;

	filterExtrema(DoG, &minima, &maxima);

	//showDifferenceOfGaussians(minima, true, true);
	//showDifferenceOfGaussians(maxima, true, true);

	vector<vector<vector<KeyPoint>>> extrema = calcExtrema(DoG, minima, maxima);
	showImg(drawKeypoints(img, extrema));
	showDoGkeypoints(DoG, extrema);

	return 0;
}
