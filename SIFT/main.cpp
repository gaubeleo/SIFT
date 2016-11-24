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
#define STEP_COUNT 2
//#define MIN_THRESHOLD 0.001

#define FILTER_WIDTH 9
#define FILTER_HEIGHT 9

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
	double maxValue = -1.; 
	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			if (ignoreCenter && y * 2 + 1 == img.rows && x * 2 + 1 == img.cols)
				continue;
			if (row[x] > maxValue || maxValue == -1.)
				maxValue = row[x];
		}
	}
	assert(maxValue >= 0);
	return maxValue;
}

double getMin(const Mat &img, bool ignoreCenter){
	double minValue = -1.;
	for (int y = 0; y < img.rows; y++){
		const double *row = img.ptr<double>(y);
		for (int x = 0; x < img.cols; x++){
			if (ignoreCenter && y * 2 + 1 == img.rows && x * 2 + 1 == img.cols)
				continue;
			if (row[x] < minValue || minValue == -1.)
				minValue = row[x];
		}
	}
	assert(minValue >= 0);
	return minValue;
}

Mat normalizeImage(const Mat &img){
	assert(img.type() == CV_64FC1);

	Mat normalizedImg;
	img.copyTo(normalizedImg);

	double maxValue = getMax(img, false);

	normalizedImg /= maxValue;

	return normalizedImg;
}

//////////////////////////////////////////////////////////////////////////////////
// basic Image input/output

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
	for (int s = 0; s < STEP_COUNT + 2; s++){
		double k = pow(2, (double)s / (double)STEP_COUNT);
		kernel = createGaussianKernel(k*SIGMA);
		kernels.push_back(kernel);
	}
	return kernels;
}

// returns GaussianPyramid [OCTAVECOUNT][STEPCOUNT+2](various image sizes betweeen octaves, but same within single octave)
vector<vector<Mat>> createGaussianPyramid(const Mat &img){
	cout << "creating Gaussian Pyramid: " << endl;
	vector<vector<Mat>> gp;
	
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

	const vector<Mat> kernels = createGaussianKernels();

	cout << "\tfiltering the image on different GaussianKernels" << endl;
	Mat octaveImg, bluredImg, temp;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		//cout << "\toctave: " << o << endl;
		vector<Mat> octave;
		octaveImg = downscale(grayImg, 1<<o); // downscale original image by 2^o
		if (o == 0)
			octave.push_back(octaveImg); // include original image in first octave
		for (int s = 0; s < STEP_COUNT + 2; s++){
			//cout << "\t\tstep: " << s << endl;
			//printMat(kernels.at(s));
			//filter2D(octaveImg, bluredImg, -1, kernels.at(s));
			bluredImg = filter(octaveImg, kernels.at(s));	
			//filter2D(octaveImg, temp, -1, kernels.at(s-1));
			//isIdentical(temp, bluredImg);
			octave.push_back(bluredImg);
		}
		gp.push_back(octave);
	}
	return gp;
}

void showGaussianPyramid(const vector<vector<Mat>> &gp, bool scaleUp){
	cout << "displaying Gaussian Pyramid:" << endl;
	assert(gp.size() == OCTAVE_COUNT);

	Mat img;
	for (unsigned o = 0; o < OCTAVE_COUNT; o++){
		const vector<Mat> octave = gp.at(o);

		if (o == 0)
			assert(octave.size() == STEP_COUNT + 3);
		else 
			assert(octave.size() == STEP_COUNT + 2);

		for (unsigned s = 0; s < octave.size(); s++){
			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;
			octave.at(s).copyTo(img);
			if (scaleUp)
				img = upscale(img, 1 << (o + 1));

			imshow("GP at o=" + to_string(o) + ", s=" + to_string(sigma), img);
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
			rowDiff[x] = absolute(row0[x] - row1[x]);
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

		if (o == 0)
			assert(octave.size() == STEP_COUNT + 2);
		else
			assert(octave.size() == STEP_COUNT + 1);

		for (unsigned s = 0; s < octave.size(); s++){
			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;

			octave.at(s).copyTo(img);
			if (scaleUp)
				img = upscale(img, 1 << (o + 1));
			if (normalize)
				img = normalizeImage(img);
				
			imshow("DoG at o=" + to_string(o) + ", s=" + to_string(sigma), img);
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


struct Extremum
{
	int y;
	int x;
	double sigma;
	bool isMax;
};

// calculate all Minima and Maximas in DoG
vector<Extremum> calcExtrema(const vector<vector<Mat>>& DoG, const vector<vector<Mat>> &minima, const vector<vector<Mat>> &maxima)
{
	cout << "calculating Extrema in DoG: " << endl;
	vector<Extremum> extrema;

	Extremum extremum;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		cout << "\toctave: " << o << endl;
		int scale = (1 << o);

		const vector<Mat> octaveDoG = DoG.at(o);
		const vector<Mat> octaveMin = minima.at(o);
		const vector<Mat> octaveMax = maxima.at(o);

		for (unsigned s = 1; s < octaveDoG.size() - 1; s++)
		{
			cout << "\t\tstep: " << s << endl;
			double k = pow(2, (double)s / (double)STEP_COUNT);

			const Mat diff = octaveDoG.at(s);

			const Mat min0 = octaveMin.at(s-1);
			const Mat min2 = octaveMin.at(s+1);

			const Mat max0 = octaveMax.at(s-1);
			const Mat max2 = octaveMax.at(s+1);

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

					//if (diffValue > maxValue1 && diffValue > rowMax0[x])
					//{
					//	cout << diffValue << ", " << rowMax0[x] << endl;
					//}

					if (isMin(diffValue, rowMin0[x], minValue1, rowMin2[x]))
					{
						double sigma = scale * k * SIGMA;
						extremum = { scale*y + scale / 2, scale*x + scale / 2, sigma, false };
						extrema.push_back(extremum);
					}
					else if (isMax(diffValue, rowMax0[x], maxValue1, rowMax2[x]))
					{
						double sigma = scale * k * SIGMA;
						extremum = { scale*y + scale / 2, scale*x + scale / 2, sigma, true };
						extrema.push_back(extremum);
					}
				}
			}
		}
	}
	return extrema;
}

void showKeypoints(const Mat &img, const vector<Extremum>keypoints){
	
	for (Extremum keypoint : keypoints){
		
	}
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	Mat img = loadImg("src", "lenna.jpg", IMREAD_COLOR); //IMREAD_COLOR

	SiftFeatureDetector detector;
	vector<KeyPoint> keypoints;
	detector.detect(img, keypoints);

	// Add results to image and save.
	Mat output;
	drawKeypoints(img, keypoints, output, Scalar(0., 0., 255.), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("OPENCV_SIFT", output);
	waitKey();
	destroyAllWindows();

	resize(img, img, Size(256, 256));

	vector<vector<Mat>> gp = createGaussianPyramid(img);
	//showGaussianPyramid(gp, true);

	vector<vector<Mat>> DoG = createDifferenceOfGaussians(gp);
	//showDifferenceOfGaussians(DoG, true, true);

	vector<vector<Mat>> minima;
	vector<vector<Mat>> maxima;

	filterExtrema(DoG, &minima, &maxima);

	showDifferenceOfGaussians(minima, true, true);
	showDifferenceOfGaussians(maxima, true, true);

	vector<Extremum> extrema = calcExtrema(DoG, minima, maxima);
	showKeypoints(img, extrema);


	for (auto extremum : extrema)
	{
		cout << extremum.y << ", " << extremum.x << ", " << extremum.isMax << ", " << extremum.sigma << endl;
	}

	return 0;
}
