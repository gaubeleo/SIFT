#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <direct.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

#define SIGMA 1.6
#define OCTAVE_COUNT 4
#define STEP_COUNT 4

#define FILTER_WIDTH 9
#define FILTER_HEIGHT 9

// Load an image from a given path using OpenCV 
// use IMREAD_COLOR or IMREAD_GRAYSCALE as flags
Mat loadImg(string directory, string filename, int flags){
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
bool saveImg(string directory, string filename, Mat img){
	string fullFilename = string(directory + "\\" + filename);
	struct stat sb;

	if (!(stat(directory.c_str(), &sb) == 0 && sb.st_mode == S_IFDIR)){
		_mkdir(directory.c_str());
	}

	cout << "successfully written '" << fullFilename << "' to file!" << endl;

	return imwrite(fullFilename, img);
}

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
	Mat octaveImg, bluredImg;
	for (int o = 0; o < OCTAVE_COUNT; o++){
		cout << "\toctave: " << o << endl;
		vector<Mat> octave;
		octaveImg = downscale(grayImg, 1<<o); // downscale original image by 2^o
		if (o == 0)
			octave.push_back(octaveImg); // include original image in first octave
		for (int s = 0; s < STEP_COUNT + 2; s++){
			cout << "\t\tstep: " << s << endl;
			bluredImg = filter(octaveImg, kernels.at(s));	
			octave.push_back(bluredImg);
		}
		gp.push_back(octave);
	}
	return gp;
}

void showGaussianPyramid(vector<vector<Mat>> &gp, bool scaleUp){
	cout << "displaying Gaussian Pyramid:" << endl;
	assert(gp.size() == OCTAVE_COUNT);

	for (int o = 0; o < OCTAVE_COUNT; o++){
		vector<Mat> octave = gp.at(o);

		if (o == 0)
			assert(octave.size() == STEP_COUNT + 3);
		else 
			assert(octave.size() == STEP_COUNT + 2);

		for (int s = 0; s < octave.size(); s++){
			double k = pow(2, (double)s / (double)STEP_COUNT);
			double sigma = (1 << o) * k*SIGMA;
			Mat img = octave.at(s);
			if (scaleUp)
				imshow("GP at o=" + to_string(o) + ", s=" + to_string(sigma), upscale(img, 1<<o+1));
			else
				imshow("GP at o=" + to_string(o) + ", s=" + to_string(sigma), img);
			waitKey();
			destroyAllWindows();
		}
	}
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	Mat img = loadImg("src", "line.png", IMREAD_COLOR); //IMREAD_COLOR

	vector<vector<Mat>> gp = createGaussianPyramid(img);
	showGaussianPyramid(gp, true);

	return 0;
}