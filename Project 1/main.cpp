#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"
#include <cmath>  
#define GAUSSIAN 0

using namespace std;
using namespace cv;

const bool disp_dynamic = true; // Set to true: DP disparities, false: naive disparities

int main(int argc, char** argv) 
{
	// Parameters
	const double focal_length = 3740;
	const double baseline = 160;
	const float lambda = 140;        //tune 10-200
	const int window_size = 3;        //tune 1-3
	const int dmin = 190;			   //tune 10-200
	const float max_disparity = 200;
	time_t start, end;
	time(&start);
	


	if (argc < 4) { // If there's too few arguments provided --> terminate
		std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_FILE" << std::endl;
		return 1;
	}

	cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE); // Left image
	cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE); // Right image
	
	const std::string output_file = argv[3];

	if (!image1.data) { // Terminate if any of the images is coid of data
		std::cerr << "No image1 data" << std::endl;
		return EXIT_FAILURE;
	}

	if (!image2.data) {
		std::cerr << "No image2 data" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "------------------ Parameters -------------------" << std::endl;
	std::cout << "lambda = " << lambda << std::endl;
	std::cout << "window_size = " << window_size << std::endl;
	std::cout << "disparity added due to image cropping = " << dmin << std::endl;
	std::cout << "output filename = " << argv[3] << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;

	int height = image1.size().height;
	int width = image1.size().width;

	cv::Mat disparities = cv::Mat::zeros(height, width, CV_8UC1);

	if (disp_dynamic)
		StereoEstimation_DP(window_size, dmin, height, width, image1, image2, disparities, lambda); // compute dynamic disparities
	else 
		StereoEstimation_Naive(window_size, dmin, height, width, image1, image2, disparities); // compute naive disparities

	// 3D reconstruction
	Disparity2PointCloud(output_file, height, width, disparities, window_size, dmin, baseline, focal_length); // Create a depth map using triangulation


	//metrics
	cv::Mat GT = cv::imread(argv[4], cv::IMREAD_GRAYSCALE); //read ground truth image

	float m1 = RMSE(disparities, GT);
	std::cout << "root mean square error = " << m1 << std::endl;

	float m2 = SSIM(disparities, GT, window_size);
	std::cout << "Structural similarity = " << m2 << std::endl;

	float m3 = PSNR(disparities, GT);
	std::cout << "Peak Signal-to-Noise Ratio = " << m3 << std::endl;


	time(&end);
	double timenow = double(end - start);
	std::cout << "calculated time = " << timenow << std::endl;


	// Save and display images
	std::stringstream out1;
	out1 << output_file << ".png";
	cv::imwrite(out1.str(), disparities);
	cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
	cv::imshow("Naive", disparities);
	cv::waitKey(0);

	return 0;
}


void StereoEstimation_Naive(const int& window_size, const int& dmin, int height, int width,
	cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
	int half_window_size = window_size / 2;
	int max_disparity = 40;

	for (int i = half_window_size; i < height - half_window_size; ++i) {

		std::cout
			<< "Calculating disparities for the naive approach... "
			<< std::ceil(((i - half_window_size + 1) / static_cast<double>(height - window_size + 1)) * 100) << "%\r"
			<< std::flush;

		for (int j = half_window_size; j < width - half_window_size; ++j) {
			int min_ssd = INT_MAX;
			int disparity = 0;

			for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) {
				int ssd = 0;

				for (int k = -half_window_size; k < half_window_size + 1; k++)
				{
					for (int l = -half_window_size; l < half_window_size + 1; l++)
					{
						int image1_coord = (int)image1.at<char>(i + k, j + l);
						int image2_coord = (int)image2.at<char>(i + k, j + l + d);

						ssd += (image1_coord - image2_coord) * (image1_coord - image2_coord);
					}
				}

				if (ssd < min_ssd) {
					min_ssd = ssd;
					disparity = d;
				}
			}

			if (abs(disparity) < max_disparity) 
			{
				naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity);
			}
		}
	}

	std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
	std::cout << std::endl;
}


void StereoEstimation_DP(const int& window_size, const int& dmin, int height, int width,
	cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, float lambda) // Homework to implement
{
	Size imageSize = image1.size();
	Mat disparityMap = Mat::zeros(imageSize, CV_16UC1);
#pragma omp parallel for
	for (int y_0 = window_size; y_0 < imageSize.height - window_size; ++y_0)
	{
		Mat C = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_16UC1);
		Mat M = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_8UC1);
		C.at<unsigned short>(0, 0) = 0;
		M.at<unsigned char>(0, 0) = 0;
		for (int i = 1; i < C.size().height; ++i)
		{
			C.at<unsigned short>(i, 0) = i * lambda;
			M.at<unsigned char>(i, 0) = 1;
		}
		for (int j = 1; j < C.size().width; ++j)
		{
			C.at<unsigned short>(0, j) = j * lambda;
			M.at<unsigned char>(0, j) = 2;
		}
		for (int r = 1; r < C.size().height; ++r)
		{
			for (int l = 1; l < C.size().width; ++l)
			{
				Rect leftROI = Rect(l, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1);
				Rect rightROI = Rect(r, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1);
				Mat leftWindow = image1(leftROI);
				Mat rightWindow = image2(rightROI);
				Mat diff;
				absdiff(leftWindow, rightWindow, diff);
				int SAD = sum(diff)[0];
				int cMatch = C.at<unsigned short>(r - 1, l - 1) + SAD;
				int cLeftOccl = C.at<unsigned short>(r - 1, l) + lambda;
				int cRightOccl = C.at<unsigned short>(r, l - 1) + lambda;
				// optimize cost
				int c = cMatch;
				int m = 0;
				if (cLeftOccl < c)
				{
					c = cLeftOccl;
					m = 1;
					if (cRightOccl < c)
					{
						c = cRightOccl;
						m = 2;
					}
				}
				C.at<unsigned short>(r, l) = c;
				M.at<unsigned char>(r, l) = m;
			}
		}
		// disparities
		int i = M.size().height - 1;
		int j = M.size().width - 1;
		while (j > 0) {
			if (M.at<unsigned char>(i, j) == 0)
			{
				disparityMap.at<unsigned short>(y_0, j) = abs(i - j);
				i--;
				j--;
			}
			else if (M.at<unsigned char>(i, j) == 1)
			{
				i--;
			}
			else if (M.at<unsigned char>(i, j) == 2)
			{
				disparityMap.at<unsigned short>(y_0, j) = 0;
				j--;
			}
		}
#pragma omp critical
		cout << "Progress: " << y_0 - window_size + 1 << "/" << imageSize.height - 2 * window_size << "\r" << flush;
	}
	Mat disparityMap_CV_8UC1;
	disparityMap.convertTo(disparityMap_CV_8UC1, CV_8UC1);
	dp_disparities = disparityMap_CV_8UC1;
}


void Disparity2PointCloud(const std::string& file, int height, int width, cv::Mat& disp,
	const int& window_size, const int& dmin, const double& baseline, const double& focal_length) 
{
	std::stringstream out3d;
	out3d << file << ".xyz";
	std::ofstream outfile(out3d.str());

	int xOffset = disp.cols / 2;
	int yOffset = disp.rows / 2;
	double x;
	double y;
	double z;
	
	cv::Mat disp_tmp;
	disp.convertTo(disp_tmp, CV_32FC1);
	disp_tmp = disp_tmp + dmin;

	for (int u = 0; u < disp.cols; u++) 
	{
		for (int v = 0; v < disp.rows; v++) 
		{
			float d = disp_tmp.at<float>(v, u) + dmin;
			if (d != 0) 
			{
				double u1 = (double)u - (double)width / 2.0;
				double u2 = (double)u + d - (double)width / 2.0;

				double v1 = (double)v - (double)height / 2.0;
				double v2 = v1;

				x = -(baseline * (u1 + u2) / (2 * d));
				y = baseline * v2 / d;
				z = baseline * focal_length / (double)d;
			}
			outfile << x << " " << y << " " << z << std::endl;
		}
	}
	std::cout << "Done witing point cloud" << std::endl;
}


//define metrics functions
//root mean squared error
float RMSE(cv::Mat image, cv::Mat GT)
{
	image.convertTo(image, CV_32F, 1.0 / 255, 0);
	GT.convertTo(GT, CV_32F, 1.0 / 255, 0);
	float sum = 0;
	float rmse = 0;
	float T = image.size().height * GT.size().width;
	for (int i = 0; i < image.size().height; i++)
	{
		for (int j = 0; j < image.size().width; j++)
		{
			float image1_coord = image.at<float>(i,j);
			float image2_coord = GT.at<float>(i,j);
			sum += pow((image1_coord - image2_coord),2);
		}
	}
	rmse = sqrt(sum / T);
	return rmse;
}

//Structural similarity
// ref:https://github.com/shibarain666/SSIM
float SSIM(cv::Mat image1, cv::Mat image2, int kernel_size) {

	static const double C1 = 6.5025;
	static const double C2 = 58.5225;
	cv::Mat img1_f, img2_f;
	image1.convertTo(img1_f, CV_32F);
	image2.convertTo(img2_f, CV_32F);
	cv::Mat tmp;

	/* Perform mean filtering on image using boxfilter */
	cv::Mat img1_avg, img2_avg;
	cv::boxFilter(img1_f, img1_avg, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
	cv::boxFilter(img2_f, img2_avg, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
	GaussianBlur(img1_f, img1_avg, cv::Size(kernel_size, kernel_size), 1.5);
	GaussianBlur(img2_f, img2_avg, cv::Size(kernel_size, kernel_size), 1.5);
#endif // GAUSSIAN
	cv::Mat img1_avg_sqr = img1_avg.mul(img1_avg);
	cv::Mat img2_avg_sqr = img2_avg.mul(img2_avg);

	/* Calculate variance map */
	cv::Mat img1_1 = img1_f.mul(img1_f);
	cv::Mat img2_2 = img2_f.mul(img2_f);
	cv::boxFilter(img1_1, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
	GaussianBlur(img1_1, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
	cv::Mat img1_var = tmp - img1_avg_sqr;
	cv::boxFilter(img2_2, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
	GaussianBlur(img2_2, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
	cv::Mat img2_var = tmp - img2_avg_sqr;

	/* Calculate covariance map */
	cv::Mat src_mul = img1_f.mul(img2_f);
	cv::Mat avg_mul = img1_avg.mul(img2_avg);
	cv::boxFilter(src_mul, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
	GaussianBlur(src_mul, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
	cv::Mat covariance = tmp - avg_mul;

	auto num = ((2 * avg_mul + C1).mul(2 * covariance + C2));
	auto den = ((img1_avg_sqr + img2_avg_sqr + C1).mul(img1_var + img2_var + C2));

	cv::Mat ssim_map;
	cv::divide(num, den, ssim_map);

	cv::Scalar mean_val = cv::mean(ssim_map);
	float mssim = (float)mean_val.val[0];

	return mssim;
}

//Peak signal-to-noise ratio
// ref:https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
float PSNR(cv::Mat image, cv::Mat GT)
{
	image.convertTo(image, CV_32F, 1.0 / 255, 0);
	GT.convertTo(GT, CV_32F, 1.0 / 255, 0);
	float sum = 0;
	float mse = 0;
	float T = image.size().height * image.size().width;
	double maxVal;
	minMaxLoc(image, 0, &maxVal, 0, 0);
	double maxVal2;
	minMaxLoc(GT, 0, &maxVal2, 0, 0);
	if (maxVal > maxVal2)
	{
		maxVal = maxVal;
	}
	else
	{
		maxVal = maxVal2;
	}

	for (int i = 0; i < image.size().height; i++)
	{
		for (int j = 0; j < image.size().width; j++)
		{
			float image1_coord = image.at<float>(i, j);
			float image2_coord = GT.at<float>(i, j);
			sum += pow((image1_coord - image2_coord),2);
		}
	}
	mse = (sum / T);
	float psnr = 0;
	float R = (float) maxVal;
	psnr = 10 * log10f(pow(R,2) / mse);
	return psnr;
}