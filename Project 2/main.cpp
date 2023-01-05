#include <iostream>
#include <opencv2/opencv.hpp>
#include "main.h"
#include <fstream>

using namespace cv;

void CreateGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask)
{
	Size mask_size(window_size, window_size);
	mask = Mat(mask_size, CV_8UC1);

	const double hw = window_size / 2;
	const double sigma = std::sqrt(2.0) * hw / 2.5;
	const double sigmaSq = sigma * sigma;

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			double r2 = (r - hw) * (r - hw) + (c - hw) * (c - hw);
			mask.at<uchar>(r, c) = 255 * std::exp(-r2 / (2 * sigmaSq));
		}
	}

	for (int r = 0; r < window_size; ++r) {
		for (int c = 0; c < window_size; ++c) {
			sum_mask += static_cast<int>(mask.at<uchar>(r, c));
		}
	}
}

void OurBilateralFilter(const cv::Mat& input, cv::Mat& output) {

	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 17;

	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			output.at<uchar>(r, c) = 0;
		}
	}

	cv::Mat mask;
	int sum_mask = 0;

	CreateGaussianMask(window_size, mask, sum_mask);

	const float sigmaRange = 20;
	const float sigmaRangeSq = sigmaRange * sigmaRange;

	float range_mask[256];
	for (int diff = 0; diff < 256; ++diff) {
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}

	for (int r = window_size / 2; r < height - window_size / 2; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			int intensity_center = static_cast<int>(input.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(input.at<uchar>(
						r + i,
						c + j));
					int diff = std::abs(intensity_center - intensity);
					float weight_range = range_mask[diff];
					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2,
						j + window_size / 2));
					float weight = weight_range * weight_spatial;
					sum += intensity * weight;
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;
		}
	}
}

Mat JointBilateral(const Mat input, const Mat guide) {

	const auto width = input.cols;
	const auto height = input.rows;

	const int window_size = 5;

	Mat output = Mat::zeros(input.size(), input.type());

	Mat mask;
	int sum_mask = 0;

	CreateGaussianMask(window_size, mask, sum_mask);

	const float sigmaRange = 10;
	const float sigmaRangeSq = sigmaRange * sigmaRange;

	float range_mask[256];
	for (int diff = 0; diff < 256; ++diff) {
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}
	for (int r = window_size / 2; r < height - window_size / 2; ++r) {       //we are using guided image to compare
		for (int c = window_size / 2; c < width - window_size / 2; ++c) {
			int intensity_center = static_cast<int>(guide.at<uchar>(r, c));

			int sum = 0;
			float sum_Bilateral_mask = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(guide.at<uchar>(
						r + i,
						c + j));
					int diff = std::abs(intensity_center - intensity);
					float weight_range = range_mask[diff];
					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2,
						j + window_size / 2));
					float weight = weight_range * weight_spatial;
					sum += intensity * weight;
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;
		}
	}

	return output;
}


Mat JointBilateralUpsampling(const Mat input, const Mat guide) {

	const auto width = guide.cols;
	const auto height = guide.rows;

	const int window_size = 5;
	float width_factor = (float)input.cols / (float)guide.cols;                //we are using these factors
	float height_factor = (float)input.rows / (float)guide.rows;

	Mat output = Mat::zeros(guide.size(), guide.type());

	Mat mask;
	int sum_mask = 0;

	CreateGaussianMask(window_size, mask, sum_mask);

	const float sigmaRange = 20;
	const float sigmaRangeSq = sigmaRange * sigmaRange;

	float range_mask[256];
	for (int diff = 0; diff < 256; ++diff) {
		range_mask[diff] = std::exp(-diff * diff / (2 * sigmaRangeSq));
	}
	for (int r = window_size / 2; r < height - window_size / 2 - 1; ++r) {
		for (int c = window_size / 2; c < width - window_size / 2 - 1; ++c) {
			int intensity_center = static_cast<int>(guide.at<uchar>(r, c));
			int sum = 0;
			float sum_Bilateral_mask = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i) {
				for (int j = -window_size / 2; j <= window_size / 2; ++j) {
					int intensity = static_cast<int>(guide.at<uchar>(
						r + i,
						c + j));
					int diff = std::abs(intensity_center - intensity);
					float weight_range = range_mask[diff];
					int weight_spatial = static_cast<int>(mask.at<uchar>(
						i + window_size / 2,
						j + window_size / 2));
					float weight = weight_range * weight_spatial;

					float x = round((r + i) * height_factor);
					float y = round((c + j) * width_factor);

					sum += (float)input.at<uchar>(x,y) * weight;                 //here too
					sum_Bilateral_mask += weight;
				}
			}
			output.at<uchar>(r, c) = sum / sum_Bilateral_mask;
		}
		//std::cout << r << "\n" << std::endl;
	}

	return output;
}


Mat IterativeJointUpsampling(Mat& D, Mat& I) {
	int uf = log2(I.size().height / D.size().height);
	Mat D_ = D;
	for (int i = 1; i < uf - 1; ++i)
	{
		resize(D_, D_, Size(), 2, 2);
		Mat I_lo;
		resize(I, I_lo, D_.size());
		D_ = JointBilateral(I_lo, D_);
	}
	resize(D_, D_, I.size());
	D_ = JointBilateral(I, D_);
	return D_;
}


int dmin = 200;
double baseline = 160;
double focal_length = 3740;

void Disparity2PointCloud(const std::string& file, int height, int width, cv::Mat& disp,
	const int& window_size, const int& dmin, const double& baseline, const double& focal_length)   //from hw1
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
float RMSE(cv::Mat image, cv::Mat GT)                                //from hw1
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
			float image1_coord = image.at<float>(i, j);
			float image2_coord = GT.at<float>(i, j);
			sum += pow((image1_coord - image2_coord), 2);
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
			sum += pow((image1_coord - image2_coord), 2);
		}
	}
	mse = (sum / T);
	float psnr = 0;
	float R = (float)maxVal;
	psnr = 10 * log10f(pow(R, 2) / mse);
	return psnr;
}


int main() {
	time_t start, end;
	time(&start);

	//read images
	std::cout << "Start!\n";
	String input = "motorcycle"; 
	Mat im = imread("D:/Programming/3d sensing/hw2/data/" + input + ".png", IMREAD_GRAYSCALE);
	Mat guide = imread("D:/Programming/3d sensing/hw2/data/" + input + "2.png", IMREAD_GRAYSCALE);
	Mat impfm = imread("D:/Programming/3d sensing/hw2/data/" + input + ".pfm", IMREAD_GRAYSCALE);
	Mat impfmResized;
	resize(impfm, impfmResized, Size(), 0.5, 0.5);
	std::cout << "read the image successfully\n";

	if (im.data == nullptr) {
		std::cerr << "Failed to load image" << std::endl;
	}

	//create output matrices
	Mat output = Mat::zeros(im.size(), im.type());
	Mat output2 = Mat::zeros(im.size(), im.type());
	Mat output3 = Mat::zeros(im.size(), im.type());
	Mat output4 = Mat::zeros(im.size(), im.type());

	double window_size = 5;

	//functions calling
	OurBilateralFilter(im, output);
	std::cout <<"OurBilateralFilter is done\n";
	output2 = JointBilateral(im, guide);
	std::cout << "JointBilateral is done\n";
	output3 = JointBilateralUpsampling(impfmResized, guide);
	std::cout << "JointBilateralUpsampling is done\n";
	output4 = IterativeJointUpsampling(impfmResized, guide);
	std::cout << "IterativeJointUpsampling is done\n";
	int height = output3.rows;
	int width = output3.cols;
	Disparity2PointCloud("PointCloudJointBilateralUpsampling" + input, height, width, output3, window_size, dmin, baseline, focal_length);
	Disparity2PointCloud("PointCloudIterativeJointUpsampling" + input, height, width, output4, window_size, dmin, baseline, focal_length);

	//output to image writing
	imwrite("D:/Programming/3d sensing/hw2/data/" + input + "Out.png", output);
	imwrite("D:/Programming/3d sensing/hw2/data/" + input + "Out2.png", output2);
	imwrite("D:/Programming/3d sensing/hw2/data/" + input + "Out3.png", output3);
	imwrite("D:/Programming/3d sensing/hw2/data/" + input + "Out4.png", output4);


	//metrices
	cv::Mat GT = impfm; //read ground truth image

	float m1 = RMSE(output3, GT);
	std::cout << "root mean square error for output3 = " << m1 << std::endl;

	float m2 = SSIM(output3, GT, window_size);
	std::cout << "Structural similarity for output3 = " << m2 << std::endl;

	float m3 = PSNR(output3, GT);
	std::cout << "Peak Signal-to-Noise Ratio for output3 = " << m3 << std::endl;

	float m11 = RMSE(output4, GT);
	std::cout << "root mean square error for output4 = " << m11 << std::endl;

	float m22 = SSIM(output4, GT, window_size);
	std::cout << "Structural similarity for output4 = " << m22 << std::endl;

	float m33 = PSNR(output4, GT);
	std::cout << "Peak Signal-to-Noise Ratio for output4 = " << m33 << std::endl;


	time(&end);
	double timenow = double(end - start);
	std::cout << "calculated time = " << timenow << std::endl;



	return 0;
}