#pragma once

void CreateGaussianMask(const int& window_size, cv::Mat& mask, int& sum_mask);
void OurBilateralFilter(const cv::Mat& input, cv::Mat& output);
cv::Mat JointBilateral(const cv::Mat input, const cv::Mat guide);
cv::Mat JointBilateralUpsampling(const cv::Mat input, const cv::Mat guide);
cv::Mat IterativeJointUpsampling(cv::Mat& D, cv::Mat& I);
void Disparity2PointCloud(const std::string& file, int height, int width, cv::Mat& disp,
	const int& window_size, const int& dmin, const double& baseline, const double& focal_length);
float RMSE(cv::Mat image, cv::Mat GT);
float SSIM(cv::Mat image1, cv::Mat image2, int kernel_size);
float PSNR(cv::Mat image, cv::Mat GT);
