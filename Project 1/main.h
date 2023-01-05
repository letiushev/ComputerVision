#pragma once

void StereoEstimation_Naive(
  const int& window_size,
  const int& dmin,
  int height,
  int width,
  cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void Disparity2PointCloud(
  const std::string& output_file,
  int height, int width, cv::Mat& disparities,
  const int& window_size,
  const int& dmin, const double& baseline, const double& focal_length);

void StereoEstimation_DP(const int& window_size, const int& dmin, int height, int width,
	cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, float lambda);

// metrics
float RMSE(cv::Mat image, cv::Mat GT);
float SSIM(cv::Mat image1, cv::Mat image2, int kernel_size);
float PSNR(cv::Mat image, cv::Mat GT);