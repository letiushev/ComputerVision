#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "math.h"
#include "D:\Programming\3d sensing\Project3(3hw)\nanoflann.hpp"
#include "D:\Programming\3d sensing\Project3(3hw)\eigen-3.4.0\Eigen\Dense"
#include <chrono> 
#include <fstream>
#include <random>

using namespace Eigen;

void SearchNearestNeighbours(const MatrixXf& cloud1, const MatrixXf& cloud2, const size_t numberOfNeighbors, MatrixXi& indices, MatrixXf& dists, const bool isReturnedSquaredDist);

void MatrixSort(const MatrixXf& m, MatrixXf& out, MatrixXi& indexes, int saveAllElements);

int GetRotationAndTranslation(const MatrixXd& A, const MatrixXd& B, Matrix3d& R, Vector3d& t);

int ICP(const MatrixXf& sourceMatrix, const MatrixXf& secondMatrix, MatrixXf& outputMatrix, const int maxIterations, const double droppingThreshold);

int TR_ICP(const MatrixXf& sourceMatrix, const MatrixXf& secondMatrix, MatrixXf& outputMatrix, const int maxIterations, const double trIcpErrorThreshold, const double distanceThreshold, const int pointsThreshold);