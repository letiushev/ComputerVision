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
#include "main.h"

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace nanoflann;
using namespace chrono;

const int NumberOfNeighbors = 5; // neighbors to find in KNN

const int NumberOfIterations = 100; // max iterations for icp and tr-icp

const double DroppingThreshold = 0.0001; // if err below stop condition for icp
const double ICPErrorThreshold = 0.01;

const double DistanceThreshold = 130.0; // if distance below stop condition for tr-icp
const double TRICPErrorThreshold = 0.048; // if err below stop condition for tr-icp

int PointsThreshold = 45 * pow(10, 3); // how much pts process in tr-icp

const bool AddGausianNoise = false; // add gaussian noise

double PreviousError = 0;

vector<Point3d> ReadData(char* filename);

vector<Point3d> ReadData(char* filename)
{
	bool flag = false;
	string line;
	ifstream inputFile;
	inputFile.open(filename);
	vector<Point3d> result;
	while (getline(inputFile, line)) // get data through lines of file
	{
		if (flag) // header check
		{
			string arr[3];
			int i = 0;
			stringstream temp(line); // strignsteam from file
			while (temp.good() && i < 3) // iter over data in line
			{
				temp >> arr[i];
				i++;
			}
			if (i == 3) // when 3 coords only here
			{
				Point3d tmp = { stof(arr[0]), stof(arr[1]), stof(arr[2]) }; // pts to the vector
				result.push_back(tmp);
			}
			else // reached end sucessfully
			{
				break;
			}
		}
		if (line.find("end_header") != string::npos) // if read header set flag
		{
			flag = true;
		}
	}
	return result;
}

void SearchNearestNeighbours(const MatrixXf& cloud1, const MatrixXf& cloud2, const size_t numberOfNeighbors, MatrixXi& indices, MatrixXf& dists, const bool isReturnedSquaredDist = 0)
{
	const int leafCount = 15; // complexity of tree
	typedef Matrix<float, Dynamic, 3, RowMajor> RowMatX3f; // custom type, fill in by (row, column)
	RowMatX3f coords1 = cloud1.leftCols(3); // get first 3(4) corrdinates starting from the left
	RowMatX3f coords2 = cloud2.leftCols(3);
	nanoflann::KDTreeEigenMatrixAdaptor<RowMatX3f> matrixIndices(3, coords1, leafCount); // reorg to kdtree 
	matrixIndices.index->buildIndex(); // fill up kdtree with data
	indices.resize(cloud2.rows(), numberOfNeighbors); // for each index find numberOfNeighbors closest neighbors mat 0->2 3 4 9
	dists.resize(cloud2.rows(), numberOfNeighbors); // same for distances 
	for (int i = 0; i < coords2.rows(); ++i)
	{
		vector<float> currentPoint{ coords2.data()[i * 3 + 0], coords2.data()[i * 3 + 1], coords2.data()[i * 3 + 2] }; // current point we iter over (indx,indx,dist)
		vector<size_t> retrievedIndices(numberOfNeighbors); // indeses of closest points to current point
		vector<float> outDistancesSqrt(numberOfNeighbors); // distances of closest points to current point
		nanoflann::KNNResultSet<float> resultSet(numberOfNeighbors); // define type of closest point (index, dist)
		resultSet.init(&retrievedIndices[0], &outDistancesSqrt[0]); // put data into knn datastructure
		matrixIndices.index->findNeighbors(resultSet, &currentPoint[0], nanoflann::SearchParams(10)); // find closest neighbors
		for (size_t j = 0; j < numberOfNeighbors; ++j) // fill up the kdtree with indices and distances with amount = numberOfNeighbors
		{											   
			indices(i, j) = retrievedIndices[j];
			if (isReturnedSquaredDist)
			{
				dists(i, j) = outDistancesSqrt[j]; // euclidian dist
			}
			else
			{
				dists(i, j) = sqrt(outDistancesSqrt[j]); // squared dist useful if dist is small

			}
		}
	}
}

void MatrixSort(const MatrixXf& m, MatrixXf& out, MatrixXi& indexes, int saveAllElements = -1)
{
	vector<pair<float, int> > vectorPair; // 2 elements: dist and idices. point from cloud2 and corresp point from cloud1
	if (saveAllElements == -1)  // all elms of orig mat are saved, otherwise elements number is the same
	{
		saveAllElements = m.rows(); // useful for tr_icp how much to track based on PointsThreshold
	}

	for (int index = 0; index < m.rows(); index++)
	{
		float distance = m(index);
		vectorPair.push_back(make_pair(distance, index)); // fill up vector pair wiht dist and index
	}

	stable_sort(vectorPair.begin(), vectorPair.end(), [](const auto& a, const auto& b) {return a.first < b.first; }); // sort vectorPair with lambda based on distances

	MatrixXi tempIndexes(saveAllElements, 1); // we rebuild based on indices so the 1 elem will be with smallest distance
	MatrixXf tempOut(saveAllElements, 1);
	for (int i = 0; i < saveAllElements; i++)
	{
		tempOut(i, 0) = m(vectorPair[i].second); // add point to correct position (based on index which has been sorted)
		tempIndexes(i, 0) = vectorPair[i].second; // save index to indeces matrix
	}
	out = tempOut;
	indexes = tempIndexes;
}

int GetRotationAndTranslation(const MatrixXd& A, const MatrixXd& B, Matrix3d& R, Vector3d& t)
{
	Vector3d centroidA(0, 0, 0);
	Vector3d centroidB(0, 0, 0);
	MatrixXd copyOfA = A;   //bcs they passed as reference
	MatrixXd copyOfB = B;

	for (int i = 0; i < A.rows(); i++) // get the row from matrix
	{
		centroidA += A.block<1, 3>(i, 0).transpose(); // one row from matrix and add
		centroidB += B.block<1, 3>(i, 0).transpose();
	}
	centroidA /= A.rows(); // avg centroid value of each conlumn
	centroidB /= A.rows();
	for (int i = 0; i < A.rows(); i++) // substract in order to get normalized point cloud
	{
		copyOfA.block<1, 3>(i, 0) = A.block<1, 3>(i, 0) - centroidA.transpose();
		copyOfB.block<1, 3>(i, 0) = B.block<1, 3>(i, 0) - centroidB.transpose();
	}

	MatrixXd H = copyOfA.transpose() * copyOfB; // get covariance matrix = 3xn * nx3 = 3x3
	MatrixXd U; 
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;

	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV); // does svd on covariance matrix
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

	R = Vt.transpose() * U.transpose(); // 3x3 * 3x3

	if (R.determinant() < 0) // if two point clounds are mirror
	{
		Vt.block<1, 3>(2, 0) *= -1; // get negative values
		R = Vt.transpose() * U.transpose(); // put obtained negative values
	}

	t = centroidB - R * centroidA; // get translation vector

	return 1;
}

int ICP(const MatrixXf& sourceMatrix, const MatrixXf& secondMatrix, MatrixXf& outputMatrix, const int maxIterations, const double droppingThreshold)
{
	MatrixXi indices;
	MatrixXf distances;
	outputMatrix = sourceMatrix;
	float prevSumOfDistances = 10;

	MatrixXf sourceNeighbours(secondMatrix.rows(), 3); // new mat = source but order like second Mat with its num of rows
	double meanError = 0;

	float sumOfDistances = 0;

	for (int i = 0; i < maxIterations; i++) // iterate untill max of iterations or break point
	{
		SearchNearestNeighbours(outputMatrix, secondMatrix, NumberOfNeighbors, indices, distances);
		sumOfDistances = 0; // error btw pts of source and second matx
		for (int d = 0; d < distances.size(); d++)
		{
			sumOfDistances += distances(d);
		}
		meanError = sumOfDistances / distances.size();

		if (abs(meanError) < ICPErrorThreshold)
		{
			cout << "In ICP ICPErrorThreshold is reached" << endl;
			cout << "Iterations: " + to_string(maxIterations) << endl;
			return 1;
		}

		for (int j = 0; j < sourceNeighbours.rows(); j++) // reorder source mat to fit NN algorithm
		{
			int index = indices(j, 1);
			sourceNeighbours(j, 0) = outputMatrix(index, 0);
			sourceNeighbours(j, 1) = outputMatrix(index, 1);
			sourceNeighbours(j, 2) = outputMatrix(index, 2);
		}

		Matrix3d tempR;
		Vector3d tempt;
		GetRotationAndTranslation(sourceNeighbours.cast <double>(), secondMatrix.cast <double>(), tempR, tempt); // get Rotation mat and translation vector
		Matrix3f R = tempR.cast<float>(); // cast = convert from double to float
		Vector3f t = tempt.cast<float>();

		outputMatrix = (R * outputMatrix.transpose()).transpose(); // apply rotation

		for (int tIter = 0; tIter < outputMatrix.rows(); tIter++) // apply transformation. iterate over rows
		{
			for (int j = 0; j < 3; j++) // iter over columns
			{
				outputMatrix(tIter, j) = outputMatrix(tIter, j) + t(j);
			}
		}

		cout << "ICP iteration: " + to_string(i) << endl;
		cout << "Mean squared error: " + to_string(meanError) + "/" + to_string(ICPErrorThreshold) << endl;
		PreviousError = meanError;
		prevSumOfDistances = sumOfDistances;
	}
}
int TR_ICP(const MatrixXf& sourceMatrix, const MatrixXf& secondMatrix, MatrixXf& outputMatrix, const int maxIterations, const double trIcpErrorThreshold, const double distanceThreshold, const int pointsThreshold)
{
	float prevSumOfDistances = 10;
	MatrixXi indices;
	MatrixXf squaredDistances, trimmedSquaredDistances(pointsThreshold, 1);
	MatrixXf trimmedDistances(pointsThreshold, 3), trimmedSource(pointsThreshold, 3);
	outputMatrix = sourceMatrix;

	MatrixXf sourceNeighbours(secondMatrix.rows(), 3);
	double meanError = 0;

	for (int i = 0; i < maxIterations; i++) // iterate untill max of iterations of break point
	{
		SearchNearestNeighbours(outputMatrix, secondMatrix, NumberOfNeighbors, indices, squaredDistances, 1); // finds nearest neighbors

		MatrixXi oldDistancesIndeces;
		MatrixXf sortedTrimmedIndeces;
		MatrixSort(squaredDistances, sortedTrimmedIndeces, oldDistancesIndeces, pointsThreshold); // sort matrix based on indeces and trimm distances

		for (int i = 0; i < pointsThreshold; i++) // big matrix we want to save only points threshold and save trimmed data
		{
			int destinationIndex = oldDistancesIndeces(i);
			int sourceIndex = indices(destinationIndex, 1);
			trimmedSource.block<1, 3>(i, 0) = outputMatrix.block<1, 3>(sourceIndex, 0); // row of matrix
			trimmedDistances.block<1, 3>(i, 0) = secondMatrix.block<1, 3>(destinationIndex, 0);
		}

		// calculate breakdown points
		float sumOfDist = squaredDistances.sum(); // sum smallest distances
		float error = sumOfDist / pointsThreshold; // trimmed mse

		if (error < trIcpErrorThreshold || abs(prevSumOfDistances - sumOfDist) < distanceThreshold) // stop contition of tr-icp
		{
			cout << "TR_ICP reached breakdown point" << endl;
			cout << "Iterations: " + to_string(maxIterations) << endl;
			return 1;
		}

		Matrix3d tempR;
		Vector3d tempt;
		GetRotationAndTranslation(trimmedSource.cast<double>(), trimmedDistances.cast<double>(), tempR, tempt); // get Rotation and translation
		Matrix3f R = tempR.cast<float>(); // cast = convert from double to float
		Vector3f t = tempt.cast<float>();

		outputMatrix = (R * outputMatrix.transpose()).transpose(); // apply rotation

		for (int tIter = 0; tIter < outputMatrix.rows(); tIter++) // apply translation. iter over rows
		{
			for (int j = 0; j < 3; j++) // iter over columns
			{
				outputMatrix(tIter, j) = outputMatrix(tIter, j) + t(j);
			}
		}

		cout << "TR-ICP iteration: " + to_string(i) << endl;
		cout << "Trimmed mean squared error: " + to_string(error) + "/" + to_string(trIcpErrorThreshold) << endl;
		cout << "Change of distances: " + to_string(abs(prevSumOfDistances - sumOfDist)) + "/" + to_string(distanceThreshold) << endl;
		prevSumOfDistances = sumOfDist;
	}
	return 1;
}

int main(int argc, char** argv) 
{
	vector<Point3d> points1 = ReadData(argv[1]); // store data here
	vector<Point3d> points2 = ReadData(argv[2]);

	MatrixXf source(points1.size(), 3);
	for (int i = 0; i < points1.size(); i++) {
		source(i, 0) = points1[i].x;
		source(i, 1) = points1[i].y;
		source(i, 2) = points1[i].z;
	}

	MatrixXf destination(points2.size(), 3);
	for (int i = 0; i < points2.size(); i++) {
		source(i, 0) = points2[i].x;
		source(i, 1) = points2[i].y;
		source(i, 2) = points2[i].z;
	}

	if (AddGausianNoise) // add noise here
	{
		default_random_engine random; // random number generator
		normal_distribution<double> dist(0, 0.2); // offset, noise from Gaussian distribution
		float sample;

		for (int i = 0; i < points2.size(); i++) 
		{
			sample = dist(random); 
			destination(i, 0) = points2[i].x + sample;
			destination(i, 1) = points2[i].y + 1 + sample;
			destination(i, 2) = points2[i].z + 1 + sample;
		}
	}
	else 
	{
		for (int i = 0; i < points2.size(); i++) 
		{
			destination(i, 0) = points2[i].x;
			destination(i, 1) = points2[i].y;
			destination(i, 2) = points2[i].z;
		}
	}

	MatrixXf out(destination.rows(), 3);
	MatrixXf out2(destination.rows(), 3);

	time_t startICP, startTRICP, endICP, endTRICP;

	time(&startICP);
	ICP(source, destination, out, NumberOfIterations, DroppingThreshold);
	time(&endICP);
	double timenow = double(endICP - startICP);
	cout << "calculated time for ICP = " << timenow << endl;

	time(&startTRICP);
	TR_ICP(source, destination, out2, NumberOfIterations, TRICPErrorThreshold, DistanceThreshold, PointsThreshold);
	time(&endTRICP);
	timenow = double(endTRICP - startTRICP);
	cout << "calculated time for TR-ICP = " << timenow << endl;

	ofstream outputFile1("D:/Programming/3d sensing/NEWNEWHW3/Project2/data/out/outIcp.xyz");
	for (int i = 0; i < out.rows(); i++) {
		for (int j = 0; j < 3; j++) {
			outputFile1 << out(i, j) << " ";
		}
		outputFile1 << endl;
	}
	outputFile1.close();

	ofstream outputFile2("D:/Programming/3d sensing/NEWNEWHW3/Project2/data/out/outTr_Icp.xyz");
	for (int i = 0; i < out2.rows(); i++) {
		for (int j = 0; j < 3; j++) {
			outputFile2 << out2(i, j) << " ";
		}
		outputFile2 << endl;
	}
	outputFile2.close();

	return 0;
}