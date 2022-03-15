/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include <opencv2/core/types.hpp>

#include "Reconstructor.h"

#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/photo.hpp>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	vector<Point2f> temp = vector<Point2f>(NULL);
	for (size_t i = 0; i < 4; i++)
	{
		center_coordinates.push_back(temp);
	}

	initialize();

	std::vector<int> frameIDs = { 60, 2550, 175, 510};
	for (int c = 0; c < m_cameras.size(); c++)
	{
		// set the frame the the correct id
		for (int cci = 0; cci < m_cameras.size(); cci++)
		{
			m_cameras[cci]->setVideoFrame(frameIDs[c]);
			m_cameras[cci]->advanceVideoFrame();
			generateForegroundImage(m_cameras[cci]);
		}

		update();
	}
	// set the frame the the correct id
	for (int cci = 0; cci < m_cameras.size(); cci++)
	{
		m_cameras[cci]->setVideoFrame(0);
		m_cameras[cci]->advanceVideoFrame();
	}
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}
void Reconstructor::generateForegroundImage(Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis


	// Background subtraction H
	Mat dilation, erosion, tmp, foreground, background, blur, img;
	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	threshold(tmp, foreground, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);

	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	threshold(tmp, background, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	threshold(tmp, background, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	bitwise_or(foreground, background, foreground);

	// Improve the foreground image
	// First erode to remove lines then dilate to fill holes in the models
	// Lastly erode a bit from the model to give the models a bit more of a "human" look
	int ekernel_size = 2;
	Mat ekernel = getStructuringElement(MORPH_RECT, Size(2 * ekernel_size + 1, 2 * ekernel_size + 1), Point(ekernel_size, ekernel_size));
	erode(foreground, erosion, ekernel);

	int dkernel_size = ekernel_size + 6;
	Mat dkernel = getStructuringElement(MORPH_RECT, Size(2 * dkernel_size + 1, 2 * dkernel_size + 1), Point(dkernel_size, dkernel_size));
	dilate(erosion, dilation, dkernel);

	ekernel_size = 2;
	ekernel = getStructuringElement(MORPH_RECT, Size(2 * ekernel_size + 1, 2 * ekernel_size + 1), Point(ekernel_size, ekernel_size));
	erode(dilation, img, ekernel);

	camera->setForegroundImage(img);
}

double Dist(double x1, double x2, double y1, double y2)
{
	double x = x1 - x2;
	double y = y1 - y2;

	return sqrt(pow(x, 2) + pow(y, 2));
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	// ------------------------------------
	int width = m_cameras[0]->getFrame().cols;
	int height = m_cameras[0]->getFrame().rows;

	std::vector<std::vector<double>> m_DepthMaps;
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		std::vector<double> tmp(m_cameras[c]->getFrame().rows * m_cameras[c]->getFrame().cols, 0.0f);
		m_DepthMaps.push_back(tmp);
	}

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
	for (v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				voxel;
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
			double dist = FLT_MAX;
			for (size_t c = 0; c < m_cameras.size(); ++c) {
				double distance = Dist(m_cameras[c]->getCameraLocation().x, m_cameras[c]->getCameraLocation().y, voxel->x, voxel->y);
				if (distance < dist)
				{
					dist = distance;
					voxel->ClosestCameraIndex = c;
				}

				cv::Point2i p = voxel->camera_projection[c];
				if (m_DepthMaps[c][p.x + p.y * width] <= 0.01f || distance < m_DepthMaps[c][p.x + p.y * width])
					m_DepthMaps[c][p.x + p.y * width] = distance;
			}

#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.clear();
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());


	for (v = 0; v < (int)m_visible_voxels.size(); ++v)
	{
		Voxel* voxel = m_visible_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			double distance = Dist(m_cameras[c]->getCameraLocation().x, m_cameras[c]->getCameraLocation().y, voxel->x, voxel->y);
			if (voxel->valid_camera_projection[c])
			{
				cv::Point2i p = voxel->camera_projection[c];

				if (distance - m_DepthMaps[c][p.x + p.y * width] <= 0.01f)
				{
					cv::Mat image = m_cameras[c]->getFrame();
					cv::Vec3b color = image.at<cv::Vec3b>(p.y, p.x);

					voxel->color = cv::Scalar(color[2], color[1], color[0], 255);
					voxel->ClosestCameraIndex = c;
				}
			}
		}

	}

	// Extract ground coordinates from all visible voxels
	std::vector<cv::Point2f> coordinatesxy(m_visible_voxels.size());
	for (int i = 0; i < m_visible_voxels.size(); i++)
		coordinatesxy[i] = cv::Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);

	// COlors for each cluster
	int clusterCount = 4;
	int clusterAttemots = 5;
	cv::Scalar clusterColors[] = { cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), };

	Mat labels;
	std::vector<cv::Point2f> centers;
	cv::kmeans(coordinatesxy, clusterCount, labels, TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 10000, 0.0001), clusterAttemots, KMEANS_PP_CENTERS, centers);

	// Calculates the sizes of each cluster
	std::vector<int> clusterSizes = { 0,0,0,0 };
	for (int v = 0; v < m_visible_voxels.size(); v++)
	{
		const int clusterID = labels.at<int>(v);
		clusterSizes[clusterID]++;
	}

	// Initialize people point Mats
	vector<vector<Mat>> people_Points(4, { Mat(), Mat(), Mat(), Mat() });
	for (int clust = 0; clust < 4; clust++) {
		for (int cam = 0; cam < 4; cam++) {
			people_Points[clust][cam] = Mat(clusterSizes[clust], 3, CV_64FC1);
		}
	}

	std::vector<int> clusterPointIndices = { 0,0,0,0 };
	for (v = 0; v < (int)m_visible_voxels.size(); ++v)
	{
		Voxel* voxel = m_visible_voxels[v];
		const int clusterIdx = labels.at<int>(v);

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			cv::Scalar color = voxel->color;

			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 0) = (double)color[0];
			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 1) = (double)color[1];
			people_Points[clusterIdx][c].at<double>(clusterPointIndices[clusterIdx], 2) = (double)color[2];
		}
		clusterPointIndices[clusterIdx]++;
	}

	if (m_color_models.size() < 4)
	{
		std::vector<cv::Ptr<cv::ml::EM>> model;
		for (int i = 0; i < 4; i++)
		{
			Ptr<cv::ml::EM> em_model = cv::ml::EM::create();
			//Set K
			em_model->setClustersNumber(4);
			//Set covariance matrix type
			em_model->setCovarianceMatrixType(cv::ml::EM::COV_MAT_SPHERICAL);
			//Convergence condition
			em_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));

			//train
			Mat training_labels;
			em_model->trainEM(people_Points[i][1], noArray(), training_labels, noArray());

			model.push_back(em_model);
		}

		m_color_models.push_back(model);
	}
	else
	{
		for (int it = 0; it < 4; it++)
		{
			cv::Mat sample(1, 3, CV_64FC1);
			std::map<int, int> clusterClassifications;
			std::vector<float> min_diffs;
			std::vector<float> max_diffs;
			std::vector<float> cam_bests;
			std::vector<float> cam_labels;
			// For each cluster
			for (int i = 0; i < 4; i++)
			{
				vector<vector<double>> avg_model_likelihoods = { 4, { 0.0, 0.0, 0.0, 0.0 } };

				for (int row = 0; row < people_Points[i][0].rows; row++) {


						sample.at<double>(0) = people_Points[i][it].at<double>(row, 0);
						sample.at<double>(1) = people_Points[i][it].at<double>(row, 1);
						sample.at<double>(2) = people_Points[i][it].at<double>(row, 2);

						for (int modelIndx = 0; modelIndx < 4; modelIndx++) {
							//Vec2d predict = m_color_models[it][modelIndx]->predict(sample, noArray());
							Vec2d predict2 = m_color_models[it][modelIndx]->predict2(sample, noArray());


							double likelihood = predict2[0];
							avg_model_likelihoods[modelIndx][0] += likelihood;
						}
				}
				vector<float> local_diffs = vector<float>(3);

				double prob = -10000.0f;
				int label = 0;

				//std::cout << "Predict: ";

				float clusterSize = (float)people_Points[i][0].rows;
				for (int its = 0; its < 4; its++)
				{
					avg_model_likelihoods[its][0] /= clusterSize;
				//	std::cout << avg_model_likelihoods[its][0] << ", ";
				

					if (avg_model_likelihoods[its][0] > prob)
					{
						prob = avg_model_likelihoods[its][0];
						label = its;
					}
				}

			//	std::cout << std::endl;
				clusterClassifications[i] = label;
			}

			cv::Mat frame = m_cameras[it]->getFrame();

			// Assign colors to each voxel based on GMM predictions
			for (int i = 0; i < (int)m_visible_voxels.size(); i++) {
				Voxel* voxel = m_visible_voxels[i];
				int clusterIdx = labels.at<int>(i);
				double distance = Dist(m_cameras[it]->getCameraLocation().x, m_cameras[it]->getCameraLocation().y, voxel->x, voxel->y);

				cv::Point2i p = voxel->camera_projection[it];
				if (distance - m_DepthMaps[it][p.x + p.y * width] < 0.01f)
				{
					int label = clusterClassifications[clusterIdx];
					//cv::Vec3b& color = frame.at<cv::Vec3b>(p.y, p.x);
					frame.at<Vec3b>(p.y, p.x)[0] = clusterColors[label][0];
					frame.at<Vec3b>(p.y, p.x)[1] = clusterColors[label][1];
					frame.at<Vec3b>(p.y, p.x)[2] = clusterColors[label][2];

				}
			}

			//m_cameras[it]->setForegroundImage(frame);
			if (it == 0)
				cv::imshow("CAMERA0", frame);
		}
	}

	}
} /* namespace nl_uu_science_gmt */