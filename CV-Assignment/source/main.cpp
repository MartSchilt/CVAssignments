#include <opencv2\opencv.hpp>

#include "./hardware/video_device.h"

#define CHECKER_BOARD_WIDTH 8
#define CHECKER_BOARD_HEIGHT 8

#define FIELD_WIDTH  4.5f // TODO:: check these
#define FIELD_HEIGHT 4.5f

int main(char* argc, char** argv)
{
	// Window stuff
	cv::Mat webcamImage;
	cv::namedWindow("Display window");

	// Capture image
	// The image we will do calculations on when captured
	uint32_t captureCount = 0;
	cv::Mat  captureImage;

	// Point buffers
	std::vector<std::vector<cv::Point2f>> imagePoints; // Local-space
	//std::vector<std::vector<cv::Point3f>> objectPoints; // World-space
	imagePoints.resize(1);

	cv::Mat cameraMatrix, distCoeffs;

	// Precalculate world-space positions
	std::vector<cv::Point3f> worldSpace;
	for (int i = 0; i < CHECKER_BOARD_HEIGHT-1; i++)
	{
		for (int j = 0; j < CHECKER_BOARD_WIDTH-1; j++)
		{
			worldSpace.push_back(cv::Point3f(j* FIELD_WIDTH, i*FIELD_HEIGHT, 0));
		}
	}

	// Instantiate webcam device
	// Error checkt to see if we found a valid webcam
	VideoDevice* device = VideoDevice::FindVideoDevice();
	if (device == nullptr)
	{
		printf("ERROR - No user webcam found...\n");
		return -1;
	}

	// Main loop
	bool running = true;
	while (running == true) {
		// Capture an image from webcam device
		device->CaptureImage(webcamImage);

		// If we have not taken a capture yet, display webcam image
		if (captureCount == 0)
			cv::imshow("Display window", webcamImage);

		// --------------------------------------------------
		// Handle key events
		{
			// Retreive key and handle events
			char c = (char)cv::waitKey(1);
			switch (c)
			{
				// Take a capture for the callibration using the webcam 
				case 120 : // 'X'
				{
					imagePoints[captureCount].clear();
					captureImage = webcamImage; // retreive current image

					// Find the corners of the checkerBoard
					int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
					bool found = cv::findChessboardCorners(captureImage, cv::Size(CHECKER_BOARD_WIDTH-1, CHECKER_BOARD_HEIGHT-1), imagePoints[captureCount]);

					// Check if we found the actual corners
					if (found)
					{
						cv::Mat viewGray;
						cv::cvtColor(captureImage, viewGray, cv::COLOR_BGR2GRAY);
						cv::cornerSubPix(viewGray, imagePoints[captureCount], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

						// Draw the corners.
						cv::drawChessboardCorners(captureImage, { CHECKER_BOARD_WIDTH - 1, CHECKER_BOARD_HEIGHT - 1 }, cv::Mat(imagePoints[captureCount]), found);

						// Display current image to the window
						cv::imshow("Display window", captureImage);

						//objectPoints.push_back(worldSpace);

						// Increament capture count only if we found something
						captureCount++;
						imagePoints.resize(captureCount+1);
					}
					break;
				}
				case 99: // 'C'
				{
					imagePoints.pop_back();
					printf("Calibration started...\n");
					
					std::vector<cv::Mat> rvecs, tvecs; // rotation and translation matrix
					std::vector<float> reprojErrs;
					double totalAvgErr = 0;

					cv::Size imageSize(device->GetFrameWidth(), device->GetFrameHeight());

					cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
					cameraMatrix.at<double>(0, 0) = 1.0; // Fix aspect ratio

					distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

					std::vector<std::vector<cv::Point3f>> objectPoints(1);
					objectPoints[0] = worldSpace;

					objectPoints.resize(imagePoints.size(), objectPoints[0]);

					int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;

					double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
						distCoeffs, rvecs, tvecs, flags | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5);

					cv::Mat mapX, mapY;
					cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0), imageSize, CV_16SC2, mapX, mapY);

					break;
				}
				// Close application
				case 27 : // 'ESC'
				{
					running = false;
					break;
				}
			}
		}

	}

	return 0;

}