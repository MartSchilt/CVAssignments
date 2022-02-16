#pragma once
#include <opencv2\opencv.hpp>

class VideoDeviceDescriptor
{
public:
	VideoDeviceDescriptor() = default;

	uint32_t ID;
	cv::VideoCapture Device;

	uint32_t Width;
	uint32_t Height;
};