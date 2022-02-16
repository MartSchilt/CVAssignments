#pragma once
#include "./hardware/video_device_descriptor.h"

class VideoDevice
{
public:
	VideoDevice() = default;
	VideoDevice(const VideoDeviceDescriptor* _videoDeviceDescriptor);

	// Capture function
	void CaptureImage(cv::Mat& _image);

	// Retreive device functions
	static VideoDevice* GetVideoDevice(uint32_t _index = 0);
	static VideoDevice* FindVideoDevice();

	// Getters
	inline uint32_t GetFrameWidth() const { return m_FrameWidth; }
	inline uint32_t GetFrameHeight() const { return m_FrameHeight; }

private:
	uint32_t         m_DeviceID;
	cv::VideoCapture m_Device;

	uint32_t m_FrameWidth;
	uint32_t m_FrameHeight;
};