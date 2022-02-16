#include "./hardware/video_device.h"

VideoDevice::VideoDevice(const VideoDeviceDescriptor* _videoDeviceDescriptor)
	: m_DeviceID(_videoDeviceDescriptor->ID)
	, m_Device(_videoDeviceDescriptor->Device)
	, m_FrameWidth(_videoDeviceDescriptor->Width)
	, m_FrameHeight(_videoDeviceDescriptor->Height)
{ }

void VideoDevice::CaptureImage(cv::Mat& _image)
{
	m_Device >> _image;
}

VideoDevice* VideoDevice::GetVideoDevice(uint32_t _index)
{
	// Create device at index given by the user
	cv::VideoCapture cap(_index);

	// Check if device is open
	if (cap.isOpened() == true)
	{
		// Setup device descriptor
		VideoDeviceDescriptor deviceDesc = {};
		deviceDesc.ID     = _index;
		deviceDesc.Device = cap;
		deviceDesc.Width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		deviceDesc.Height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

		// Create and return device
		return new VideoDevice(&deviceDesc);
	}

	// No device found at this index
	return nullptr;
}

VideoDevice* VideoDevice::FindVideoDevice()
{
	for (int i = -10; i < 10; i++)
	{
		VideoDevice* device = GetVideoDevice(i);
		if (device != nullptr)
			return device;
	}
	return nullptr;
}
