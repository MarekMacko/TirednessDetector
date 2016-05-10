#pragma once

class FaceDetect
{
public:
	FaceDetect();
	~FaceDetect();
	int initialize(void);
	void run(void);

private:
	cv::VideoCapture videoCapture;
	cv::Mat frame;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	vector<cv::Rect> facesRect;
	vector<cv::Rect> eyesRects;

	const string WINDOW_NAME = "Window";
	const string CASCADE_FACE_FILE = "C:/Libraries/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
	const string CASCADE_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades/haarcascade_eye.xml";
};

