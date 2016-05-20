#pragma once

class FaceDetect
{
public:
	FaceDetect();
	~FaceDetect();
	int initialize(void);
	void run(void);

private:
	void setInfoToFrame(cv::Mat &frame);

	cv::VideoCapture videoCapture;
	cv::Mat frame;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	cv::CascadeClassifier rightEyeCascade;
	cv::CascadeClassifier leftEyeCascade;
	vector<cv::Rect> facesRect;
	vector<cv::Rect> eyesRects;

	vector<cv::Rect> rightEyesRects;
	vector<cv::Rect> leftEyesRects;

	int blinksCounter = 0;
	clock_t startTime;

	const string MAIN_WINDOW_NAME = "Main window";
	const string CASCADE_FACE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml";
	const string CASCADE_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml";
	const string CASCADE_RIGHT_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_righteye_2splits.xml";
	const string CASCADE_LEFT_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_lefteye_2splits.xml";
};

