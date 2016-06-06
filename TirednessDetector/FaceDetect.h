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
	void showError(string errorMessage);
	cv::Rect getFaceRect(const cv::Mat image);
	cv::Rect getRightEyeRect(const cv::Mat image, const cv::Rect faceRect);
	cv::Rect getLeftEyeRect(const cv::Mat image, const cv::Rect faceRect);
	cv::Rect getRightOpenEyeRect(const cv::Mat image, const cv::Rect faceRect);
	cv::Rect getLeftOpenEyeRect(const cv::Mat image, const cv::Rect faceRect);

	cv::VideoCapture videoCapture;
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;
	cv::CascadeClassifier rightEyeCascade;
	cv::CascadeClassifier leftEyeCascade;
	vector<cv::Rect> eyesRects;

	int blinksCounter = 0;
	clock_t startTime;

	const string MAIN_WINDOW_NAME = "Main window";
	const string CASCADE_FACE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml";
	//const string CASCADE_FACE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/lbpcascade_frontalface.xml";
	const string CASCADE_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml";
	const string CASCADE_RIGHT_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_righteye_2splits.xml";
	const string CASCADE_LEFT_EYE_FILE = "C:/Libraries/opencv/sources/data/haarcascades_cuda/haarcascade_lefteye_2splits.xml";
	const string VIDEO_FILE = "D:/video.avi";
};

