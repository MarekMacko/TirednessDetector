#include "stdafx.h"

FaceDetect::FaceDetect()
{
}

FaceDetect::~FaceDetect()
{
}

int FaceDetect::initialize(void)
{
	if (!videoCapture.open(0)) {
		cerr << "Error: VideoCapture open!";
		return -1;
	}
	if (!faceCascade.load(CASCADE_FACE_FILE)) {
		cerr << "Error: FaceCascade load!";
		return -1;
	}
	if (!eyeCascade.load(CASCADE_EYE_FILE)) {
		cerr << "Error: EyeCascade load!";
		return -1;
	}
	
	
	cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);
	
	return 0;
}

void FaceDetect::run(void)
{
	if (initialize() != 0) {
		cerr << "Error: FaceDetect initialize!";
		return;
	}

	cv::Mat frameGray;
	while (1) {
		videoCapture >> frame;
		cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
		faceCascade.detectMultiScale(frameGray, facesRect);//, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(20, 20), cv::Size(200, 200));

		for (int i = 0; i < facesRect.size(); i++) {
			cv::Rect faceRect = facesRect.at(i);
			cv::rectangle(frame, facesRect.at(i), cv::Scalar(0, 0, 255), 2);

			cv::Mat faceGray = frame(cv::Rect(faceRect.x, faceRect.y, faceRect.width, faceRect.height));
			cv::cvtColor(faceGray, faceGray, cv::COLOR_BGR2GRAY);
			eyeCascade.detectMultiScale(faceGray, eyesRects);

			for (int j = 0; j < eyesRects.size(); j++) {
				cv::Rect eyeRect = eyesRects.at(j);
				cv::rectangle(faceGray, eyeRect, cv::Scalar(0, 0, 255), 2);
			}
			cv::imshow("Face", faceGray);
		}

		cv::imshow(WINDOW_NAME, frame);

		if (cv::waitKey(1) == 27) {
			break;
		}
	}
}
