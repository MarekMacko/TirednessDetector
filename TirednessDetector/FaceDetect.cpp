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
		cerr << "Error: VideoCapture open!" << endl;
		return -1;
	}
	if (!faceCascade.load(CASCADE_FACE_FILE)) {
		cerr << "Error: FaceCascade load!" << endl;
		return -1;
	}
	if (!eyeCascade.load(CASCADE_EYE_FILE)) {
		cerr << "Error: EyeCascade load!" << endl;
		return -1;
	}
	if (!leftEyeCascade.load(CASCADE_LEFT_EYE_FILE)) {
		cerr << "Error: Right eye cascade load!" << endl;
		return -1;
	}
	if (!rightEyeCascade.load(CASCADE_RIGHT_EYE_FILE)) {
		cerr << "Error: Left eye cascad load!" << endl;
		return -1;
	}

	cv::namedWindow(MAIN_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
	cv::namedWindow("right", cv::WINDOW_NORMAL);
	cv::namedWindow("left", cv::WINDOW_NORMAL);

	cv::resizeWindow("left", 200, 300);
	cv::resizeWindow("right", 200, 300);
	return 0;
}

void FaceDetect::run(void) {
	if (initialize() != 0) {
		cerr << "Error: FaceDetect initialize!";
		return;
	}

	cv::Mat frameGray;
	while (1) {
		videoCapture >> frame;

		if (frame.empty()) {
			continue;
		}
		cv::flip(frame, frame, 1);
		cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
		faceCascade.detectMultiScale(frameGray, facesRect, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(0, 0));

		for (int i = 0; i < facesRect.size(); i++) {
			cv::Rect faceRect = facesRect.at(i);
			cv::rectangle(frame, facesRect.at(i), cv::Scalar(0, 0, 255), 2);

			// split face to left and right side
			cv::Mat rightSideFace = frame(cv::Rect(faceRect.x, faceRect.y, faceRect.width / 2, faceRect.height));
			cv::Mat leftSideFace = frame(cv::Rect(faceRect.x + faceRect.width / 2, faceRect.y, faceRect.width / 2, faceRect.height));

			rightEyeCascade.detectMultiScale(rightSideFace, rightEyesRects, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));
			leftEyeCascade.detectMultiScale(leftSideFace, leftEyesRects, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));

			if (rightEyesRects.size() < 1 || leftEyesRects.size() < 1) {
				continue;
			}

			// check whether eyes is open
			vector<cv::Rect> leftEyesOpen;
			vector<cv::Rect> rightEyesOpen;

			eyeCascade.detectMultiScale(leftSideFace, leftEyesOpen, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));
			eyeCascade.detectMultiScale(rightSideFace, rightEyesOpen, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));

			if (rightEyesOpen.size() > 0 || leftEyesOpen.size() > 0) {
				cout << "Eyes are open";
			}
			else {
				cout << "Eyes are close";
			}

			cout << endl;

			cv::Rect rightEyeRect = rightEyesRects.at(0);
			cv::Rect leftEyeRect = leftEyesRects.at(0);

			cv::rectangle(rightSideFace, rightEyeRect, cv::Scalar(0, 0, 255), 2);
			cv::rectangle(leftSideFace, leftEyeRect, cv::Scalar(0, 0, 255), 2);

			cv::imshow("left", leftSideFace);
			cv::imshow("right", rightSideFace);
		}

		cv::imshow(MAIN_WINDOW_NAME, frame);

		if (cv::waitKey(1) == 27) {
			break;
		}
	}
}
