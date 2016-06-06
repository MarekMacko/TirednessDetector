#include "stdafx.h"

FaceDetect::FaceDetect() {
}

FaceDetect::~FaceDetect() {
}

int FaceDetect::initialize(void) {
	//	if (!videoCapture.open(VIDEO_FILE)) {
	if (!videoCapture.open(0)) {
		showError("Error: VideoCapture open!");
		return -1;
	}
	if (!faceCascade.load(CASCADE_FACE_FILE)) {
		showError("Error: FaceCascade load!");
		return -1;
	}
	if (!eyeCascade.load(CASCADE_EYE_FILE)) {
		showError("Error: EyeCascade load!");
		return -1;
	}
	if (!leftEyeCascade.load(CASCADE_LEFT_EYE_FILE)) {
		showError("Error: Right eye cascade load!");
		return -1;
	}
	if (!rightEyeCascade.load(CASCADE_RIGHT_EYE_FILE)) {
		showError("Error: Left eye cascade load!");
		return -1;
	}

	cv::namedWindow(MAIN_WINDOW_NAME, cv::WINDOW_AUTOSIZE);
	return 0;
}

void FaceDetect::showError(string errorMessage) {
	cerr << errorMessage << endl;
	system("pause");
}

void FaceDetect::run(void) {
	cv::Mat frame, image;
	cv::Rect rightEyeRect;

	bool faceIsTracked = false;
	cv::Mat hsv, hue, mask, hist, histimg = cv::Mat::zeros(200, 320, CV_8UC3), backproj;
	int vmin = 10, vmax = 256, smin = 30;
	cv::Rect faceRect;
	int hsize = 16;
	float hranges[] = { 0, 180 };
	const float *phranges = hranges;
	cv::Rect trackWindow;
	int trackObject = -1;
	bool backprojMode = false;

	if (initialize() != 0) {
		cerr << "Error: FaceDetect initialize!";
		return;
	}

	startTime = clock();
	while (true) {
		videoCapture >> frame;

		if (frame.empty()) {
			break;
		}

		frame.copyTo(image);
		
		cv::flip(frame, frame, 1);

		// first we must detect face to tracking
		if (!faceIsTracked) {
			faceRect = getFaceRect(image);
			if (faceRect.area() < 1) {
				cout << "Detect: No face detected" << endl;
				cv::imshow(MAIN_WINDOW_NAME, image);
				continue;
			}

			cv::Rect rightEyeRect = getRightEyeRect(image, faceRect);
			cv::Rect leftEyeRect = getLeftEyeRect(image, faceRect);

			if (rightEyeRect.area() == 0 || leftEyeRect.area() == 0) {
				cout << "Detect: Eyes not detecked" << endl;
				continue;
			}

			faceIsTracked = true;
		} else {
			int _vmin = vmin, _vmax = vmax;

			cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
			cv::inRange(
				hsv,
				cv::Scalar(0, smin, MIN(_vmin, _vmax)),
				cv::Scalar(180, 256, MAX(_vmin, _vmax)), mask);
			int ch[] = {0, 0};

			hue.create(hsv.size(), hsv.depth());

			cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);

			if (trackObject < 0) {
				cv::Mat roi(hue, faceRect), maskRoi(mask, faceRect);
				cv::calcHist(&roi, 1, 0, maskRoi, hist, 1, &hsize, &phranges);
				cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

				trackWindow = faceRect;
				trackObject = 1;

				histimg = cv::Scalar::all(0);
				int binW = histimg.cols / hsize;
				cv::Mat buf(1, hsize, CV_8UC3);
				for (int i = 0; i < hsize; i++) {
					buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180. / hsize), 255, 255);
				}
				
				cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);

				for (int i = 0; i < hsize; i++) {
					int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);
					rectangle(histimg, cv::Point(i*binW, histimg.rows),
						cv::Point((i + 1)*binW, histimg.rows - val),
						cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8);
				}
			}

			calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
			backproj &= mask;
			cv::TermCriteria termCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1);
			/*cv::RotatedRect trackBox =*/ cv::meanShift(backproj, trackWindow, termCriteria);
			cv::rectangle(image, trackWindow, cv::Scalar(0, 0, 255), 2);

			cv::imshow("backproj", backproj);

			cv::Mat faceImage = image(trackWindow);

			cv::Rect faceImageRect(0, 0, faceImage.cols, faceImage.rows);
			cv::Rect rightEyeRect = getRightEyeRect(faceImage, faceImageRect);
			cv::Rect leftEyeRect = getLeftEyeRect(faceImage, faceImageRect);

			if (rightEyeRect.area() == 0 || leftEyeRect.area() == 0) {
				cout << "Tracking: Eyes not detecked" << endl;
				faceIsTracked = false;
				continue;
			}

			cv::Rect leftEyeOpenRect = getLeftOpenEyeRect(faceImage, faceImageRect);
			cv::Rect rightEyeOpenRect = getRightOpenEyeRect(faceImage, faceImageRect);

			if (rightEyeOpenRect.area() > 0 || leftEyeOpenRect.area() > 0) {
				cout << "tracking: eyes are open" << endl;
			} else {
				blinksCounter++;
				cout << "tracking: eyes are close " << blinksCounter << endl;
			}
		}

		setInfoToFrame(image);
		cv::imshow(MAIN_WINDOW_NAME, image);

		if (cv::waitKey(1) == 27) {
			break;
		}
	}
}


cv::Rect FaceDetect::getFaceRect(const cv::Mat image) {
	cv::Mat grayImage;
	vector<cv::Rect> facesRects;

	if (image.empty()) {
		return cv::Rect(0, 0, 0, 0);
	}

	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
	faceCascade.detectMultiScale(grayImage, facesRects, 1.1, 3, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(30, 30), cv::Size(400, 400));

	if (facesRects.size() > 0) {
		return facesRects.at(0);
	} else {
		return cv::Rect(0, 0, 0, 0);
	}
}

cv::Rect FaceDetect::getRightEyeRect(cv::Mat image, const cv::Rect faceRect) {
	vector<cv::Rect> rightEyesRects;

	cv::Mat rightSideFace = image(cv::Rect(faceRect.x, faceRect.y, faceRect.width / 2 + faceRect.width * 0.1, faceRect.height));
	rightEyeCascade.detectMultiScale(rightSideFace, rightEyesRects, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));	
	
	if (rightEyesRects.size() > 0) {
		cv::Rect rightEyeRect = rightEyesRects.at(0);
		float scale = 0.3;
		
		int scaleX = rightEyeRect.width * scale;
		int scaleY = rightEyeRect.height * scale;
		rightEyeRect.x -= scaleX;
		rightEyeRect.y -= scaleY;
		rightEyeRect.width += 2 * scaleX;
		rightEyeRect.height += 2 * scaleY;
		cv::rectangle(rightSideFace, rightEyeRect, cv::Scalar(255, 0, 0), 2);
		imshow("right", rightSideFace);
		return rightEyeRect;
	} else {
		return cv::Rect(0, 0, 0, 0);
	}
}

cv::Rect FaceDetect::getLeftEyeRect(cv::Mat image, const cv::Rect faceRect) {
	vector<cv::Rect> leftEyesRects;

	cv::Rect leftFaceRect(faceRect.x + faceRect.width / 2 - faceRect.width * 0.1, faceRect.y, faceRect.width / 2, faceRect.height);
	cv::Mat leftSideFace = image(leftFaceRect);
	leftEyeCascade.detectMultiScale(leftSideFace, leftEyesRects, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(0, 0));

	if (leftEyesRects.size() > 0) {
		cv::Rect leftEyeRect = leftEyesRects.at(0);
		float scale = 0.3;
		int scaleX = leftEyeRect.width * scale;
		int scaleY = leftEyeRect.height * scale;
		leftEyeRect.x -= scaleX;
		leftEyeRect.y -= scaleY;
		leftEyeRect.width += 2 * scaleX;
		leftEyeRect.height += 2 * scaleY;
		cv::rectangle(leftSideFace, leftEyeRect, cv::Scalar(0, 255, 0), 2);
		imshow("left", leftSideFace);
		return leftEyeRect;
	} else {
		return cv::Rect(0, 0, 0, 0);
	}
}

cv::Rect FaceDetect::getRightOpenEyeRect(const cv::Mat image, const cv::Rect faceRect) {
	vector<cv::Rect> rightEyesOpen;
	
	cv::Mat rightSideFace = image(cv::Rect(faceRect.x, faceRect.y, faceRect.width / 2 + faceRect.width * 0.05, faceRect.height));
	eyeCascade.detectMultiScale(rightSideFace, rightEyesOpen, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(10, 10));

	if (rightEyesOpen.size() > 0) {
		return rightEyesOpen.at(0);
	} else {
		return cv::Rect(0, 0, 0, 0);
	}
}

cv::Rect FaceDetect::getLeftOpenEyeRect(const cv::Mat image, const cv::Rect faceRect) {
	vector<cv::Rect> leftEyesOpen;
	
	cv::Mat leftSideFace = image(cv::Rect(faceRect.x + faceRect.width / 2 - faceRect.width * 0.05, faceRect.y, faceRect.width / 2, faceRect.height));
	eyeCascade.detectMultiScale(leftSideFace, leftEyesOpen, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(10, 10));

	if (leftEyesOpen.size() > 0) {
		return leftEyesOpen.at(0);
	} else {
		return cv::Rect(0, 0, 0, 0);
	}
}

void FaceDetect::setInfoToFrame(cv::Mat &frame) {
	cv::Scalar textColor(0, 0, 255);
	int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
	double fontScale = 1;
	int thickness = 1;

	cv::putText(frame,
		"Total blinks: " + to_string(blinksCounter),
		cv::Point(10, 20),
		fontFace,
		fontScale,
		textColor,
		thickness,
		CV_AA);

	float elapsedSeconds = (clock() - startTime) / CLOCKS_PER_SEC;

	cv::putText(frame,
		"Total time elapsed: " + to_string(elapsedSeconds),
		cv::Point(10, 40),
		fontFace,
		fontScale,
		textColor,
		thickness,
		CV_AA);

	float blinksForMinute = 0;
	if (elapsedSeconds > 0) {
		blinksForMinute = blinksCounter / elapsedSeconds * 60.f;
	}

	cv::putText(frame,
		"Blink per minute: " + to_string(blinksForMinute),
		cv::Point(10, 60),
		fontFace,
		fontScale,
		textColor,
		thickness,
		CV_AA);
}