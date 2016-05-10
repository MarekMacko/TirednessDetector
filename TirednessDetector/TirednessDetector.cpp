#include "stdafx.h"

using namespace std;

int main() {
	const string WINDOW_NAME = "Window";
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}

	while (1) {
		cv::Mat frame, fliped;
		cap >> frame;
		cv::flip(frame, fliped, 1);
		cv::imshow(WINDOW_NAME, fliped);

		if (cv::waitKey(33) == 27) {
			break;
		}
	}
}


