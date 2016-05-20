#include "stdafx.h"


int main() {
	FaceDetect *faceDetect = new FaceDetect();
	faceDetect->run();
	free(faceDetect);
}


