#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <Windows.h>

#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

using namespace cv;
using std::vector;

int main2()
{
	// KalmanFilter (int dynamParams/stateParams, int measureParams, int controlParams=0, int type=CV_32F)
	KalmanFilter KF(4, 2, 0);
	POINT mousePos;
	GetCursorPos(&mousePos);

	// intialization of KF...
	// Use a comma-separated initializer:
	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);

	// transitionMatrix is like:
	// [ 1 0 1 0 ]
	// [ 0 1 0 1 ]
	// [ 0 0 1 0 ]
	// [ 0 0 0 1 ]

	// In the usual case measurement comes with some uncertainty, like GPS location.
	Mat_<float> measurement(2, 1); measurement.setTo(Scalar(0)); 

	//true state at time k is evolved from the state at (k − 1) 
	//predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k), where A is the state-transitional model, 
	//B is control-input model, u is control vector
	KF.statePre.at<float>(0) = static_cast<float>(mousePos.x);
	KF.statePre.at<float>(1) = static_cast<float>(mousePos.y);
	KF.statePre.at<float>(2) = 0.0;
	KF.statePre.at<float>(3) = 0.0;

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(10));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// Image to show mouse tracking
	Mat img(600, 800, CV_8UC3);
	vector<Point> mousev, kalmanv;
	mousev.clear();
	kalmanv.clear();

	//main loop
	while (true)
	{
		// First predict, to update the internal statePre variable
		Mat prediction = KF.predict();
		//Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

		// Get mouse point
		GetCursorPos(&mousePos);
		measurement(0) = static_cast<float>(mousePos.x);
		measurement(1) = static_cast<float>(mousePos.y);

		// The update phase, compare the difference of prediction and measurement
		Mat estimated = KF.correct(measurement);

		Point statePt((int)(estimated.at<float>(0)), (int)(estimated.at<float>(1)));
		Point measPt((int)(measurement(0)), (int)(measurement(1)));

		// plot points
		imshow("mouse kalman", img);
		img = Scalar::all(0);

		mousev.push_back(measPt); // the real mouse movement locus
		kalmanv.push_back(statePt); // the estimated pt by kalman filter
		drawCross(statePt, Scalar(255, 255, 255), 5);
		drawCross(measPt, Scalar(0, 0, 255), 5);

		for (int i = 0; i < mousev.size() - 1; i++)
			line(img, mousev[i], mousev[i + 1], Scalar(255, 255, 0), 1);

		for (int i = 0; i < kalmanv.size() - 1; i++)
			line(img, kalmanv[i], kalmanv[i + 1], Scalar(0, 155, 255), 1);

		waitKey(10);
	}
	return 0;
}