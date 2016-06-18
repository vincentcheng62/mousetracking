/******************************************
* OpenCV Tutorial: Ball Tracking using   *
* Kalman Filter                          *
******************************************/

// Module "core"
#include <opencv2/core/core.hpp>

// Module "highgui"
#include <opencv2/highgui/highgui.hpp>

// Module "imgproc"
#include <opencv2/imgproc/imgproc.hpp>

// Module "video"
#include <opencv2/video/video.hpp>

#include <iostream>
#include <vector>

using std::vector;

// >>>>> Color to be tracked
#define MIN_H_BLUE 200
#define MAX_H_BLUE 300
// <<<<< Color to be tracked

int main() {
	// Camera frame
	cv::Mat frame;      // >>>> Kalman Filter
	int stateSize = 6;
	int measSize = 4;
	int contrSize = 0;

	unsigned int type = CV_32F;
	cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

	cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
	cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
	//cv::Mat procNoise(stateSize, 1, type)
	// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

	// Transition State Matrix A
	// Note: set dT at each processing step!
	// [ 1 0 dT 0  0 0 ]
	// [ 0 1 0  dT 0 0 ]
	// [ 0 0 1  0  0 0 ]
	// [ 0 0 0  1  0 0 ]
	// [ 0 0 0  0  1 0 ]
	// [ 0 0 0  0  0 1 ]
	cv::setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex 0  0    0 0    0 ]
	// [ 0  Ey 0    0 0    0 ]
	// [ 0  0  Ev_x 0 0    0 ]
	// [ 0  0  0    1 Ev_y 0 ]
	// [ 0  0  0    0 1    Ew ]
	// [ 0  0  0    0 0    Eh ]
	//cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 2.0f;
	kf.processNoiseCov.at<float>(21) = 1.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
	// <<<< Kalman Filter
	// Camera Index
	int idx = 0;      // Camera Capture
	cv::VideoCapture cap;      // >>>>> Camera Settings
	if (!cap.open(idx))
	{
		std::cout << "Webcam not connected.\n" << "Please verify\n";
		return EXIT_FAILURE;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
	// <<<<< Camera Settings

	std::cout << "\nHit 'q' to exit...\n";
	char ch = 0;      double ticks = 0;    bool found = false;      int notFoundCount = 0;            // >>>>> Main loop  
	while (ch != 'q' || ch != 'Q')
	{
		double precTick = ticks;
		ticks = (double)cv::getTickCount();

		double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

		// Frame acquisition
		cap >> frame;

		cv::Mat res;
		frame.copyTo(res);

		if (found)
		{
			// >>>> Matrix A
			kf.transitionMatrix.at<float>(2) = dT;
			kf.transitionMatrix.at<float>(9) = dT;
			// <<<< Matrix A

			std::cout << "dT:" << std::endl << dT << std::endl;

			state = kf.predict();
			std::cout << "State post:" << std::endl << state << std::endl;
			cv::Rect predRect;
			predRect.width = state.at<float>(4);
			predRect.height = state.at<float>(5);
			predRect.x = state.at<float>(0) - predRect.width / 2;
			predRect.y = state.at<float>(1) - predRect.height / 2;
			cv::Point center;          center.x = state.at<float>(0);
			center.y = state.at<float>(1);
			cv::circle(res, center, 2, CV_RGB(255, 0, 0), -1);
			cv::rectangle(res, predRect, CV_RGB(255, 0, 0), 2);
		}         // >>>>> Noise smoothing

		cv::Mat blur;
		cv::GaussianBlur(frame, blur, cv::Size(5, 5), 3.0, 3.0);
		// <<<<< Noise smoothing         // >>>>> HSV conversion
		//cv::Mat frmHsv;
		//cv::cvtColor(blur, frmHsv, CV_BGR2HSV);
		// <<<<< HSV conversion         // >>>>> Color Thresholding
		// Note: change parameters for different colors
		cv::Mat rangeRes = cv::Mat::zeros(frame.size(), CV_8UC1);
		//cv::inRange(frmHsv, cv::Scalar(MIN_H_BLUE / 2, 100, 80),
			//cv::Scalar(MAX_H_BLUE / 2, 255, 255), rangeRes);
		//cv::inRange(frmHsv, cv::Scalar(0, 100, 0), // change value range from 0 to 50 to find black ball
			//cv::Scalar(360, 255, 30), rangeRes);
		// <<<<< Color Thresholding         // >>>>> Improving the result
		//cv::erode(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
		//cv::dilate(rangeRes, rangeRes, cv::Mat(), cv::Point(-1, -1), 2);
		// <<<<< Improving the result         // Thresholding viewing

		// Circle detection
		cv::Mat blurgray;
		cv::cvtColor(blur, blurgray, CV_BGR2GRAY);
		vector<cv::Vec3f> circles;

		/// Apply the Hough Transform to find the circles
		HoughCircles(blurgray, circles, CV_HOUGH_GRADIENT, 1,30, 200, 500, 0, 0);

		/// Draw the circles detected
		cv::Mat display;
		blurgray.copyTo(display);
		vector<cv::Rect> ballsBox;
		for (size_t i = 0; i < circles.size(); i++)
		{
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			cv::circle(display, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			cv::circle(display, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
			cv::Rect bBox(center.x-radius, center.y-radius, radius*2, radius*2);
			ballsBox.push_back(bBox);
		}

		cv::imshow("Threshold", display);         // >>>>> Contours detection
		//vector<vector<cv::Point> >  contours;
		//cv::findContours(rangeRes, contours, CV_RETR_EXTERNAL,
		//	CV_CHAIN_APPROX_NONE);
		// <<<<< Contours detection         // >>>>> Filtering

		vector<vector<cv::Point> > balls;

		//for (size_t i = 0; i < contours.size(); i++)       {
		//	cv::Rect bBox;
		//	bBox = cv::boundingRect(contours[i]);
		//	float ratio = (float)bBox.width / (float)bBox.height;
		//	if (ratio > 1.0f)
		//		ratio = 1.0f / ratio;

		//	// Searching for a bBox almost square
		//	if (ratio > 0.75 && bBox.area() >= 400)
		//	{
		//		balls.push_back(contours[i]);
		//		ballsBox.push_back(bBox);
		//	}
		//}
		// <<<<< Filtering

		std::cout << "Balls found:" << ballsBox.size() << std::endl;         // >>>>> Detection result
		for (size_t i = 0; i < ballsBox.size(); i++)
		{
			//cv::drawContours(res, balls, i, CV_RGB(20, 150, 20), 1);
			cv::rectangle(res, ballsBox[i], CV_RGB(0, 255, 0), 2);

			cv::Point center;
			center.x = ballsBox[i].x + ballsBox[i].width / 2;
			center.y = ballsBox[i].y + ballsBox[i].height / 2;
			cv::circle(res, center, 2, CV_RGB(20, 150, 20), -1);

			stringstream sstr;
			sstr << "(" << center.x << "," << center.y << ")";
			cv::putText(res, sstr.str(),
				cv::Point(center.x + 3, center.y - 3),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20, 150, 20), 2);
		}
		// <<<<< Detection result         // >>>>> Kalman Update
		if (ballsBox.size() == 0)
		{
			notFoundCount++;
			std::cout << "notFoundCount:" << notFoundCount << std::endl;          if (notFoundCount >= 10)
			{
				found = false;
			}
			else
				kf.statePost = state;
		}
		else
		{
			notFoundCount = 0;

			meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
			meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
			meas.at<float>(2) = (float)ballsBox[0].width;
			meas.at<float>(3) = (float)ballsBox[0].height;

			if (!found) // First detection!
			{
				// >>>> Initialization
				kf.errorCovPre.at<float>(0) = 1; // px
				kf.errorCovPre.at<float>(7) = 1; // px
				kf.errorCovPre.at<float>(14) = 1;
				kf.errorCovPre.at<float>(21) = 1;
				kf.errorCovPre.at<float>(28) = 1; // px
				kf.errorCovPre.at<float>(35) = 1; // px

				state.at<float>(0) = meas.at<float>(0);
				state.at<float>(1) = meas.at<float>(1);
				state.at<float>(2) = 0;
				state.at<float>(3) = 0;
				state.at<float>(4) = meas.at<float>(2);
				state.at<float>(5) = meas.at<float>(3);
				// <<<< Initialization

				found = true;
			}
			else
				kf.correct(meas); // Kalman Correction

			std::cout << "Measure matrix:" << std::endl << meas << std::endl;
		}
		// <<<<< Kalman Update

		// Final result
		cv::imshow("Risultato finale", res);

		// User key
		ch = cv::waitKey(10);
	}
	// <<<<< Main loop

	return EXIT_SUCCESS;
}