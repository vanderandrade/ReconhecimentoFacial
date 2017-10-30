#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}

	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
int main(int argc, const char *argv[]) {
	if (argc != 2) {
		cout << "usage: " << argv[0] << " <csv.ext>" << endl;
		exit(1);
	}

	string fn_csv = string(argv[1]);
	
	vector<Mat> images;
	vector<int> labels;

	try {
		read_csv(fn_csv, images, labels);
	} catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	if(images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}
	
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();
	// The following lines create an LBPH model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	//
	// The LBPHFaceRecognizer uses Extended Local Binary Patterns
	// (it's probably configurable with other operators at a later
	// point), and has the following default values
	//
	// radius = 1
	// neighbors = 8
	// grid_x = 8
	// grid_y = 8
	//
	// So if you want a LBPH FaceRecognizer using a radius of
	// 2 and 16 neighbors, call the factory method with:
	//
	// cv::face::LBPHFaceRecognizer::create(2, 16);
	//
	// And if you want a threshold (e.g. 123.0) call it with its default values:
	//
	// cv::face::LBPHFaceRecognizer::create(1,8,8,8,123.0)
	//
	Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
	model->train(images, labels);
	// The following line predicts the label of a given
	// test image:
	int predictedLabel = model->predict(testSample);
	//
	// To get the confidence of a prediction call the model with:
	//
	// int predictedLabel = -1;
	// double confidence = 0.0;
	// model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// First we'll use it to set the threshold of the LBPHFaceRecognizer
	// to 0.0 without retraining the model. This can be useful if
	// you are evaluating the model:
	//
	model->setThreshold(0.0);
	// Now the threshold of this model is set to 0.0. A prediction
	// now returns -1, as it's impossible to have a distance below
	// it
	predictedLabel = model->predict(testSample);
	cout << "Predicted class = " << predictedLabel << endl;
	// Show some informations about the model, as there's no cool
	// Model data to display as in Eigenfaces/Fisherfaces.
	// Due to efficiency reasons the LBP images are not stored
	// within the model:
	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
	model->getRadius(),
	model->getNeighbors(),
	model->getGridX(),
	model->getGridY(),
	model->getThreshold());
	cout << model_info << endl;
	// We could get the histograms for example:
	vector<Mat> histograms = model->getHistograms();
	// But should I really visualize it? Probably the length is interesting:
	cout << "Size of the histograms: " << histograms[0].total() << endl;
	return 0;
}