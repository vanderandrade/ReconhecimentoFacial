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
	
	if (!file)
		CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");

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

	if(images.size() <= 1)
		CV_Error(Error::StsError, "This demo needs at least 2 images to work. Please add more images to your data set!");

	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();
	model->train(images, labels);

	int predictedLabel = model->predict(testSample);

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	return 0;
}