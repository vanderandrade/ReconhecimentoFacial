#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    
    return dst;
}

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
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
        exit(1);
    }
    
    string output_folder = ".";
    if (argc == 3)
        output_folder = string(argv[2]);

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

    int height = images[0].rows;
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    
    images.pop_back();
    labels.pop_back();

    Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    
    int predictedLabel = model->predict(testSample);
    
    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    cout << result_message << endl;
    
    Mat eigenvalues = model->getEigenValues();
    Mat W = model->getEigenVectors();
    Mat mean = model->getMean();

    if(argc == 2) {
        imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
    } else {
        imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
    }

    for (int i = 0; i < min(10, W.cols); i++) {
        string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        cout << msg << endl;

        Mat ev = W.col(i).clone();
        Mat grayscale = norm_0_255(ev.reshape(1, height));
        Mat cgrayscale;
        applyColorMap(grayscale, cgrayscale, COLORMAP_JET);

        if(argc == 2) {
            imshow(format("eigenface_%d", i), cgrayscale);
        } else {
            imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
        }
    }

    for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15) {
        Mat evs = Mat(W, Range::all(), Range(0, num_components));
        Mat projection = LDA::subspaceProject(evs, mean, images[0].reshape(1,1));
        Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);

        reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

        if(argc == 2) {
            imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
        } else {
            imwrite(format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
        }
    }

    if(argc == 2) {
        waitKey(0);
    }
    return 0;
}