#include <opencv2/opencv.hpp>  // OpenCV DNN
#include <iostream>
#include <fstream>
#include <H5Cpp.h>  // HDF5
#include <Eigen/Dense>  // Eigen

int main() {
    // Paths to files
    std::string model_path = "C:\\Users\\Administrator\\Documents\\Coding Projects\\Trials\\ONNX\\resnet34.onnx";  // Use .onnx instead of .dnn
    std::string hdf5_path = "C:\\Users\\Administrator\\Documents\\Coding Projects\\zip Repos\\Mammogram\\all_mias_scans.h5";
    std::string info_txt_path = "C:\\Users\\Administrator\\Documents\\Coding Projects\\zip Repos\\Mammogram\\all-mias\\Info.txt";

    try {
        // 1. Load the ONNX model using OpenCV
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            throw std::runtime_error("Failed to load the ONNX model from: " + model_path);
        }
        std::cout << "ONNX model loaded successfully from: " << model_path << "\n";

        // 2. Open the HDF5 file
        H5::H5File file(hdf5_path, H5F_ACC_RDONLY);
        std::cout << "HDF5 file loaded successfully from: " << hdf5_path << "\n";

        // Example: Read a dataset from the HDF5 file
        H5::DataSet dataset = file.openDataSet("X");  // Replace "X" with actual dataset name if different
        H5::DataSpace dataspace = dataset.getSpace();

        // Get dataset dimensions
        hsize_t dims[2];  // Assuming 2D dataset (adjust if needed)
        dataspace.getSimpleExtentDims(dims, nullptr);
        std::cout << "Dataset dimensions: " << dims[0] << " x " << dims[1] << std::endl;

        // Check if the dataset is reasonable
        if (dims[1] > 1000000) {
            std::cerr << "Dataset too large, please check dimensions!" << std::endl;
            return -1;
        }

        // Read data into Eigen matrix
        Eigen::MatrixXf data(dims[0], dims[1]);
        dataset.read(data.data(), H5::PredType::NATIVE_FLOAT);

        // Normalize or preprocess data (if required)
        data = data / 255.0;  // Example normalization

        // 3. Read the Info.txt file
        std::ifstream info_file(info_txt_path);
        if (!info_file.is_open()) {
            throw std::runtime_error("Failed to open Info.txt file at: " + info_txt_path);
        }
        std::cout << "Info.txt contents:\n";
        std::string line;
        while (std::getline(info_file, line)) {
            std::cout << line << std::endl;  // Print metadata
        }
        info_file.close();

        // 4. Preprocess the data for the model using OpenCV
        // Convert Eigen matrix to OpenCV Mat
        cv::Mat image(dims[0], dims[1], CV_32F, data.data());

        // Resize to 224x224 for ResNet34 (or any ResNet model input size)
        cv::resize(image, image, cv::Size(224, 224));

        // Convert OpenCV Mat to blob
        cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(224, 224), cv::Scalar(104.0, 177.0, 123.0), false, false);

        // Set the input to the network
        net.setInput(blob);

        // Run forward pass and get the output
        cv::Mat prob = net.forward();

        // Post-process output
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        std::cout << "Predicted class: " << classIdPoint.x << " with confidence: " << confidence << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
