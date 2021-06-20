#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>
//
// http://stackoverflow.com/a/21444792/1072039
template<class Archive>
void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
{
    int cols, rows, type;
    bool continuous;

    if (Archive::is_saving::value) {
        cols = mat.cols; rows = mat.rows; type = mat.type();
        continuous = mat.isContinuous();
    }

    ar & cols & rows & type & continuous;

    if (Archive::is_loading::value)
        mat.create(rows, cols, type);

    if (continuous) {
        const unsigned int data_size = rows * cols * mat.elemSize();
        ar & boost::serialization::make_array(mat.ptr(), data_size);
    } else {
        const unsigned int row_size = cols*mat.elemSize();
        for (int i = 0; i < rows; i++) {
            ar & boost::serialization::make_array(mat.ptr(i), row_size);
        }
    }
}
void serializeMat(const std::string& filename, cv::Mat& mat) {
    std::ofstream ofs(filename.c_str());
    boost::archive::binary_oarchive oa(ofs);
    serialize(oa, mat, 0);
}

void deserializeMat(cv::Mat& mat, const std::string& filename) {
    std::ifstream ifs(filename.c_str());
    if (!ifs)
    {
        std::cerr << "Fail to find centroids" << std::endl;
        return;
    }

    boost::archive::binary_iarchive ia(ifs);
    serialize(ia, mat, 0);
}



