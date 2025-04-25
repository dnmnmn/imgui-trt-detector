//
// Created by Dongmin on 25. 3. 13.
//

#ifndef COLORMAP_H
#define COLORMAP_H
#include <opencv2/opencv.hpp>

class ColorMap {
public:
    ColorMap();
    ~ColorMap() {custom_map_.release();};
    cv::Mat GetColorMap(cv::Mat _image);
    cv::Mat GetColorOverlay(cv::Mat _image, cv::Mat _overlay, float _alpha);
private:
    cv::Mat custom_map_;
};

#endif //COLORMAP_H
