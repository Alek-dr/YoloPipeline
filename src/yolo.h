#ifndef YOLOPIPELINE_YOLO_H
#define YOLOPIPELINE_YOLO_H

#include "string"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

struct Detection {
    //center_x center_y w h
    float bbox[4];
    float conf;
    int class_id;
};

class YoloModel {
private:
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context;
    cudaStream_t stream;
    float *buffers[5];
    uint8_t *img_device = nullptr;
    int input_h = 640;
    int input_w = 640;
    const int n_channels = 3;
    const size_t size_image_dst = input_w * input_h * n_channels;
    int max_input_buffer_size = 3000 * 3000;
    const int top_n = 100;
    int num_dets_size = 1;
    int det_scores_size = top_n;
    int det_boxes_size = top_n * 4;
    int det_classes_size = top_n;

    static inline bool file_exists(const std::string &);

protected:
    int batch_size = 1;


public:
    YoloModel(std::string, int);

    ~YoloModel();

//    static const int output_size = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
    static const int output_size = 100 * sizeof(int);

    void preprocess(const std::vector<cv::Mat> &) const;
//
    void postprocess(std::vector<std::vector<Detection>> &batch_res, const std::vector<cv::Mat> &, const int *dets, const float *boxes, const float *scores, const int *labels) const;
//
    std::vector<std::vector<Detection>> forward(const std::vector<cv::Mat> &original_images);

};

#endif //YOLOPIPELINE_YOLO_H
