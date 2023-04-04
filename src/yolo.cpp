#include "yolo.h"
#include "string"
#include <sys/stat.h>
#include <iostream>
#include "logging.h"
#include <fstream>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_utils.h"

using namespace nvinfer1;
using namespace std;

static Logger gLogger;


YoloModel::YoloModel(string engine_path, int batch_size) {
    if (!file_exists(engine_path)) {
        cerr << "No such file: " << engine_path << endl;
    } else {
        cudaSetDevice(0);
        size_t size = 0;
        ifstream file(engine_path, ios::binary);
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        char *trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream, size);

        delete[] trtModelStream;
        assert(engine != nullptr);
        assert(runtime != nullptr);
        context = engine->createExecutionContext();
        assert(context != nullptr);
        assert(engine->getNbBindings() == 5);
        this->batch_size = batch_size;
        // Create GPU buffers on device
        num_dets_size *= batch_size;
        det_boxes_size *= batch_size;
        det_scores_size *= batch_size;
        det_classes_size *= batch_size;
        // Images -> 0 float
        CUDA_CHECK(cudaMalloc((void **) &buffers[0], batch_size * size_image_dst * sizeof(float)));
        // num_dets -> 1 int
        CUDA_CHECK(cudaMalloc((void **) &buffers[1], num_dets_size * sizeof(int)));
        // det_boxes -> 2 float
        CUDA_CHECK(cudaMalloc((void **) &buffers[2], det_boxes_size * sizeof(float)));
        // det_scores -> 3 float
        CUDA_CHECK(cudaMalloc((void **) &buffers[3], det_scores_size * sizeof(float)));
        // det_classes -> 4 int
        CUDA_CHECK(cudaMalloc((void **) &buffers[4], det_classes_size * sizeof(int)));
        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));
        // prepare input data cache in device memory
        CUDA_CHECK(cudaMalloc((void **) &img_device, max_input_buffer_size * n_channels));
    }
}

YoloModel::~YoloModel() {
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    CUDA_CHECK(cudaFree(buffers[2]));
    CUDA_CHECK(cudaFree(buffers[3]));
    CUDA_CHECK(cudaFree(buffers[4]));

    // order is matter
    delete context;
    delete engine;
    delete runtime;
};

bool YoloModel::file_exists(const string &name) {
    struct stat buffer{};
    return (stat(name.c_str(), &buffer) == 0);
}

void YoloModel::preprocess(const vector<cv::Mat> &original_images) const {
    cv::Mat nchw = cv::dnn::blobFromImages(
            original_images,
            1 / 255.0f,
            cv::Size(640, 640),
            cv::Scalar(0.5, 0.5, 0.5),
            false
    );
    CUDA_CHECK(cudaMemcpyAsync(
            this->buffers[0],
            nchw.data,
            this->size_image_dst * batch_size * sizeof(float),
            cudaMemcpyHostToDevice,
            this->stream)
    );
}

vector<vector<Detection>> YoloModel::forward(const vector<cv::Mat> &original_images) {
    preprocess(original_images);
    context->executeV2((void **) buffers);
    // Get num_dets
    int dets[this->batch_size];
    CUDA_CHECK(cudaMemcpyAsync(dets, buffers[1],
                               num_dets_size * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    // Get boxes
    float boxes[det_boxes_size];
    CUDA_CHECK(cudaMemcpyAsync(boxes, buffers[2],
                               det_boxes_size * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream));

    // Get scores
    float scores[det_scores_size];
    CUDA_CHECK(cudaMemcpyAsync(scores, buffers[3],
                               det_scores_size * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream));
    // Get classes
    int labels[det_classes_size];
    CUDA_CHECK(cudaMemcpyAsync(labels, buffers[4],
                               det_classes_size * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);

    vector<vector<Detection>> batch_res(batch_size);
    postprocess(batch_res, original_images, dets, boxes, scores, labels);
    return batch_res;
}

void YoloModel::postprocess(
        vector<std::vector<Detection>> &batch_res,
        const vector<cv::Mat> &original_images,
        const int *dets,
        const float *boxes,
        const float *scores,
        const int *labels) const {
    for (int i = 0; i < original_images.size(); ++i) {
        float original_h = (float) original_images[i].rows;
        float original_w = (float) original_images[i].cols;
        float scale = 1.0f / (std::min((float) input_h / original_h, (float) input_w / original_w));
        int x_offset = (input_w * scale - original_w) / 2;
        int y_offset = (input_h * scale - original_h) / 2;
        auto &image_res = batch_res[i];
        for (int n = 0; n < dets[i]; ++n) {
            Detection det{};
            int detection_offset = top_n * i;
            det.conf = scores[detection_offset + n];
            det.class_id = labels[detection_offset + n];

            float x0 = (boxes[4 * detection_offset + (n * 4)]);
            float y0 = (boxes[4 * detection_offset + (n * 4) + 1]);
            float x1 = (boxes[4 * detection_offset + (n * 4) + 2]);
            float y1 = (boxes[4 * detection_offset + (n * 4) + 3]);

            x0 *= scale;
            x0 -= x_offset;
            y0 *= scale;
            y0 -= y_offset;
            x1 *= scale;
            x1 -= x_offset;
            y1 *= scale;
            y1 -= y_offset;

            x0 = std::max(std::min(x0, (float) (original_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float) (original_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float) (original_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float) (original_h - 1)), 0.f);

            det.bbox[0] = x0;
            det.bbox[1] = y0;
            det.bbox[2] = x1;
            det.bbox[3] = y1;
            image_res.push_back(det);
        }
    }
}
