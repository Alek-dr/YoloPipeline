#include <iostream>
#include <dirent.h>
#include "string"
#include "vector"
#include "src/yolo.h"

using namespace std;

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent *p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            std::string out_name = p_dir_name;
            file_names.push_back(out_name + "/" + cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

void show_detection(Detection det) {
    std::cout << "Label: " << det.class_id << std::endl;
    std::cout << "Score: " << det.conf << std::endl;
    std::cout << "Box: [" << det.bbox[0] << " " << det.bbox[1] << " " << det.bbox[2] << " " << det.bbox[3] << "]"
              << std::endl;
}

int main() {
    string img_dir = "./examples";
    // Записать имена файлов в file_names
    vector<string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        cerr << "read_files_in_dir failed." << endl;
        return -1;
    }
    // Считать batch_size изображений
    int batch_size = 2;
    vector<cv::Mat> batch_images;
    for (int i = 0; i < batch_size; ++i) {
        string file_path = file_names[i];
        cv::Mat img = cv::imread(file_path);
        std::cout << file_names[i] << std::endl;
        batch_images.push_back(img);
    }
    // Инициализировать yolo
    YoloModel model = YoloModel("./weights/yolov7-nms-b1.trt", batch_size);
    // warmup
    for (int i = 0; i < 10; ++i) {
        model.forward(batch_images);
    }
    // Инференс
    auto start = std::chrono::system_clock::now();
    vector<vector<Detection>> res = model.forward(batch_images);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // Отрисовка
    int idx = 0;
    vector<vector<Detection>>::iterator img_res_it = res.begin();
    while (img_res_it != res.end()) {
        vector<Detection>::iterator detection_it = img_res_it->begin();
        while (detection_it != img_res_it->end()) {
            int w = detection_it->bbox[2] - detection_it->bbox[0];
            int h = detection_it->bbox[3] - detection_it->bbox[1];
            cv::Rect rect(detection_it->bbox[0],
                          detection_it->bbox[1],
                          w,
                          h);
            cv::rectangle(batch_images[idx], rect, cv::Scalar(0x27, 0xC1, 0x36), 3);
            show_detection(*detection_it);
            ++detection_it;
        }
        cv::imwrite("result_" + std::to_string(idx) + ".jpg", batch_images[idx]);
        ++img_res_it;
        idx++;
    }
    return 0;
}
