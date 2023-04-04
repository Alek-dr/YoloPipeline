### Получаем модель

1. Скачиваем по [ссылке](https://github.com/WongKinYiu/yolov7)
2. Конвертация в onnx

```
python export.py --weights ./yolov7-tiny.pt \
    --grid \
    --end2end \
    --simplify \
    --topk-all 100 \
    --iou-thres 0.65 \
    --conf-thres 0.35 \
    --img-size 640 640
```

3. Конвертация в trt  
Рекомендуется устанавливать из tar.gz пакетов

* Установить cundd
  https://developer.nvidia.com/rdp/cudnn-download

* Установить tensorrt
  https://developer.nvidia.com/nvidia-tensorrt-8x-download

enviroments example:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-8.6.0.12:/usr/local/cuda-11.4/targets/x86_64-linux/lib
```

Конвертация с помощью trtexec
  
```
/usr/local/TensorRT-8.6.0.12/bin/trtexec --onnx=yolov7-b2.onnx --saveEngine=yolov7-nms-b2.trt --fp16
```