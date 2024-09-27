#include <stdio.h>
#include <mutex>
#include "rknn_api.h"

#include "postprocess.h"
#include "preprocess.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "coreNum.hpp"
#include "rkYolov5s.hpp"

#include "sort.h"

#include <queue>
#include <vector>
#include <iostream>
#include <map>
#include <set>

const int LINE_Y = 300; // 越线的 Y 坐标
int count_up = 0;       // 上行计数
int count_down = 0;     // 下行计数
int per_num = 0;     // 下行计数

std::vector<int> previous_ids; // 存储上一帧的 ID
std::set<int> crossed_ids;
std::set<int> crossed_up;   // 记录越线向上状态的 ID
std::set<int> crossed_down; // 记录越线向下状态的 ID
std::map<int, int> previous_y_positions;

bool isPointInsideRectangle(const cv::Rect& rect, const cv::Point& point) {
    return (point.x >= rect.x && point.x <= (rect.x + rect.width) &&
            point.y >= rect.y && point.y <= (rect.y + rect.height));
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

    // printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
    //        "type=%s, qnt_type=%s, "
    //        "zp=%d, scale=%f\n",
    //        attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
    //        attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
    //        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

rkYolov5s::rkYolov5s(const std::string &model_path)
{
    this->model_path = model_path;
    nms_threshold = NMS_THRESH;      // 默认的NMS阈值
    box_conf_threshold = BOX_THRESH; // 默认的置信度阈值
}

int rkYolov5s::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);
    // 模型参数复用/Model parameter reuse
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // 获取模型输入输出参数/Obtain the input and output parameters of the model
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数/Set the input parameters
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // 设置输出参数/Set the output parameters
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    return 0;
}

rknn_context *rkYolov5s::get_pctx()
{
    return &ctx;
}

cv::Mat rkYolov5s::infer(cv::Mat &orig_img)
{
    std::lock_guard<std::mutex> lock(mtx);
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    cv::Size target_size(width, height);
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);
    // 计算缩放比例/Calculate the scaling ratio
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;
    static TrackingSession *sess = CreateSession(2, 3, 0.01);
    // 图像缩放/Image scaling
    if (img_width != width || img_height != height)
    {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));
        memset(&dst, 0, sizeof(dst));
        ret = resize_rga(src, dst, img, resized_img, target_size);
        if (ret != 0)
        {
            fprintf(stderr, "resize with rga error\n");
        }
        /*********
        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, resized_img, pads, min_scale, target_size);
        *********/
        inputs[0].buf = resized_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    // 模型推理/Model inference
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    // 后处理/Post-processing
    detect_result_group_t detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                 box_conf_threshold, nms_threshold, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // 绘制框体/Draw the box
    // line(orig_img, cv::Point(0, 300), cv::Point(orig_img.cols, 300), cv::Scalar(0, 0, 255), 2);
    // cv::Rect o_rectangle(600, 100, 200, 150); // (x, y, width, height)
    // cv::rectangle(orig_img, o_rectangle, cv::Scalar(0, 0, 255), 2);
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
        // 打印预测物体的信息/Prints information about the predicted object
        // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
        //        det_result->box.right, det_result->box.bottom, det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        // rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(256, 0, 0, 256), 3);
        // putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }


    // 生成 SORT 所需的格式
    std::vector<DetectionBox> detections;
    for (int i = 0; i < detect_result_group.count; i++)
    {
        detect_result_t *det_result = &(detect_result_group.results[i]);
        if (det_result->prop * 100 >= box_conf_threshold && strcmp(det_result->name, "person") == 0) // 置信度过滤
        {
            DetectionBox detection;
            detection.box = {det_result->box.left, det_result->box.top, 
                             det_result->box.right - det_result->box.left, 
                             det_result->box.bottom - det_result->box.top};
            detection.score = det_result->prop;
            // detection.class_id = 0; /* 适当的类 ID */;
            detections.push_back(detection);
        }
    }

    // 更新 TrackingSession
    auto trks = sess->Update(detections);
    // 绘制跟踪框
    per_num = 0;
    for (const auto& track : trks) {
        int x1 = track.box.x;
        int y1 = track.box.y;
        int x2 = x1 + track.box.width;
        int y2 = y1 + track.box.height;
        cv::Scalar color((track.id * 123) % 256, (track.id * 456) % 256, (track.id * 789) % 256);
        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);

        // 获取文本大小
        std::string text = std::to_string(track.id);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        textSize.width = std::max(textSize.width, static_cast<int>(track.box.width));

        // 矩形的左上角和右下角坐标
        cv::Point textOrigin(x1, y1); // 调整到文本上方
        // cv::Point textOrigin(x1, y1 - textSize.height - 10); // 调整到文本上方
        cv::Rect textRect(textOrigin, textSize);

        // 绘制矩形背景
        cv::rectangle(orig_img, textRect, color, cv::FILLED); // 填充矩形背景

        // 绘制文本，调整位置使其居中
        putText(orig_img, text, textOrigin + cv::Point(0, textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255)); // 白色文本

        int center_y = track.box.y + track.box.height / 2;
        // 越线检测
        if (previous_y_positions.size() > track.id) {
            if (previous_y_positions[track.id] <= 300 && center_y > 300) {
                // if (crossed_up.find(track.id) == crossed_up.end()) {
                    count_up++;
                //     crossed_up.insert(track.id); // 记录已越线 ID
                // }
            }
            // Down crossing
            else if (previous_y_positions[track.id] >= 300 && center_y < 300) {
                // if (crossed_down.find(track.id) == crossed_down.end()) {
                    count_down++;
                //     crossed_down.insert(track.id); // 记录已越线 ID
                // }
            }
        }
        previous_y_positions[track.id] = center_y;
        // if (isPointInsideRectangle(o_rectangle, cv::Point(track.box.x + track.box.width / 2, track.box.y + track.box.height / 2))) {
        //     per_num++;
        // }
    }
    // cv::rectangle(ori_img, track.box,  cv::Scalar(0, 255, 0), 2, 8, 0);                                                                                       
    // putText(ori_img, std::to_string(track.id), cv::Point(track.box.x, track.box.y+ 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255, 0));
    // 显示上行和下行人数
    // putText(orig_img, "Up through: " + std::to_string(count_up), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    // putText(orig_img, "Down through: " + std::to_string(count_down), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    // putText(orig_img, "Number of regions: " + std::to_string(per_num), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
     
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

    return orig_img;
}

rkYolov5s::~rkYolov5s()
{
    deinitPostProcess();

    ret = rknn_destroy(ctx);

    if (model_data)
        free(model_data);

    if (input_attrs)
        free(input_attrs);
    if (output_attrs)
        free(output_attrs);
}
