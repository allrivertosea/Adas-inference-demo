

#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>

#include <dnndk/dnndk.h>
#include "dputils.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace std;
using namespace cv;
using namespace std::chrono;



#define NMS_THRESHOLD 0.5f
#define INPUT_NODE "conv1"

#define tcls_mx "tsr_cls"
#define tcls_input "conv1"
#define out_tcls "fc1"


int idxInputImage = 0;
int idxShowImage = 0;
bool bReading = true;
chrono::system_clock::time_point start_time;

typedef pair<int, Mat> imagePair;
class paircomp
{
    public:
    bool operator()(const imagePair &n1, const imagePair &n2) const
    {
        if (n1.first == n2.first)
        {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

mutex mtxQueueInput;
mutex mtxQueueShow;
queue<pair<int, Mat>> queueInput;
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;



void readFrame(const char *fileName)
{
    VideoCapture video;
    cv::Size od_size(512, 288);
    string videoFile = fileName;
	start_time = chrono::system_clock::now();

    while (true)
    {
        if (!video.open(videoFile))
        {
            exit(-1);
        }

        while (true)
        {
            usleep(20000);
            Mat img;
            if (queueInput.size() < 30)
            {
                if (!video.read(img))
                {
                    cout << "Failed to read frame from video device." << endl;
                    break;
                }
                resize(img, img, od_size, 0, 0, INTER_LINEAR);
                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage++, img));
                mtxQueueInput.unlock();
            }
            else
            {
                usleep(10);
            }
        }
        video.release();
    }
    bReading = false;
}


using namespace cv;
VideoWriter videoWriter;

void displayFrame()
{
    Mat frame;
    while (true)
    {
        mtxQueueShow.lock();
        if (queueShow.empty())
        {
            mtxQueueShow.unlock();
            usleep(1000);
        }
        else if (idxShowImage == queueShow.top().first)
        {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;

            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS"; 
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240},1);
            if (!videoWriter.isOpened())
            {
                Size frameSize = frame.size();
                videoWriter.open("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, frameSize);
                
                if (!videoWriter.isOpened())
                {
                    cerr << "Error: Couldn't open video writer." << endl;
                }
            }
            videoWriter.write(frame);

            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();

        }
        else
        {
            mtxQueueShow.unlock();
        }
    }
}


std::string run_Tsr_cls(DPUTask* tck, cv::Mat cropimg) {

    cv::Size tsr_size(35, 35);
    vector<float> mean{ 0.0f, 0.0f, 0.0f };
    float scale = 1. / 255.;
  
    Mat image = cropimg;

    int8_t* outAddr = (int8_t*)dpuGetOutputTensorAddress(tck, out_tcls);
    int size = dpuGetOutputTensorSize(tck, out_tcls);
    int channel = dpuGetOutputTensorChannel(tck, out_tcls);
    float out_scale = dpuGetOutputTensorScale(tck, out_tcls);
    float* softmax = new float[size];

    resize(image, image, tsr_size, 0, 0, INTER_LINEAR);
    dpuSetInputImageWithScale(tck, (char*)tcls_input, image, mean.data(), scale);
    dpuRunTask(tck);
    dpuRunSoftmax(outAddr, softmax, channel, size / channel, out_scale);
    std::vector<std::string> topKResults = TopK(softmax, channel, 5);
    std::string result;
    if (!topKResults.empty()) {
        result = topKResults[0];
        std::cout << "First top-K result: " << result << std::endl;
    }
    else {
        result = "";
    }
    delete[] softmax;
    return result;
}


void runYOLO(DPUTask* task, DPUTask* tsck)
{
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float scale = 1. / 255.;

    int sHeight = dpuGetInputTensorHeight(task, INPUT_NODE);
    int sWidth = dpuGetInputTensorWidth(task, INPUT_NODE);
    const string classes_result[13] = {"5","10","15","20","25","30","40","50","60","70","80","90","100"};
    const string outputs_node[3] = {"conv58", "conv59", "conv60"};

    int stride[3] = {8, 16, 32};
    std::vector<std::string> tsrname;
    std::vector<std::string> tsrname_results;
    auto startTime = steady_clock::now();
    bool timerStarted = false;
    while (true)
    {
            pair<int, Mat> pairIndexImage;
            mtxQueueInput.lock();
            if (queueInput.empty())
            {
                mtxQueueInput.unlock();
                if (bReading)
                {
                    continue;
                }
                else
                {
                    break;
                }
            }
            else
            {
                pairIndexImage = queueInput.front();
                queueInput.pop();
                mtxQueueInput.unlock();
            }
            
            Mat img = pairIndexImage.second;
            cv::Mat frame = img.clone();
            dpuSetInputImageWithScale(task, (char *)INPUT_NODE, img, mean, scale);
            dpuRunTask(task);
            vector<vector<float>> boxes;
        
    	    for(int i = 0; i < 3; i++)
    	    {
    	        string output_node = outputs_node[i];
    	        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());	
    	        int width = dpuGetOutputTensorWidth(task, output_node.c_str());		
    	        int height = dpuGetOutputTensorHeight(task, output_node.c_str());	
    	        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());	
    	        int8_t* dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
    	        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
    	        vector<float> result(sizeOut);
    	        boxes.reserve(sizeOut);
    	        get_output(dpuOut, sizeOut, scale, channel, height, width, result);
    	        detect(boxes, result, channel, height, width, i, sHeight, sWidth, stride[i]);
    
    	    }
    	    correct_region_boxes(boxes, boxes.size(), img.cols, img.rows, sWidth, sHeight);
    	    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);
    	    float h = img.rows;
    	    float w = img.cols;
         
    	    for(size_t i = 0; i < res.size(); ++i)
    	    {
      	        float xmin = res[i][0] - res[i][2]/2.0;
      	        float ymin = res[i][1] - res[i][3]/2.0;
                float xmax = res[i][0] + res[i][2] / 2.0;
      			float ymax = res[i][1] + res[i][3] / 2.0;
                float w_ = res[i][2];
      	        float h_ = res[i][3];
                float xmin_ori = xmin * 2.5;
                float ymin_ori = ymin * 2.5;
                float width_ori = (xmax - xmin) * 2.5;
                float height_ori = (ymax - ymin) * 2.5;
                cv::Rect_<float> rec_obj = Rect_<float>(xmin_ori, ymin_ori, width_ori, height_ori);
                cv::Mat raw_img;
                cv::Size od_size(1280, 720);
                resize(frame, raw_img, od_size, 0, 0, INTER_LINEAR);
                cv::Mat cropped_img;
                std::string classname;
                
                if (width_ori > 25)
                {
                	try {
                		    cropped_img = CroppedImage(raw_img, rec_obj);
                            if (!cropped_img.empty()) {
                                classname = run_Tsr_cls(tsck, cropped_img);
                                 
                            }
                		}
                	catch (const std::exception& e) {
                			continue;
                		}
                    tsrname.push_back(classname);   
                    std::string tsrend;
                    if (tsrname.size() >= 2) {
                        size_t lastIndex = tsrname.size() - 1;
                        if (tsrname[lastIndex] == tsrname[lastIndex - 1]) {
                            tsrend = tsrname[lastIndex];
                            if (std::find(std::begin(classes_result), std::end(classes_result), tsrend) != std::end(classes_result)) {
                                std::cout << "LIMIT SPEED: " << tsrend << std::endl;
                                
                                rectangle(img, cvPoint(xmin, ymin), cvPoint(xmin + w_, ymin + h_), Scalar(0, 0, 255), 1, 1, 0);
                                int baseline = 0;
                                Point labelPos(xmin, ymin - 10); 
                                if (labelPos.y < 0) {
                                    labelPos.y = 0;
                                    }
                                putText(img, tsrend, labelPos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
                            } else {
                                tsrname.clear();
                            }
                            
                            
                        }
                        }
                   }
                
      	    }
            
            pairIndexImage.second = img; 
            mtxQueueShow.lock();
            queueShow.push(pairIndexImage);
            mtxQueueShow.unlock();

    }
    
}

int main(const int argc, const char** argv)
{
 
    DPUKernel *kernel_tsr;
    DPUTask *task_tsr;
    dpuOpen();
    kernel_tsr = dpuLoadKernel("yolov5n");
    task_tsr = dpuCreateTask(kernel_tsr, 0);
    DPUKernel* kernel_cls = dpuLoadKernel(tcls_mx);
    DPUTask* task_cls = dpuCreateTask(kernel_cls, 0);

    array<thread, 3> threadsList = 
    {
	    thread(readFrame, argv[1]),
        thread(displayFrame),
	    thread(runYOLO, task_tsr,task_cls),
    };

    for (int i = 0; i < 3; i++)
    {
        threadsList[i].join();
    }

    dpuDestroyKernel(kernel_tsr);
    dpuClose();

    return 0;
}

