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
#include <sys/timeb.h>
#include <sys/time.h>

#include <dnndk/dnndk.h>
#include "dputils.h"
#include "utils.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono;


#define NMS_THRESHOLD 0.4f
#define INPUT_NODE "conv1"

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
    string videoFile = fileName;
    start_time = chrono::system_clock::now();
    cv::Size od_size(512, 288);
    while (true)
    {
        if (!video.open(videoFile))
        {
            cout<<"Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }
        while (true) {
            usleep(20000);
            Mat img;
            if (queueInput.size() < 30)
            {
                if (!video.read(img) )
                {
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
            cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{240, 240, 240}, 1);
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





std::atomic<bool> processingFrames(true);

void runYOLO(DPUTask* task)
{
    struct timeval stamp;
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float scale = 1. / 255.;

    int sHeight = dpuGetInputTensorHeight(task, INPUT_NODE);
    int sWidth = dpuGetInputTensorWidth(task, INPUT_NODE);

    const string classes[8] = {"car", "truck", "person", "bicycle", "cyclist", "van", "tricycle", "bus"};
    const string outputs_node[3] = {"conv58", "conv59", "conv60"};

    int stride[3] = {8, 16, 32};

    while (processingFrames)
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
                processingFrames = false;
                break;
            }
        }
        else
        {
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        gettimeofday(&stamp, NULL);
		double task_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        Mat img = pairIndexImage.second;
        dpuSetInputImageWithScale(task, (char *)INPUT_NODE, img, mean, scale);
        dpuRunTask(task);
        gettimeofday(&stamp, NULL);
    	double task_over_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        double preprocess_time = task_over_timestamp - task_timestamp;
        printf("OD preprocess_time: %f s\n", preprocess_time);

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
	        float w_ = res[i][2];
	        float h_ = res[i][3];
	       
	        if( res[i][res[i][4] + 6] > CONF )
	        {
              int type = res[i][4];
	          string classname = classes[type];
              rectangle(img, cvPoint(xmin, ymin), cvPoint(xmin + w_, ymin + h_), Scalar(0, 0, 255), 1, 1, 0);              
	            
	        }
	    }
      
        gettimeofday(&stamp, NULL);
        double task_hou_over_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        double postprocess_time = task_hou_over_timestamp-task_over_timestamp;
        printf("OD postprocess_time: %f s\n", postprocess_time);

	    pairIndexImage.second = img;
        
        mtxQueueShow.lock();
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
    }
}


int main(const int argc, const char** argv)
{

    dpuOpen();
    DPUKernel *kernel = dpuLoadKernel("yolov5n_od");
    vector<DPUTask *> task(4);
    generate(task.begin(), task.end(),
    std::bind(dpuCreateTask, kernel, 0));
    array<thread, 3> threadsList = 
    {
	    thread(readFrame, argv[1]),
	    thread(displayFrame),
	    thread(runYOLO, task[0]),
    };

    for (int i = 0; i < 6; i++)
    {
        threadsList[i].join();
    }

    for_each(task.begin(), task.end(), dpuDestroyTask);
    dpuDestroyKernel(kernel);
    dpuClose();

    return 0;
}

