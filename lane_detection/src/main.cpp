#include <stdint.h>
#include <unistd.h>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h> 

#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h> 

#include "dputils.h"
#include "l_0001.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

#define w512h288
#define ld_mx "lanenet_cd"
#define ld_mx_input "conv1"
#define out_ins "conv19"

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

    cv::Size ld_size(512, 288);
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
                cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        		resize(img, img, ld_size, 0, 0, INTER_LINEAR);
                cv::Mat rec_img (img, cv::Rect(0, 32, 512, 256));
                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage++, rec_img));
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
            if (frame.empty())
            {
                cerr << "Error: Frame is empty, skipping." << endl;
                continue;
            }
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

void RS(DPUTask* tsk)
{
    struct timeval stamp;
  	DPUTensor *out_tensor = dpuGetOutputTensor(tsk, out_ins);
  	int outH = dpuGetTensorHeight(out_tensor);
  	int outW = dpuGetTensorWidth(out_tensor);
  	int channel = dpuGetOutputTensorChannel(tsk, out_ins);
  	int sizeOut = dpuGetOutputTensorSize(tsk, out_ins);
  	float scale = dpuGetOutputTensorScale(tsk, out_ins);
  	int8_t *outTensorAddr = dpuGetTensorAddress(out_tensor);

    int frame_index = 0;
    
    while (processingFrames)
    {
        pair<int, Mat> pairIndexImage1;
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
            pairIndexImage1 = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        gettimeofday(&stamp, NULL);
  		double task_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        float mean[3] = {104.8, 111.6, 112.4};
	    float mscale = 0.01562f;
        Mat img = pairIndexImage1.second;

        dpuSetInputImageWithScale(tsk, (char *)ld_mx_input, img, mean, mscale);
        dpuRunTask(tsk);
        gettimeofday(&stamp, NULL);
    	double task_over_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        double preprocess_time = task_over_timestamp - task_timestamp;
        printf("RS preprocess_time: %f s\n", preprocess_time);
        cv::Mat segMat_r = cv::Mat::zeros(outH, outW, CV_8UC1);
    	vector<int> mindex = lane_func_01(255, 1, 1);
        for (int kk = 0; kk < mindex.size(); kk++){
            int row = mindex[kk];
            for (int col = 0; col < outW; col++){
                int i = row * outW * 2 + col * 2;
                int posit = max_position(outTensorAddr, i, 2);
                if (posit == 1){
                img.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                }		
            }        
		}
      
        frame_index++;
     
        gettimeofday(&stamp, NULL);
    	double task_hou_over_timestamp = stamp.tv_sec * 1.0 + ((stamp.tv_usec * 1.0) / 1000000);
        double postprocess_time = task_hou_over_timestamp-task_over_timestamp;
        printf("RS postprocess_time: %f s\n", postprocess_time);
		pairIndexImage1.second = img;
        mtxQueueShow.lock();
        queueShow.push(pairIndexImage1);
        mtxQueueShow.unlock();
    }
}

int main(const int argc, const char** argv)
{
    
    DPUKernel *kernel_conv;
    DPUTask *task_conv_1, *task_conv_2;
    dpuOpen();
    kernel_conv = dpuLoadKernel(ld_mx);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
    task_conv_2 = dpuCreateTask(kernel_conv, 0);

    array<thread, 4> threadsList = 
    {
	    thread(readFrame, argv[1]),
	    thread(displayFrame),
	    thread(RS, task_conv_1),
    };

    for (int i = 0; i < 4; i++)
    {
        threadsList[i].join();
    }

    dpuDestroyTask(task_conv_1);
    dpuDestroyTask(task_conv_2);
    dpuDestroyKernel(kernel_conv);
    dpuClose();

    return 0;
}

