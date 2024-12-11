#ifndef _my_lane_lib
#define _my_lane_lib
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include<ctime>
#include<vector>
#include <numeric>

using namespace cv;
using namespace std;


vector<int> lane_func_01(int h, int sssss_t, int eeeee_t){
    int n = eeeee_t - sssss_t + 1;
    float slope = (float)n / (float)h;
    int i = h;
   
    vector<int> index;
    index.push_back(i);
    while (i > 0){
        float down = slope * (float)i;
        i = (int)(i - down);
        index.push_back((int)i);
    };
    return index;
}

int max_position(int8_t*& output, int j, int k){
    int8_t max_value = *(output+j+0);
    int posit = 0;
    for(int c=1;c<k;c++){
        if(*(output+j+c)>max_value){
            posit = c;
        }
    }
    return posit;
}


#endif

