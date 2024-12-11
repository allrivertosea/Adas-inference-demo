
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>


using namespace std;
using namespace std::chrono;

#define CONF 0.5
const int classificationCnt = 13;
const int anchorCnt = 3;

std::vector<std::string> vkinds={"5", "10", "15", "20","25", "30", "40", "50", "60", "70", "80", "90","100"};

std::vector<std::string> TopK(const float *d, int size, int k) {
    assert(d && size > 0 && k > 0);
    std::priority_queue<std::pair<float, int>> q;

    for (int i = 0; i < size; ++i) {
        q.push(std::pair<float, int>(d[i], i));
    }

    std::vector<std::string> topKResults;
    for (int i = 0; i < k; ++i) {
        std::pair<float, int> ki = q.top();
        if (ki.first > 0.85) {
            topKResults.push_back(vkinds[ki.second]);
        }
        q.pop();
    }

    return topKResults;
}

inline float sigmoid(float p)
{
    return 1.0 / (1 + exp(-p * 1.0));
}

inline float overlap(float x1, float w1, float x2, float w2)
{
    float left = max(x1 - w1 / 2.0, x2 - w2 / 2.0);
    float right = min(x1 + w1 / 2.0, x2 + w2 / 2.0);
    return right - left;
}

inline float cal_iou(vector<float> box, vector<float>truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w < 0 || h < 0) return 0;

    float inter_area = w * h;
    float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
    return inter_area * 1.0 / union_area;
}

inline float cal_diou(vector<float> box, vector<float>truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w <= 0 || h <= 0) return 0;
    float c = w * w + h * h;
    float inter_area = w * h;
    float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
    float iou = inter_area * 1.0 / union_area;
    float d = (box[0] - truth[0]) * (box[0] - truth[0]) + (box[1] - truth[1]) * (box[1] - truth[1]);
    float diou_term = pow(d / c, 0.6);
    return iou - diou_term;
}

void correct_region_boxes(vector<vector<float>>& boxes, int n, int w, int h, int netw, int neth, int relative = 0)
{
    int new_w=0;
    int new_h=0;

    if (((float)netw/w) < ((float)neth/h))
    {
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (int i = 0; i < n; ++i)
    {
        boxes[i][0] =  (boxes[i][0] - (netw - new_w)/2./netw) / ((float)new_w/(float)netw);
        boxes[i][1] =  (boxes[i][1] - (neth - new_h)/2./neth) / ((float)new_h/(float)neth);
        boxes[i][2] *= (float)netw/new_w;
        boxes[i][3] *= (float)neth/new_h;
    }
}


void detect(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int weight, int num, int sh, int sw, int stride_);

void detect(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int width, int num, int sHeight, int sWidth, int stride_)
{
    
    vector<float> biases{5,5, 6,6, 7,8, 8,8, 10,10, 12,12, 14,15, 19,20, 29,32};
    int conf_box = 5 + classificationCnt;
    float swap[height * width][anchorCnt][conf_box];

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < channel; ++c)
            {
                int temp = c * height * width + h * width + w;
                swap[h * width + w][c / conf_box][c % conf_box] = result[temp];
            }
        }
    }
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < anchorCnt; ++c)
            {
                float obj_score = sigmoid(swap[h * width + w][c][4]);
                if (obj_score < CONF)
                    continue;
                vector<float> box;

                box.push_back( (sigmoid(swap[h * width + w][c][0]) * 2 - 0.5 + w) * stride_);
                box.push_back( (sigmoid(swap[h * width + w][c][1]) * 2 - 0.5 + h) * stride_);
                
                box.push_back( pow( (sigmoid(swap[h * width + w][c][2]) * 2), 2 ) * biases[2 * c + 6 * num] );
                box.push_back( pow( (sigmoid(swap[h * width + w][c][3]) * 2), 2 ) * biases[2 * c + 6 * num + 1] );
                box.push_back(-1);
                box.push_back(obj_score);

                for (int p = 0; p < classificationCnt; p++) {
                    box.push_back(obj_score * sigmoid(swap[h * width + w][c][5 + p]));
                }
                boxes.push_back(box);
            }
        }
    }
}


vector<vector<float>> applyNMS(vector<vector<float>>& boxes,int classes, const float thres) {
    vector<pair<int, float>> order(boxes.size());
    vector<vector<float>> result;

    for(int k = 0; k < classes; k++) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            order[i].first = i;
            boxes[i][4] = k;
            order[i].second = boxes[i][6 + k];
        }
        sort(order.begin(), order.end(),
             [](const pair<int, float> &ls, const pair<int, float> &rs) { return ls.second > rs.second; });

        vector<bool> exist_box(boxes.size(), true);

        for (size_t _i = 0; _i < boxes.size(); ++_i) {
            size_t i = order[_i].first;
            if (!exist_box[i]) continue;
            if (boxes[i][6 + k] < CONF) {
                exist_box[i] = false;
                continue;
            }
            /* add a box as result */
            result.push_back(boxes[i]);

            for (size_t _j = _i + 1; _j < boxes.size(); ++_j) {
                size_t j = order[_j].first;
                if (!exist_box[j]) continue;
                float ovr = cal_iou(boxes[j], boxes[i]);
                if (ovr >= thres) exist_box[j] = false;
            }
        }
    }

    return result;
}
				
void get_output(int8_t* dpuOut, int sizeOut, float scale, int oc, int oh, int ow, vector<float>& result)
{
    vector<int8_t> nums(sizeOut);
    memcpy(nums.data(), dpuOut, sizeOut);
    for(int a = 0; a < oc; ++a)	
    {
        for(int b = 0; b < oh; ++b)	
        {
            for(int c = 0; c < ow; ++c)	
            {
                int offset = b * oc * ow + c * oc + a;
                result[a * oh * ow + b * ow + c] = nums[offset] * scale;
            }
        }
    }
}
