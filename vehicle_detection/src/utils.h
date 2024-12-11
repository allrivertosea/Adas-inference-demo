/*
  The following source code derives from Darknet
*/

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
const int classificationCnt = 8;
const int anchorCnt = 3;

typedef struct
{
    int w;
    int h;
    int c;
    float *data;
} image;

image load_image_cv(const cv::Mat& img);
image letterbox_image(image im, int w, int h);
void free_image(image m);

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
std::vector<double> linspace(double start, double end, int numPoints) {
    std::vector<double> result;
    result.reserve(numPoints);

    double step = (end - start) / (numPoints - 1);
    for (int i = 0; i < numPoints; ++i) {
        result.push_back(start + i * step);
    }

    return result;
}

void detect(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int weight, int num, int sh, int sw, int stride_);
//void detect(vector<vector<float>> &boxes, int8_t* result, int channel, int height, int weight, int num, int sh, int sw, int stride_);

//void detect(vector<vector<float>> &boxes, int8_t* result, int channel, int height, int width, int num, int sHeight, int sWidth, int stride_)

void detect(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int width, int num, int sHeight, int sWidth, int stride_)
{
    //vector<float> biases{6,6, 6,14, 12,10, 10,24, 20,16, 18,41, 36,26, 58,50, 119,116};
    vector<float> biases{6,6, 5,12, 11,9, 8,21, 18,14, 16,36, 30,22, 50,41, 100,92};
                       //6,6, 5,12, 11,9, 8,21, 18,14, 16,36, 30,22, 50,41, 100,92
    
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

          
            for (int c = 0; c < anchorCnt; ++c)//anchorCnt=3，num为第几个输出节点
            {
                float obj_score = sigmoid(swap[h * width + w][c][4]);
                if (obj_score < CONF)
                    continue;
                vector<float> box;

                box.push_back( (sigmoid(swap[h * width + w][c][0]) * 2 - 0.5 + w)*stride_);
                box.push_back( (sigmoid(swap[h * width + w][c][1]) * 2 - 0.5 + h)*stride_);
                box.push_back( pow( (sigmoid(swap[h * width + w][c][2]) * 2), 2 ) *biases[2 * c + 6 * num]);
                box.push_back( pow( (sigmoid(swap[h * width + w][c][3]) * 2), 2 ) *biases[2 * c + 6 * num + 1]);
                box.push_back(-1);
                box.push_back(obj_score);
                
                for (int p = 0; p < classificationCnt; p++) {
                    box.push_back(obj_score * sigmoid(swap[h * width + w][c][5 + p]));
                }

                //box.push_back(obj_score * sigmoid(swap[h * width + w][c][5]));
                
                boxes.push_back(box);
            }
        }
    }
}


// height: 36, width 64
// height: 18, width 32
// height: 9, width 16
void detectn(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int width, int num, int sHeight, int sWidth, int8_t* dpuOut, float scale, int sizeOut, int stride_);

void detectn(vector<vector<float>> &boxes, vector<float> result, int channel, int height, int width, int num, int sHeight, int sWidth, int8_t* dpuOut, float scale, int sizeOut, int stride_)
{
	vector<float> biases{6,6, 5,12, 11,9, 8,21, 18,14, 16,36, 30,22, 50,41, 100,92};
    int conf_box = 5 + classificationCnt;
    int idx;
    		  // w*h            3           5+9
    float swap[height * width][anchorCnt][conf_box];
    vector<int8_t> nums(sizeOut);
    memcpy(nums.data(), dpuOut, sizeOut);
    
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int c = 0; c < channel; ++c)
            {
                int temp = c * height * width + h * width + w;
                swap[h * width + w][c / conf_box][c % conf_box] = nums[temp] * scale;
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
                box.push_back( (sigmoid(swap[h * width + w][c][1]) * 2 - 0.5 + w) * stride_);
                box.push_back( pow( (sigmoid(swap[h * width + w][c][2]) * 2), 2 ) * biases[2 * c + 6 * num] );
                box.push_back( pow( (sigmoid(swap[h * width + w][c][3]) * 2), 2 ) * biases[2 * c + 6 * num + 1] );
                box.push_back(-1);
                box.push_back(obj_score);
				
                box.push_back(obj_score * sigmoid(swap[h * width + w][c][5]));
                
                //box.push_back((w + sigmoid(swap[h * width + w][c][0])) / width);
                //box.push_back((h + sigmoid(swap[h * width + w][c][1])) / height);
				//box.push_back((w + swap[h * width + w][c][0]) / width);
				//box.push_back((h + swap[h * width + w][c][1]) / height);
                /*box.push_back(exp(swap[h * width + w][c][2]) * biases[2 * c + 2 * anchorCnt * num] / float(sWidth));
                box.push_back(exp(swap[h * width + w][c][3]) * biases[2 * c + 2 * anchorCnt * num + 1] / float(sHeight));
                box.push_back(-1);
                box.push_back(obj_score);
                for (int p = 0; p < classificationCnt; p++)
                {
                    box.push_back(obj_score * sigmoid(swap[h * width + w][c][5 + p]));
                }*/
                boxes.push_back(box);
            }
        }
    }
}

/*分类阈值设置的NMS
vector<vector<float>> applyNMS(vector<vector<float>>& boxes, int classes, const float thres)
{
    vector<pair<int, float>> order(boxes.size());
    vector<vector<float>> result;
    //vector<float> cls_thres = {0.75, 0.4, 0.75,  0.7, 0.7, 0.4, 0.4,  0.4 };
    vector<float> cls_thres = {0.5, 0.5, 0.5,  0.5, 0.5, 0.5, 0.5,  0.5 };
	//{"car", "truck", "person", "bicycle", "cyclist", "van", "tricycle", "bus"};
    

    for(int k = 0; k < classes; k++)
    {
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            order[i].first = i;
            boxes[i][4] = k;
            order[i].second = boxes[i][6 + k];
        }
        sort(order.begin(), order.end(),
             [](const pair<int, float> &ls, const pair<int, float> &rs) { return ls.second > rs.second; });

        vector<bool> exist_box(boxes.size(), true);

        for (size_t _i = 0; _i < boxes.size(); ++_i)
        {
            size_t i = order[_i].first;
            if (!exist_box[i]) continue;
            if (boxes[i][6 + k] < cls_thres[boxes[i][4]])
            {
                exist_box[i] = false;
                continue;
            }
            result.push_back(boxes[i]);

            for (size_t _j = _i + 1; _j < boxes.size(); ++_j)
            {
                size_t j = order[_j].first;
                if (!exist_box[j]) continue;
                float ovr = cal_diou(boxes[j], boxes[i]);
                if (ovr >= thres) exist_box[j] = false;
            }
        }
    }

    return result;
}*/

//未进行分类阈值设置的NMS
vector<vector<float>> applyNMS(vector<vector<float>>& boxes, int classes, const float thres) {
    vector<pair<int, float>> order(boxes.size());
    vector<vector<float>> result;
    for (int k = 0; k < classes; ++k) {
        

        for (size_t i = 0; i < boxes.size(); ++i) {
            order[i].first = i;
            boxes[i][4] = k;
            order[i].second = boxes[i][6 + k];
        }

        sort(order.begin(), order.end(),
             [](const pair<int, float> &ls, const pair<int, float> &rs) { return ls.second > rs.second; });

        vector<bool> exist_box(boxes.size(), true);

        //first-CONF
        for (size_t _i = 0; _i < boxes.size(); ++_i) {
            size_t i = order[_i].first;
            if (!exist_box[i]) continue;
            if (boxes[i][6 + k] < CONF) {
                exist_box[i] = false;
                continue;
            }
            
            result.push_back(boxes[i]);
            
            //second-IOU，筛选出一定区域内，属于同一种类得分最大的框
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



vector<vector<float>> applyoldNMS(vector<vector<float>>& boxes,int classes, const float thres) {
    
    vector<vector<float>> result;

    for(int k = 0; k < classes; k++) {
        vector<pair<int, float>> order(boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            order[i].first = i;
            boxes[i][4] = k;
            order[i].second = boxes[i][6 + k];
        }
        sort(order.begin(), order.end(),
             [](const pair<int, float> &ls, const pair<int, float> &rs) { return ls.second > rs.second; });

        vector<bool> exist_box(boxes.size(), true);
        
        //first-CONF
        for (size_t _i = 0; _i < boxes.size(); ++_i) {
            size_t i = order[_i].first;
            if (!exist_box[i]) continue;
            if (boxes[i][6 + k] < CONF) {
                exist_box[i] = false;
                continue;
            }
          
            for (size_t _j = _i + 1; _j < boxes.size(); ++_j) {
                size_t j = order[_j].first;
                if (!exist_box[j]) continue;
                float ovr = cal_iou(boxes[j], boxes[i]);
                if (ovr >= thres) exist_box[j] = false;
            }
            /* add a box as result */
            result.push_back(boxes[i]);
        }
    }
   

    

    return result;
}


vector<vector<float>> applyNMSall(vector<vector<float>>& boxes, int classes, const float thres)
{
    vector<pair<int, float>> order1(classes);
    vector<pair<int, float>> order2(boxes.size());
    vector<vector<float>> result;

    for (size_t i = 0; i < boxes.size(); ++i)
    {
    	for(int k = 0; k < classes; k++)
    	{
			order1[k].first = k;
			order1[k].second = boxes[i][6 + k];
    	}
		sort(order1.begin(), order1.end(),
			 [](const pair<int, float> &ls, const pair<int, float> &rs) { return ls.second > rs.second; });
		boxes[i][4] = order1[0].first;
		boxes[i][6] = order1[0].second;
    }
	sort(boxes.begin(), boxes.end(),
		 [](vector<float> &ls, vector<float> &rs) { return ls[6] > rs[6]; });

	vector<bool> exist_box(boxes.size(), true);

	for (size_t i = 0; i < boxes.size(); ++i)
	{
		//size_t i = order[_i].first;
		if (!exist_box[i]) continue;
		if (boxes[i][6] < CONF)
		{
			//exist_box[i] = false;
			continue;
		}
		/* add a box as result */
		result.push_back(boxes[i]);

		for (size_t j = i + 1; j < boxes.size(); ++j)
		{
			//size_t j = order[_j].first;
			if (!exist_box[j]) continue;
			float ovr = cal_diou(boxes[j], boxes[i]);
			//if (ovr >= thres) exist_box[j] = false;
		}
	}

    return result;
}

								
void get_output(int8_t* dpuOut, int sizeOut, float scale, int oc, int oh, int ow, vector<float>& result)
{
    vector<int8_t> nums(sizeOut);
    
    memcpy(nums.data(), dpuOut, sizeOut);
    
    for(int a = 0; a < oc; ++a)	// 18
    {
        for(int b = 0; b < oh; ++b)	// 36
        {
            for(int c = 0; c < ow; ++c)	// 64
            {
                int offset = b * oc * ow + c * oc + a;
                result[a * oh * ow + b * ow + c] = nums[offset] * scale;
            }
        }
    }
}

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

static void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void free_image(image m)
{
    if(m.data)
    {
        free(m.data);
    }
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*) calloc(h*w*c, sizeof(float));
    return out;
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i)
	{
    	m.data[i] = s;
    }
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k)
    {
        for(y = 0; y < source.h; ++y)
        {
            for(x = 0; x < source.w; ++x)
            {
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i)
    {
        for(k= 0; k < c; ++k)
        {
            for(j = 0; j < w; ++j)
            {
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/256.;
            }
        }
    }
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k)
    {
        for(r = 0; r < im.h; ++r)
        {
            for(c = 0; c < w; ++c)
            {
                float val = 0;
                if(c == w-1 || im.w == 1)
                {
                    val = get_pixel(im, im.w-1, r, k);
                }
                else
                {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k)
    {
        for(r = 0; r < h; ++r)
        {
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c)
            {
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1)
            	continue;
            
            for(c = 0; c < w; ++c)
            {
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}


image load_image_cv(const cv::Mat& img)
{
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    image im = make_image(w, h, c);

    unsigned char *data = img.data;

    for(int i = 0; i < h; ++i)
    {
        for(int k= 0; k < c; ++k)
        {
            for(int j = 0; j < w; ++j)
            {
                im.data[k*w*h + i*w + j] = data[i*w*c + j*c + k]/256.;
            }
        }
    }

    for(int i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }

    return im;
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h))
    {
        new_w = w;
        new_h = (im.h * w)/im.w;
    }
    else
    {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);

    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2);
    free_image(resized);

    return boxed;
}
