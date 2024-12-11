# Xilinx-adas-inference-demo
ADAS related models (vehicle and person detection, lane detection, speed limit sign detection) inference demo, the post-processing module is not added.

## 推理结果

展示系统工程中车人检测、车道线检测、限速牌检测的功能测试视频

![功能测试](https://github.com/allrivertosea/Xilinx-adas-inference-demo/blob/main/test.gif)

## 项目简介

- 基于**DPU**加速推理adas相关模型**
- 支持**Zynq UltraScale+ MPSoC系列开发板**
- 支持**XAZU3EG型号芯片**

## 环境说明

- Vitis ai 1.2.82
- Opencv 3.4.13
- cmake-3.19.3-Linux-aarch64

## 使用说明

```
make
./xxx.elf xxx.mp4
```


