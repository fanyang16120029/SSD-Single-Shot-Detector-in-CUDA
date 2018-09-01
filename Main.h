/*************************************************
Copyright:IREAL & BJTU
Author: Fan Yang
Date:2018-08-27
Description: Deep learning lib for CUDA7.0.
**************************************************/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DNNLIB.h"

#include <stdio.h>
#include <time.h>

extern "C"
void DNN_ConvLayer(DNN_MAT *in, DNN_MAT *out, DNN_KERNEL *kernel, int padding, int stride, int dilation, bool with_RELU);

extern "C"
void DNN_MaxPoolLayer(DNN_MAT *in, DNN_MAT *out, int win_size, int pad, int stride);

extern "C"
void DNN_NormLayer(DNN_MAT *in);

extern "C"
void DNN_SoftmaxLayer(DNN_MAT *in, int class_num);

extern "C"
void MATCPU2GPU(DNN_MAT *mat);

extern "C"
void MATGPU2CPU(DNN_MAT *mat);

extern "C"
void MATGPUMalloc(DNN_MAT *mat);

extern "C"
void KERNELCPU2GPU(DNN_KERNEL *ker);

extern "C"
void KERNELGPUMalloc(DNN_KERNEL *ker);