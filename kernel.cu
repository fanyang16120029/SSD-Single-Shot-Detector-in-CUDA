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
void DNN_ConvLayer(DNN_MAT *in, DNN_MAT *out, DNN_KERNEL *kernel, int pad, int stride, int dilation, bool with_RELU);

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

/*************************************************
Function:       Conv_Accel
Description:    Convolution layer accelerater.
Calls:          NULL
Input:
				in			---->	Input feature map data pointer.
				out			---->	Output feature map data pointer.
				w			---->	Kernel weight pointer.
				b			---->	Kernel bias pointer.
				in_h		---->	Input height.
				out_h		---->	Output height.
				in_c		---->	Input channel.
				out_c		---->	Output channel.
				ker_h		---->	Kernel height.
				stride		---->	Stride.
				dilation	---->	Dilation.
				with_RELU	---->	RELU or not?
Output:         out			---->	Output feature map data pointer.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

__global__ void Conv_Accel(const DNN_dtype * __restrict__ in, DNN_dtype * __restrict__ out, const DNN_dtype * __restrict__ w, const DNN_dtype *b, 
						   int in_h, int out_h, int in_c, int out_c, int ker_h, int stride, int dilation, bool with_RELU)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int kid = threadIdx.z + blockIdx.z * blockDim.z;

	DNN_dtype res = 0;

	int win_down = (ker_h & 1) ? (- ker_h * 0.5) : (- ker_h * 0.5 + 1);
	int win_up = ker_h * 0.5 + 1;

	int in_step = in_h * in_c;
	int w_step1 = ker_h * in_c;

	int x_index = 0;
	int y_index = 0;
	int m_step = 0;
	int n_step = 0;
	int kid_step = kid * w_step1 * ker_h;

	if(i < out_h && j < out_h && kid < out_c)
	{
		res = b[kid];
		for(int m = win_down; m < win_up; m++)
		{
			y_index = i * stride + m * dilation;
			if(y_index >= 0 && y_index < in_h)
			{
				m_step = (m - win_down) * w_step1;
				for(int n = win_down; n < win_up; n++)
				{
					x_index = j * stride + n * dilation;
					if(x_index >= 0 && x_index < in_h)
					{
						n_step = (n - win_down) * in_c;
						for(int k_iter = 0; k_iter < in_c; k_iter++)
						{
							res += in[y_index * in_step + x_index * in_c + k_iter] *
								w[kid_step + m_step + n_step + k_iter]; 
						}
					}
				}
			}
		}
		out[i * out_h * out_c + j * out_c + kid] = (with_RELU && res < 0) ? 0 : res;
	}
	return;
}

/*************************************************
Function:       MaxPool_Accel
Description:    Max pooling layer accelerater.
Calls:          NULL
Input:
				in			---->	Input feature map data pointer.
				out			---->	Output feature map data pointer.
				in_h		---->	Input height.
				out_h		---->	Output height.
				in_c		---->	Input channel.
				out_c		---->	Output channel.
				win_size	---->	Pooling window size.
				stride		---->	Stride.
Output:         out			---->	Output feature map data pointer.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

__global__ void MaxPool_Accel(DNN_dtype *in, DNN_dtype *out, int in_h, int out_h, int in_c, int win_size, int stride)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z;
	DNN_dtype res = -99999;
	DNN_dtype in_tmp = 0;
	int x_index = 0;
	int y_index = 0;

	if(i < out_h && j < out_h && k < in_c)
	{
		int in_step = in_h * in_c;
		int out_step = out_h * in_c;
		int win_down = (win_size % 2) ? (- win_size / 2) : (- win_size / 2 + 1);
		int win_up = win_size / 2 + 1;

		for(int k_iter = k; k_iter < in_c; k_iter += blockDim.z)
		{
			for(int m = win_down; m < win_up; m++)
			{
				y_index = i * stride + m;
				if(y_index >= 0 && y_index < in_h)
				{
					for(int n = win_down; n < win_up; n++)
					{
						x_index = j * stride + n;
						if(x_index >= 0 && x_index < in_h)
						{
							in_tmp = in[y_index * in_step + x_index * in_c + k_iter];
							res = (in_tmp > res) ? in_tmp : res;
						}
					}
				}
			}
			out[i * out_step + j * in_c + k_iter] = res;
			res = -99999;
		}
	}

	return;
}

/*************************************************
Function:       Norm_Accel
Description:    Normalization layer accelerater.
Calls:          NULL
Input:
				gpu_in		---->	In-output feature map data pointer.
				in_h		---->	Input height.
				in_c		---->	Input channel.
Output:         gpu_in		---->	In-output feature map data pointer.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

__global__ void Norm_Accel(DNN_dtype *gpu_in, int in_h, int in_c)
{
	DNN_dtype coef;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int in_step = in_h * in_c;

	for(int k = 0; k < in_c; k++)
	{
		coef += powf(gpu_in[i * in_step + j * in_c + k], 2);
	}
	coef = 5 / sqrtf(coef);

	for(int k = 0; k < in_c; k++)
	{
		gpu_in[i * in_step + j * in_c + k] = gpu_in[i * in_step + j * in_c + k] * coef;
	}

	return;
}

/*************************************************
Function:       Softmax_Accel
Description:    Softmax layer accelerater.
Calls:          NULL
Input:
				gpuin		---->	In-output feature map data pointer.
				class_num	---->	Number of class.
				in_h		---->	Input height.
				in_c		---->	Input channel.
				out_c		---->	Output channel.
Output:         gpuin		---->	In-output feature map data pointer.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

__global__ void Softmax_Accel(DNN_dtype *gpu_in, int class_num, int in_h, int in_c)
{
	DNN_dtype coef = 0;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = blockIdx.z * class_num;
	
	if(k < in_c)
	{
		int in_step = in_h * in_c;
		for(int k_iter = k; k_iter < (k + class_num); k_iter++)
		{
			coef += expf(gpu_in[i * in_step + j * in_c + k_iter]);
		}
		coef = 1 / coef;
		for(int k_iter = k; k_iter < (k + class_num); k_iter++)
		{
			gpu_in[i * in_step + j * in_c + k_iter] = expf(gpu_in[i * in_step + j * in_c + k_iter]) * coef;
		}
	}
	return;
}

/*************************************************
Function:       DNN_ConvLayer
Description:    Convolution layer.
Calls:          NULL
Input:
				in			---->	Input feature map.
				out			---->	Output feature map.
				kernel		---->	Kernel.
				pad			---->	Padding.
				stride		---->	Stride.
				dilation	---->	Dilation.
				with_RELU	---->	RELU or not?
Output:         out			---->	Output feature map.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_ConvLayer(DNN_MAT *in, DNN_MAT *out, DNN_KERNEL *kernel, int pad, int stride, int dilation, bool with_RELU)
{
	int T_h = 4;
	int T_w = T_h;
	int T_kid = 1024 / (T_h * T_w);
	dim3 grid((out->height + T_h - 1) / T_h, (out->width + T_w - 1) / T_w, (out->channel < 1024) ? ((out->channel + T_kid - 1) / T_kid) : (1024 / T_kid));
	dim3 block(T_h, T_w, T_kid);
	cudaFuncSetCacheConfig(Conv_Accel, cudaFuncCachePreferL1);
	clock_t start = clock(); 
	Conv_Accel<<<grid, block>>>(in->gpu_data, out->gpu_data, kernel->gpu_w, kernel->gpu_b, in->height, out->height, in->channel, out->channel, kernel->height, stride, dilation, with_RELU);
	clock_t end = clock();
}

/*************************************************
Function:       DNN_MaxPoolLayer
Description:    Maxpooling layer.
Calls:          NULL
Input:
				in			---->	Input feature map.
				out			---->	Output feature map.
				win_size	---->	Pooling window size.
				pad			---->	Padding.
				stride		---->	Stride.
Output:         out			---->	Output feature map.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_MaxPoolLayer(DNN_MAT *in, DNN_MAT *out, int win_size, int pad, int stride)
{
	int T_h = 4;
	int T_w = T_h;
	dim3 grid((out->height + T_h - 1) / T_h, (out->width + T_w - 1) / T_w);
	dim3 block(T_h, T_w, 512 / (T_h * T_w));	
	cudaFuncSetCacheConfig(Conv_Accel, cudaFuncCachePreferL1);
	MaxPool_Accel<<<grid, block>>>(in->gpu_data, out->gpu_data, in->height, out->height, in->channel, win_size, stride);
}

/*************************************************
Function:       DNN_NormLayer
Description:    Normalization layer.
Calls:          NULL
Input:
				in			---->	In-output feature map.
Output:         in			---->	In-output feature map.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_NormLayer(DNN_MAT *in)
{
	int T_h = 4;
	int T_w = T_h;
	dim3 grid((in->height + T_h - 1) / T_h, (in->width + T_w - 1) / T_w);
	dim3 block(T_h, T_w, 1);
	cudaFuncSetCacheConfig(Conv_Accel, cudaFuncCachePreferL1);
	Norm_Accel<<<grid, block>>>(in->gpu_data, in->height, in->channel);	
}

/*************************************************
Function:       DNN_NormLayer
Description:    Normalization layer.
Calls:          NULL
Input:
				in			---->	In-output feature map.
				class_num	---->	Number of class.
Output:         in			---->	In-output feature map.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_SoftmaxLayer(DNN_MAT *in, int class_num)
{
	int T_h = 4;
	int T_w = T_h;
	dim3 grid((in->height + T_h - 1) / T_h, (in->width + T_w - 1) / T_w, in->channel / class_num);
	dim3 block(T_h, T_w);	
	cudaFuncSetCacheConfig(Conv_Accel, cudaFuncCachePreferL1);
	Softmax_Accel<<<grid, block>>>(in->gpu_data, class_num, in->height, in->channel);	
}

/*************************************************
Function:       CPU2GPU
Description:    Transfer data form CPU to GPU.
Calls:          NULL
Input:
				mat			---->	DNN_MAT instance.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void MATCPU2GPU(DNN_MAT *mat)
{
	cudaMemcpy(mat->gpu_data, mat->data, mat->size * sizeof(DNN_dtype), cudaMemcpyHostToDevice);
}

/*************************************************
Function:       CPU2GPU
Description:    Transfer data form CPU to GPU.
Calls:          NULL
Input:
				ker			---->	DNN_KERNEL instance.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void KERNELCPU2GPU(DNN_KERNEL *ker)
{
	cudaMemcpy(ker->gpu_w, ker->w, ker->wsize * sizeof(DNN_dtype), cudaMemcpyHostToDevice);
	cudaMemcpy(ker->gpu_b, ker->b, ker->ker_num * sizeof(DNN_dtype), cudaMemcpyHostToDevice);
}

/*************************************************
Function:       GPU2CPU
Description:    Transfer data form GPU to CPU.
Calls:          NULL
Input:
				mat			---->	DNN_MAT instance.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void MATGPU2CPU(DNN_MAT *mat)
{
	cudaMemcpy(mat->data, mat->gpu_data, mat->size * sizeof(DNN_dtype), cudaMemcpyDeviceToHost);
}

/*************************************************
Function:       GPUMalloc
Description:    Malloc storage for gpu_data.
Calls:          NULL
Input:
				mat			---->	DNN_MAT instance.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void MATGPUMalloc(DNN_MAT *mat)
{
	cudaMalloc((void**)&mat->gpu_data, mat->size * sizeof(DNN_dtype));
}

/*************************************************
Function:       GPUMalloc
Description:    Malloc storage for gpu_data.
Calls:          NULL
Input:
				ker			---->	DNN_KERNEL instance.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void KERNELGPUMalloc(DNN_KERNEL *ker)
{
	cudaMalloc((void**)&ker->gpu_w, ker->wsize * sizeof(DNN_dtype));
	cudaMalloc((void**)&ker->gpu_b, ker->ker_num * sizeof(DNN_dtype));
}