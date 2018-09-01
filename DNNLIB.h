/*************************************************
Copyright:IREAL & BJTU
Author: Fan Yang
Date:2018-08-27
Description: Deep learning lib for CUDA7.0.
**************************************************/

#ifndef DNNLIB_H
#define DNNLIB_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define __max(a,b) (((a) > (b)) ? (a) : (b))
#define __min(a,b) (((a) < (b)) ? (a) : (b))

/* Define data type. */
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;
typedef long  LONG;
typedef float DNN_dtype;

/* DNN_BITMAPFILEHEADER: Struct for BMP file header.*/
typedef struct {
	WORD    bfType;
	DWORD   bfSize;
	WORD    bfReserved1;
	WORD    bfReserved2;
	DWORD   bfOffBits;
} DNN_BITMAPFILEHEADER;

/* DNN_BITMAPFILEHEADER: Struct for BMP information header.*/
typedef struct {
	DWORD      biSize;
	LONG       biWidth;
	LONG       biHeight;
	WORD       biPlanes;
	WORD       biBitCount;
	DWORD      biCompression;
	DWORD      biSizeImage;
	LONG       biXPelsPerMeter;
	LONG       biYPelsPerMeter;
	DWORD      biClrUsed;
	DWORD      biClrImportant;
} DNN_BITMAPINFOHEADER;

typedef struct {
	float score;
	float area;
	float cx;
	float cy;
	float lx;
	float ly;
	float rx;
	float ry;
	float width;
	float height;
	float u;
	float v;
	float Ps;
	float Px[4];
	float Py[4];
} DNN_BOUNDINGBOX;

/* DNN_MAT: Class to storage 3 dim array. */
class DNN_MAT
{
public:
	DNN_dtype *data;									// Pointer to image data.
	DNN_dtype *gpu_data;
	int height = 0;										// Image height.
	int width = 0;										// Image width.
	int channel = 0;									// Image channel.
	int step = 0;										// Step of each column.
	int size = 0;										// Amount of pixel.

	DNN_MAT(int Height, int Width, int Channel) : height(Height), width(Width), channel(Channel)
	{
		// Check image size.
		if (Height <= 0 || Width <= 0 || Channel <= 0)
		{
			std::cout << "Error in BJTU_DNN_MAT::BJTU_DNN_MAT : Wrong Image size." << std::endl;
		}
		else
		{
			// Allocate memory for image data.
			step = Width * Channel;
			size = Height * step;
			data = new DNN_dtype[size];
			memset(data, 0, size * sizeof(DNN_dtype));
		}
	}
	DNN_MAT(const DNN_MAT & Other)
	{
		height = Other.height;
		width = Other.width;
		channel = Other.channel;

		// Allocate memory for image data.
		step = width * channel;
		size = height * step;
		data = new DNN_dtype[size];
		memcpy(data, Other.data, size);
	}
	void Data2TXT(std::string name);
	void ReadBMP(std::string BMPName, int Blue, int Green, int Red);
	void SaveBMP(std::string BMPName);
	~DNN_MAT()
	{
		delete[] data;
	}
};

/* DNN_KERNEL: Class to storage 4 dim kernel. */
class DNN_KERNEL
{
public:
	DNN_dtype *w;										// Pointer to weight.
	DNN_dtype *gpu_w;
	DNN_dtype *b;										// Pointer to bias.
	DNN_dtype *gpu_b;
	int ker_num = 0;									// Number of sub-kernels
	int height = 0;										// Height of kernel.
	int width = 0;										// Width of Kernel.
	int channel = 0;									// Channels of kernel.
	int wstep2 = 0;										// height * width * channel
	int wstep1 = 0;										// width * channel
	int wsize = 0;										// ker_num * height * width * channel

	DNN_KERNEL(int Ker_num, int Height, int Width, int Channel) : ker_num(Ker_num), height(Height), width(Width), channel(Channel)
	{
		// Check kernel size.
		if (Ker_num < 0 || Width <= 0 || Channel <= 0 || Height <= 0)
		{
			std::cout << "Error in DNN_KERNEL::DNN_KERNEL : Wrong Kernel size." << std::endl;
		}
		else
		{
			// Allocate memory for kernel data.
			wstep1 = Width * Channel;
			wstep2 = Height * wstep1;
			wsize = Ker_num * wstep2;
			w = new DNN_dtype[wsize];
			memset(w, 0, sizeof(DNN_dtype) * wsize);
			b = new DNN_dtype[Ker_num];
			memset(b, 0, sizeof(DNN_dtype) * Ker_num);
		}
	}
	void SetWeight(std::string name);
	void SetBias(std::string name);
	~DNN_KERNEL()
	{
		delete[] w;
		delete[] b;
	}
};

void DNN_PostprocLayer(std::vector<DNN_MAT *> &conf, std::vector<DNN_MAT *> &loc, std::vector<DNN_MAT *> &prior, std::vector <DNN_BOUNDINGBOX> &out);

void DNN_PriorboxLayer(DNN_MAT *prior, std::vector<float> &scale, std::vector<float> &ar);

#endif
