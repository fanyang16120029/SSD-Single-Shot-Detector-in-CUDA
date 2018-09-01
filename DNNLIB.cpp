/*************************************************
Copyright:IREAL & BJTU
Author: Fan Yang
Date:2018-08-27
Description: Deep learning lib for CUDA7.0.
**************************************************/

#include "DNNLIB.h"

/*************************************************
Function:       DNN_MAT::Data2TXT
Description:    Save feature map to txt file.
Calls:          NULL
Input:
				name		---->	Txt file name.
Output:         NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_MAT::Data2TXT(std::string name)
{
	std::fstream matfile(name);
	if (!matfile)
	{
		std::cout << "Error in DNN_MAT::DataTXT : Can not create a txt file." << std::endl;
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < channel; k++)
			{
				matfile << data[i * step + j * channel + k] << std::endl;
			}
		}
	}
	matfile.close();
}

/*************************************************
Function:       DNN_MAT::ReadBMP
Description:    Read from BMP file.
Calls:          NULL
Input:
				BMPName		---->	Name of BMP file.
				Blue		---->	Offset of blue channel.
				Green		---->	Offset of green channel.
				Red			---->	Offset of red channel.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_MAT::ReadBMP(std::string BMPName, int Blue, int Green, int Red)
{
	// Open BMP file.
	FILE *bmp_fp;
	if (!(bmp_fp = fopen(BMPName.data(), "rb")))
	{
		std::cout << "Error in DNN_MAT::ReadBMP: Cannot open this BMP file." << std::endl;
		return;
	}

	// Read file header.
	DNN_BITMAPFILEHEADER bfheader;
	fread(&bfheader.bfType, 2, 1, bmp_fp);
	fread(&bfheader.bfSize, 4, 1, bmp_fp);
	fread(&bfheader.bfReserved1, 2, 1, bmp_fp);
	fread(&bfheader.bfReserved2, 2, 1, bmp_fp);
	fread(&bfheader.bfOffBits, 4, 1, bmp_fp);

	// Check file format.
	if (bfheader.bfType != 0x4D42)
	{
		fclose(bmp_fp);
		std::cout << "Error in DNN_MAT::ReadBMP: This file is not a bit map." << std::endl;
		std::cout << "Find " << std::hex << bfheader.bfType << " here!" << std::endl;
		return;
	}

	// Read information header.
	DNN_BITMAPINFOHEADER biheader;
	fread(&biheader.biSize, 4, 1, bmp_fp);
	fread(&biheader.biWidth, 4, 1, bmp_fp);
	fread(&biheader.biHeight, 4, 1, bmp_fp);
	fread(&biheader.biPlanes, 2, 1, bmp_fp);
	fread(&biheader.biBitCount, 2, 1, bmp_fp);
	fread(&biheader.biCompression, 4, 1, bmp_fp);
	fread(&biheader.biSizeImage, 4, 1, bmp_fp);
	fread(&biheader.biXPelsPerMeter, 4, 1, bmp_fp);
	fread(&biheader.biYPelsPerMeter, 4, 1, bmp_fp);
	fread(&biheader.biClrUsed, 4, 1, bmp_fp);
	fread(&biheader.biClrImportant, 4, 1, bmp_fp);

	int bmp_width = int(biheader.biWidth);
	int bmp_height = int(biheader.biHeight);
	int bmp_bit_count = int(biheader.biBitCount);
	int bmp_channel = bmp_bit_count / 8;
	int bmp_step = ceil(bmp_width * bmp_channel / 4) * 4;
	int bmp_size = bmp_height * bmp_step;

	// Check information.
	if (this->channel > 3)
	{
		fclose(bmp_fp);
		std::cout << "Error in DNN_MAT::ReadBMP: Channel not matched." << std::endl;
		return;
	}
	if (bmp_bit_count == 24 && this->channel == 1)
	{
		fclose(bmp_fp);
		std::cout << "Error in DNN_MAT::ReadBMP: Channel not matched." << std::endl;
		return;
	}

	// Read pixels.
	BYTE *raw_data = new BYTE[bmp_size];
	fread(raw_data, sizeof(BYTE), bmp_size, bmp_fp);
	for (int i = 0; i < this->height; i++)
	{
		for (int j = 0; j < this->width; j++)
		{
			int x = i * bmp_height / this->height;
			int y = j * bmp_width / this->width ;
			if(bmp_channel == 1)
			{
				BYTE val = raw_data[(bmp_height - x - 1) * bmp_step + y * bmp_channel];
				for(int k = 0; k < this->channel; k++)
				{
					data[i * this->step + j * this->channel + k] = float(val);
				}
			}
			else
			{
				BYTE b = raw_data[(bmp_height - x - 1) * bmp_step + y * bmp_channel];
				data[i * this->step + j * this->channel] = float(b) - Blue;
				BYTE g = raw_data[(bmp_height - x - 1) * bmp_step + y * bmp_channel + 1];
				data[i * this->step + j * this->channel + 1] = float(g) - Green;
				BYTE r = raw_data[(bmp_height - x - 1) * bmp_step + y * bmp_channel + 2];
				data[i * this->step + j * this->channel + 2] = float(r) - Red;
			}
		}
	}

	// Delete allocated memory & handle.
	delete[] raw_data;
	fclose(bmp_fp);
	return;
}

/*************************************************
Function:       DNN_MAT::SaveBMP
Description:    Save to BMP file.
Calls:          NULL
Input:
				BMPName		---->	Name of BMP file.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_MAT::SaveBMP(std::string BMPName)
{
	// Initialize file header.
	DNN_BITMAPFILEHEADER bfheader;
	DNN_BITMAPINFOHEADER biheader = { 0 };

	bfheader.bfType = 0x4D42;
	bfheader.bfReserved1 = 0;
	bfheader.bfReserved2 = 0;
	bfheader.bfSize = 54 + int(ceil(this->step / 4)) * 4 * this->channel;
	bfheader.bfOffBits = 54;

	// Initialize information header.

	biheader.biSize = 40;
	biheader.biHeight = this->height;
	biheader.biWidth = this->width;
	biheader.biPlanes = 1;
	biheader.biBitCount = 3 * 8;
	biheader.biSizeImage = int(ceil(this->step / 4)) * 4 * this->channel;
	biheader.biCompression = 0;
	biheader.biXPelsPerMeter = 0;
	biheader.biYPelsPerMeter = 0;
	biheader.biClrUsed = 0;
	biheader.biClrImportant = 0;

	// Set data.
	BYTE *raw_data = new BYTE[int(ceil(this->step / 4)) * 4 * this->height];
	memset(raw_data, 0, ceil(this->step / 4) * 4 * this->height);

	for (int i = 0; i < this->height; i++)
	{
		for (int j = 0; j < this->width; j++)
		{
			BYTE val = 0;
			if (this->channel == 1)
			{
				val = data[(this->height - i - 1) * int(ceil(this->step / 4)) * 4 + j * this->channel];
				raw_data[i * this->step + j * 3 + 0] = val;
				raw_data[i * this->step + j * 3 + 1] = val;
				raw_data[i * this->step + j * 3 + 2] = val;
			}
			else
			{
				for (int k = 0; k < this->channel; k++)
				{
					val = data[(this->height - i - 1) * this->step + j * this->channel + k];
					raw_data[i * int(ceil(this->step / 4)) * 4 + j * 3 + k] = val;
				}
			}
		}
	}

	// Write to file
	FILE *bmp_fp = fopen(BMPName.data(), "wb");

	if (bmp_fp == NULL)
	{
		std::cout << "Error in DNN_MAT::SaveBMP: Can not open BMP file." << std::endl;
		return;
	}
	else
	{
		fwrite(&bfheader.bfType, 2, 1, bmp_fp);
		fwrite(&bfheader.bfSize, 4, 1, bmp_fp);
		fwrite(&bfheader.bfReserved1, 2, 1, bmp_fp);
		fwrite(&bfheader.bfReserved1, 2, 1, bmp_fp);
		fwrite(&bfheader.bfOffBits, 4, 1, bmp_fp);

		fwrite(&biheader.biSize, 4, 1, bmp_fp);
		fwrite(&biheader.biWidth, 4, 1, bmp_fp);
		fwrite(&biheader.biHeight, 4, 1, bmp_fp);
		fwrite(&biheader.biPlanes, 2, 1, bmp_fp);
		fwrite(&biheader.biBitCount, 2, 1, bmp_fp);
		fwrite(&biheader.biCompression, 4, 1, bmp_fp);
		fwrite(&biheader.biSizeImage, 4, 1, bmp_fp);
		fwrite(&biheader.biXPelsPerMeter, 4, 1, bmp_fp);
		fwrite(&biheader.biYPelsPerMeter, 4, 1, bmp_fp);
		fwrite(&biheader.biClrUsed, 4, 1, bmp_fp);
		fwrite(&biheader.biClrImportant, 4, 1, bmp_fp);

		fwrite(raw_data, size, 1, bmp_fp);
		fclose(bmp_fp);
	}
	delete[] raw_data;
}

/*************************************************
Function:       DNN_KERNEL::SetWeight
Description:    Load txt to weights.
Calls:          NULL
Input:
				name		---->	Name of txt file.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_KERNEL::SetWeight(std::string name)
{
	std::fstream wfile(name);
	if (!wfile)
	{
		std::cout << "Error in DNN_KERNEL::SetWeight : Can not open weight txt file." << std::endl;
		return;
	}
	float tmp = 0;
	char tmp_name[100];
	wfile >> tmp_name;
	int step1 = width * channel;
	int step2 = step1 * height;
	for (int kid = 0; kid < ker_num; kid++)
	{
		for (int k = 0; k < channel; k++)
		{
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					wfile >> tmp;
					w[kid *  step2 + i * step1 + j * channel + k] = tmp;
				}
			}
		}
	}
	wfile.close();
}

/*************************************************
Function:       DNN_KERNEL::SetBias
Description:    Load txt to bias.
Calls:          NULL
Input:
				name		---->	Name of txt file.
Output:         NULL
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_KERNEL::SetBias(std::string name)
{
	std::fstream bfile(name);
	if (!bfile)
	{
		std::cout << "Error in DNN_KERNEL::SetBias : Can not open bias txt file." << std::endl;
		return;
	}
	float tmp = 0;
	char tmp_name[100];
	bfile >> tmp_name;
	for (int kid = 0; kid < ker_num; kid++)
	{
		bfile >> tmp;
		b[kid] = tmp;
	}
	bfile.close();
}

/*************************************************
Function:       DNN_PriorboxLayer
Description:    Post-processsing layer.
Calls:          NULL
Input:
				prior		---->	Output prior box.
				scale		---->	Scale vector.
				ar			---->	Aspect ratio vector.
				img_height	---->	Image height.
				img_width	---->	Image width.
Output:         prior	---->	Output prior box.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_PriorboxLayer(DNN_MAT *prior, std::vector<float> &scale, std::vector<float> &ar)
{
	float h_step = 1 / (float)prior->height;
	float w_step = 1 / (float)prior->width;
	for (int i = 0; i < prior->height; i++)
	{
		for (int j = 0; j < prior->width; j++)
		{
			for (int ks = 0; ks < scale.size(); ks++)
			{
				for (int ka = 0; ka < ar.size(); ka++)
				{
					float cy = (i + 0.5) * h_step;
					float cx = (j + 0.5) * w_step;
					float h = scale[ks] / sqrt(ar[ka]);
					float w = scale[ks] * sqrt(ar[ka]);
					
					float lx = __max(cx - 0.5 * w, 0);
					float ly = __max(cy - 0.5 * h, 0);
					float rx = __min(cx + 0.5 * w, 1);
					float ry = __min(cy + 0.5 * h, 1);

					prior->data[i * prior->step + j * prior->channel + (ks * ar.size() + ka) * 4] = lx;
					prior->data[i * prior->step + j * prior->channel + (ks * ar.size() + ka) * 4 + 1] = ly;
					prior->data[i * prior->step + j * prior->channel + (ks * ar.size() + ka) * 4 + 2] = rx;
					prior->data[i * prior->step + j * prior->channel + (ks * ar.size() + ka) * 4 + 3] = ry;
				}
			}
		}
	}
}

/*************************************************
Function:       COMPARE
Description:    Bounding box vector compare.
Calls:          NULL
Input:
				in1		---->	First input bounding box.
				in2		---->	Second input bounding box.
Output:         NULL
Return:			1 > 2 (true) or 1 < 2 (false).
Others:         Run in CUDA 7.0.
*************************************************/

bool COMPARE(const DNN_BOUNDINGBOX &in1, const DNN_BOUNDINGBOX &in2)
{
	return(in1.score > in2.score);
}

/*************************************************
Function:       NMS
Description:    Non-maximum suppression.
Calls:          NULL
Input:
				in		---->	Input bounding box.
				out		---->	Output bounding box.
Output:         out		---->	Output bounding box.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void NMS(std::vector<DNN_BOUNDINGBOX> &in, std::vector<DNN_BOUNDINGBOX> &out, float iou_th, float vis_th)
{
	sort(in.begin(), in.end(), COMPARE);
	while (!in.empty())
	{
		if (in[0].score > vis_th)
		{
			float lx1 = in[0].lx;
			float ly1 = in[0].ly;
			float rx1 = in[0].rx;
			float ry1 = in[0].ry;

			for (int i = 1; i < in.size(); i++)
			{
				float lx2 = in[i].lx;
				float ly2 = in[i].ly;
				float rx2 = in[i].rx;
				float ry2 = in[i].ry;

				float xx1 = __max(lx1, lx2);
				float yy1 = __max(ly1, ly2);
				float xx2 = __min(rx1, rx2);
				float yy2 = __min(ry1, ry2);

				float w = __max(xx2 - xx1, 0);
				float h = __max(yy2 - yy1, 0);

				float inter = w * h;

				float IOU = inter / (in[0].area + in[i].area - inter);

				if (IOU > iou_th)
				{
					in.erase(in.begin() + i);
					i--;
				}
			}
			out.push_back(in.front());
			in.erase(in.begin());
		}
		else
		{
			in.erase(in.begin());
		}
	}
}

/*************************************************
Function:       DNN_PostprocLayer
Description:    Post-processsing layer.
Calls:          NULL
Input:
				conf	---->	Input conf feature map.
				loc		---->	Input loc feature map.
				out		---->	Output bounding box.
Output:         out		---->	Output bounding box.
Return:			NULL
Others:         Run in CUDA 7.0.
*************************************************/

void DNN_PostprocLayer(std::vector<DNN_MAT *> &conf, std::vector<DNN_MAT *> &loc, std::vector<DNN_MAT *> &prior, std::vector <DNN_BOUNDINGBOX> &out)
{
	// Setting.
	int bg_id = 0;
	float conf_th = 0.1;
	int keep_topk = 200;
	float nms_th = 0.45;
	int num_class = 2;
	float visualize_th = 0.3;
	int mean_value[3] = { 115, 123, 117 };
	int img_size[2] = { 1920, 1080 };
	float MGP_decay = 0.9;

	// Generate bounding box.

	std::vector<DNN_BOUNDINGBOX> tmp_vec;

	for (int FM_id = 0; FM_id < conf.size(); FM_id++)
	{
		for (int i = 0; i < conf[FM_id]->height; i++)
		{
			for (int j = 0; j < conf[FM_id]->width; j++)
			{
				for (int k = 0; k < conf[FM_id]->channel / num_class; k++)
				{
					DNN_BOUNDINGBOX tmp;
					tmp.score = conf[FM_id]->data[i * conf[FM_id]->step + j * conf[FM_id]->channel + k * num_class + 1];
					if (tmp.score > conf_th)
					{
						if (tmp.score > 0.7)
						{
							tmp.score = tmp.score;
						}
						float lx, ly, rx, ry;
						lx = prior[FM_id]->data[i * prior[FM_id]->step + j * prior[FM_id]->channel + k * 4];
						ly = prior[FM_id]->data[i * prior[FM_id]->step + j * prior[FM_id]->channel + k * 4 + 1];
						rx = prior[FM_id]->data[i * prior[FM_id]->step + j * prior[FM_id]->channel + k * 4 + 2];
						ry = prior[FM_id]->data[i * prior[FM_id]->step + j * prior[FM_id]->channel + k * 4 + 3];

						float cx = 0.5 * (lx + rx);
						float cy = 0.5 * (ly + ry);
						float w = rx - lx;
						float h = ry - ly;

						cx = 0.1 * loc[FM_id]->data[i * loc[FM_id]->step + j * loc[FM_id]->channel + k * 4] * w + cx;
						cy = 0.1 * loc[FM_id]->data[i * loc[FM_id]->step + j * loc[FM_id]->channel + k * 4 + 1] * h + cy;
						w = exp(0.2 * loc[FM_id]->data[i * loc[FM_id]->step + j * loc[FM_id]->channel + k * 4 + 2])	* w;
						h = exp(0.2 * loc[FM_id]->data[i * loc[FM_id]->step + j * loc[FM_id]->channel + k * 4 + 3]) * h;

						tmp.cx = cx;
						tmp.cy = cy;
						tmp.lx = __max(cx - 0.5 * w, 0);
						tmp.ly = __max(cy - 0.5 * h, 0);
						tmp.rx = __min(cx + 0.5 * w, 1);
						tmp.ry = __min(cy + 0.5 * h, 1);
						tmp.width = tmp.rx - tmp.lx;
						tmp.height = tmp.ry - tmp.ly;

						tmp.area = tmp.height * tmp.width;
						tmp.u = 0;
						tmp.v = 0;
						tmp.Ps = 0;
						tmp.Px[0] = 0; tmp.Px[1] = 0; tmp.Px[2] = 0; tmp.Px[3] = 0;
						tmp.Py[0] = 0; tmp.Py[1] = 0; tmp.Py[2] = 0; tmp.Py[3] = 0;
						tmp_vec.push_back(tmp);
					}
				}
			}
		}
	}

	NMS(tmp_vec, out, nms_th, visualize_th);
}