/*************************************************
Copyright:IREAL & BJTU
Author: Fan Yang
Date:2018-08-27
Description: Deep learning lib for CUDA7.0.
**************************************************/

#include "Main.h"

class Net
{
public:
	DNN_KERNEL *conv1_1;
	DNN_KERNEL *conv1_2;

	DNN_KERNEL *conv2_1;
	DNN_KERNEL *conv2_2;

	DNN_KERNEL *conv3_1;
	DNN_KERNEL *conv3_2; 
	DNN_KERNEL *conv3_3;

	DNN_KERNEL *conv4_1;
	DNN_KERNEL *conv4_2;
	DNN_KERNEL *conv4_3;

	DNN_KERNEL *conv5_1;
	DNN_KERNEL *conv5_2;
	DNN_KERNEL *conv5_3;

	DNN_KERNEL *fc6;
	DNN_KERNEL *fc7;

	DNN_KERNEL *conv6_1;
	DNN_KERNEL *conv6_2;

	DNN_KERNEL *conv7_1;
	DNN_KERNEL *conv7_2;

	DNN_KERNEL *conv8_1;
	DNN_KERNEL *conv8_2;

	DNN_KERNEL *conv9_1;
	DNN_KERNEL *conv9_2;

	DNN_KERNEL *conv10_1;
	DNN_KERNEL *conv10_2;

	DNN_KERNEL *conv4_3_norm_mbox_conf;
	DNN_KERNEL *conv4_3_norm_mbox_loc;

	DNN_KERNEL *fc7_mbox_conf;
	DNN_KERNEL *fc7_mbox_loc;

	DNN_KERNEL *conv6_2_mbox_conf;
	DNN_KERNEL *conv6_2_mbox_loc;

	DNN_KERNEL *conv7_2_mbox_conf;
	DNN_KERNEL *conv7_2_mbox_loc;

	DNN_KERNEL *conv8_2_mbox_conf;
	DNN_KERNEL *conv8_2_mbox_loc;

	DNN_KERNEL *conv9_2_mbox_conf;
	DNN_KERNEL *conv9_2_mbox_loc;

	DNN_KERNEL *conv10_2_mbox_conf;
	DNN_KERNEL *conv10_2_mbox_loc;

	DNN_MAT *conv1_1_fm;
	DNN_MAT *conv1_2_fm;
	DNN_MAT *pool1_fm;

	DNN_MAT *conv2_1_fm;
	DNN_MAT *conv2_2_fm;
	DNN_MAT *pool2_fm;

	DNN_MAT *conv3_1_fm;
	DNN_MAT *conv3_2_fm;
	DNN_MAT *conv3_3_fm;
	DNN_MAT *pool3_fm;

	DNN_MAT *conv4_1_fm;
	DNN_MAT *conv4_2_fm;
	DNN_MAT *conv4_3_fm;
	DNN_MAT *pool4_fm;

	DNN_MAT *conv5_1_fm;
	DNN_MAT *conv5_2_fm;
	DNN_MAT *conv5_3_fm;
	DNN_MAT *pool5_fm;

	DNN_MAT *fc6_fm;
	DNN_MAT *fc7_fm;

	DNN_MAT *conv6_1_fm;
	DNN_MAT *conv6_2_fm;

	DNN_MAT *conv7_1_fm;
	DNN_MAT *conv7_2_fm;

	DNN_MAT *conv8_1_fm;
	DNN_MAT *conv8_2_fm;

	DNN_MAT *conv9_1_fm;
	DNN_MAT *conv9_2_fm;

	DNN_MAT *conv10_1_fm;
	DNN_MAT *conv10_2_fm;

	DNN_MAT *conv4_3_norm_mbox_conf_fm;
	DNN_MAT *conv4_3_norm_mbox_loc_fm;

	DNN_MAT *fc7_mbox_conf_fm;
	DNN_MAT *fc7_mbox_loc_fm;

	DNN_MAT *conv6_2_mbox_conf_fm;
	DNN_MAT *conv6_2_mbox_loc_fm;

	DNN_MAT *conv7_2_mbox_conf_fm;
	DNN_MAT *conv7_2_mbox_loc_fm;

	DNN_MAT *conv8_2_mbox_conf_fm;
	DNN_MAT *conv8_2_mbox_loc_fm;

	DNN_MAT *conv9_2_mbox_conf_fm;
	DNN_MAT *conv9_2_mbox_loc_fm;

	DNN_MAT *conv10_2_mbox_conf_fm;
	DNN_MAT *conv10_2_mbox_loc_fm;

	DNN_MAT *conv4_3_norm_mbox_prior;
	DNN_MAT *fc7_mbox_prior;
	DNN_MAT *conv6_2_mbox_prior;
	DNN_MAT *conv7_2_mbox_prior;
	DNN_MAT *conv8_2_mbox_prior;
	DNN_MAT *conv9_2_mbox_prior;
	DNN_MAT *conv10_2_mbox_prior;

	std::vector<DNN_MAT *> prior;
	std::vector<DNN_MAT *> conf;
	std::vector<DNN_MAT *> loc;

	std::vector<DNN_BOUNDINGBOX> output;

	Net(std::string path)
	{
		// Layer conv1
		conv1_1 = new DNN_KERNEL(64, 3, 3, 3);
		conv1_1->SetWeight(path + "/conv1_1_w.txt");
		conv1_1->SetBias(path + "/conv1_1_b.txt");
		KERNELGPUMalloc(conv1_1);
		KERNELCPU2GPU(conv1_1);

		conv1_2 = new DNN_KERNEL(64, 3, 3, 64);
		conv1_2->SetWeight(path + "/conv1_2_w.txt");
		conv1_2->SetBias(path + "/conv1_2_b.txt");
		KERNELGPUMalloc(conv1_2);
		KERNELCPU2GPU(conv1_2);

		// Layer conv2
		conv2_1 = new DNN_KERNEL(128, 3, 3, 64);
		conv2_1->SetWeight(path + "/conv2_1_w.txt");
		conv2_1->SetBias(path + "/conv2_1_b.txt");
		KERNELGPUMalloc(conv2_1);
		KERNELCPU2GPU(conv2_1);

		conv2_2 = new DNN_KERNEL(128, 3, 3, 128);
		conv2_2->SetWeight(path + "/conv2_2_w.txt");
		conv2_2->SetBias(path + "/conv2_2_b.txt");
		KERNELGPUMalloc(conv2_2);
		KERNELCPU2GPU(conv2_2);

		// Layer conv3
		conv3_1 = new DNN_KERNEL(256, 3, 3, 128);
		conv3_1->SetWeight(path + "/conv3_1_w.txt");
		conv3_1->SetBias(path + "/conv3_1_b.txt");
		KERNELGPUMalloc(conv3_1);
		KERNELCPU2GPU(conv3_1);

		conv3_2 = new DNN_KERNEL(256, 3, 3, 256);
		conv3_2->SetWeight(path + "/conv3_2_w.txt");
		conv3_2->SetBias(path + "/conv3_2_b.txt");
		KERNELGPUMalloc(conv3_2);
		KERNELCPU2GPU(conv3_2);

		conv3_3 = new DNN_KERNEL(256, 3, 3, 256);
		conv3_3->SetWeight(path + "/conv3_3_w.txt");
		conv3_3->SetBias(path + "/conv3_3_b.txt");
		KERNELGPUMalloc(conv3_3);
		KERNELCPU2GPU(conv3_3);

		// Layer conv4
		conv4_1 = new DNN_KERNEL(512, 3, 3, 256);
		conv4_1->SetWeight(path + "/conv4_1_w.txt");
		conv4_1->SetBias(path + "/conv4_1_b.txt");
		KERNELGPUMalloc(conv4_1);
		KERNELCPU2GPU(conv4_1);

		conv4_2 = new DNN_KERNEL(512, 3, 3, 512);
		conv4_2->SetWeight(path + "/conv4_2_w.txt");
		conv4_2->SetBias(path + "/conv4_2_b.txt");
		KERNELGPUMalloc(conv4_2);
		KERNELCPU2GPU(conv4_2);

		conv4_3 = new DNN_KERNEL(512, 3, 3, 512);
		conv4_3->SetWeight(path + "/conv4_3_w.txt");
		conv4_3->SetBias(path + "/conv4_3_b.txt");
		KERNELGPUMalloc(conv4_3);
		KERNELCPU2GPU(conv4_3);

		// Layer conv5
		conv5_1 = new DNN_KERNEL(512, 3, 3, 512);
		conv5_1->SetWeight(path + "/conv5_1_w.txt");
		conv5_1->SetBias(path + "/conv5_1_b.txt");
		KERNELGPUMalloc(conv5_1);
		KERNELCPU2GPU(conv5_1);

		conv5_2 = new DNN_KERNEL(512, 3, 3, 512);
		conv5_2->SetWeight(path + "/conv5_2_w.txt");
		conv5_2->SetBias(path + "/conv5_2_b.txt");
		KERNELGPUMalloc(conv5_2);
		KERNELCPU2GPU(conv5_2);

		conv5_3 = new DNN_KERNEL(512, 3, 3, 512);
		conv5_3->SetWeight(path + "/conv5_3_w.txt");
		conv5_3->SetBias(path + "/conv5_3_b.txt");
		KERNELGPUMalloc(conv5_3);
		KERNELCPU2GPU(conv5_3);

		// Layer fc6
		fc6 = new DNN_KERNEL(1024, 3, 3, 512);
		fc6->SetWeight(path + "/fc6_w.txt");
		fc6->SetBias(path + "/fc6_b.txt");
		KERNELGPUMalloc(fc6);
		KERNELCPU2GPU(fc6);

		// Layer fc7
		fc7 = new DNN_KERNEL(1024, 1, 1, 1024);
		fc7->SetWeight(path + "/fc7_w.txt");
		fc7->SetBias(path + "/fc7_b.txt");
		KERNELGPUMalloc(fc7);
		KERNELCPU2GPU(fc7);

		// Layer conv6
		conv6_1 = new DNN_KERNEL(256, 1, 1, 1024);
		conv6_1->SetWeight(path + "/conv6_1_w.txt");
		conv6_1->SetBias(path + "/conv6_1_b.txt");
		KERNELGPUMalloc(conv6_1);
		KERNELCPU2GPU(conv6_1);

		conv6_2 = new DNN_KERNEL(512, 3, 3, 256);
		conv6_2->SetWeight(path + "/conv6_2_w.txt");
		conv6_2->SetBias(path + "/conv6_2_b.txt");
		KERNELGPUMalloc(conv6_2);
		KERNELCPU2GPU(conv6_2);

		// Layer conv7
		conv7_1 = new DNN_KERNEL(128, 1, 1, 512);
		conv7_1->SetWeight(path + "/conv7_1_w.txt");
		conv7_1->SetBias(path + "/conv7_1_b.txt");
		KERNELGPUMalloc(conv7_1);
		KERNELCPU2GPU(conv7_1);

		conv7_2 = new DNN_KERNEL(256, 3, 3, 128);
		conv7_2->SetWeight(path + "/conv7_2_w.txt");
		conv7_2->SetBias(path + "/conv7_2_b.txt");
		KERNELGPUMalloc(conv7_2);
		KERNELCPU2GPU(conv7_2);

		// Layer conv8
		conv8_1 = new DNN_KERNEL(128, 1, 1, 256);
		conv8_1->SetWeight(path + "/conv8_1_w.txt");
		conv8_1->SetBias(path + "/conv8_1_b.txt");
		KERNELGPUMalloc(conv8_1);
		KERNELCPU2GPU(conv8_1);

		conv8_2 = new DNN_KERNEL(256, 3, 3, 128);
		conv8_2->SetWeight(path + "/conv8_2_w.txt");
		conv8_2->SetBias(path + "/conv8_2_b.txt");
		KERNELGPUMalloc(conv8_2);
		KERNELCPU2GPU(conv8_2);

		// Layer conv9
		conv9_1 = new DNN_KERNEL(128, 1, 1, 256);
		conv9_1->SetWeight(path + "/conv9_1_w.txt");
		conv9_1->SetBias(path + "/conv9_1_b.txt");
		KERNELGPUMalloc(conv9_1);
		KERNELCPU2GPU(conv9_1);

		conv9_2 = new DNN_KERNEL(256, 3, 3, 128);
		conv9_2->SetWeight(path + "/conv9_2_w.txt");
		conv9_2->SetBias(path + "/conv9_2_b.txt");
		KERNELGPUMalloc(conv9_2);
		KERNELCPU2GPU(conv9_2);

		// Layer conv10
		conv10_1 = new DNN_KERNEL(128, 1, 1, 256);
		conv10_1->SetWeight(path + "/conv10_1_w.txt");
		conv10_1->SetBias(path + "/conv10_1_b.txt");
		KERNELGPUMalloc(conv10_1);
		KERNELCPU2GPU(conv10_1);

		conv10_2 = new DNN_KERNEL(256, 4, 4, 128);
		conv10_2->SetWeight(path + "/conv10_2_w.txt");
		conv10_2->SetBias(path + "/conv10_2_b.txt");
		KERNELGPUMalloc(conv10_2);
		KERNELCPU2GPU(conv10_2);

		// Layer conv4_3_norm_mbox
		conv4_3_norm_mbox_conf = new DNN_KERNEL(6, 3, 3, 512);
		conv4_3_norm_mbox_conf->SetWeight(path + "/conv4_3_norm_mbox_confp_w.txt");
		conv4_3_norm_mbox_conf->SetBias(path + "/conv4_3_norm_mbox_confp_b.txt");
		KERNELGPUMalloc(conv4_3_norm_mbox_conf);
		KERNELCPU2GPU(conv4_3_norm_mbox_conf);

		conv4_3_norm_mbox_loc = new DNN_KERNEL(12, 3, 3, 512);
		conv4_3_norm_mbox_loc->SetWeight(path + "/conv4_3_norm_mbox_locp_w.txt");
		conv4_3_norm_mbox_loc->SetBias(path + "/conv4_3_norm_mbox_locp_b.txt");
		KERNELGPUMalloc(conv4_3_norm_mbox_loc);
		KERNELCPU2GPU(conv4_3_norm_mbox_loc);

		// Layer fc7_mbox
		fc7_mbox_conf = new DNN_KERNEL(4, 3, 3, 1024);
		fc7_mbox_conf->SetWeight(path + "/fc7_mbox_confp_w.txt");
		fc7_mbox_conf->SetBias(path + "/fc7_mbox_confp_b.txt");
		KERNELGPUMalloc(fc7_mbox_conf);
		KERNELCPU2GPU(fc7_mbox_conf);

		fc7_mbox_loc = new DNN_KERNEL(12, 3, 3, 1024);
		fc7_mbox_loc->SetWeight(path + "/fc7_mbox_locp_w.txt");
		fc7_mbox_loc->SetBias(path + "/fc7_mbox_locp_b.txt");
		KERNELGPUMalloc(fc7_mbox_loc);
		KERNELCPU2GPU(fc7_mbox_loc);

		// Layer conv6_2_mbox
		conv6_2_mbox_conf = new DNN_KERNEL(4, 3, 3, 512);
		conv6_2_mbox_conf->SetWeight(path + "/conv6_2_mbox_confp_w.txt");
		conv6_2_mbox_conf->SetBias(path + "/conv6_2_mbox_confp_b.txt");
		KERNELGPUMalloc(conv6_2_mbox_conf);
		KERNELCPU2GPU(conv6_2_mbox_conf);

		conv6_2_mbox_loc = new DNN_KERNEL(8, 3, 3, 512);
		conv6_2_mbox_loc->SetWeight(path + "/conv6_2_mbox_locp_w.txt");
		conv6_2_mbox_loc->SetBias(path + "/conv6_2_mbox_locp_b.txt");
		KERNELGPUMalloc(conv6_2_mbox_loc);
		KERNELCPU2GPU(conv6_2_mbox_loc);

		// Layer conv7_2_mbox
		conv7_2_mbox_conf = new DNN_KERNEL(4, 3, 3, 256);
		conv7_2_mbox_conf->SetWeight(path + "/conv7_2_mbox_confp_w.txt");
		conv7_2_mbox_conf->SetBias(path + "/conv7_2_mbox_confp_b.txt");
		KERNELGPUMalloc(conv7_2_mbox_conf);
		KERNELCPU2GPU(conv7_2_mbox_conf);

		conv7_2_mbox_loc = new DNN_KERNEL(8, 3, 3, 256);
		conv7_2_mbox_loc->SetWeight(path + "/conv7_2_mbox_locp_w.txt");
		conv7_2_mbox_loc->SetBias(path + "/conv7_2_mbox_locp_b.txt");
		KERNELGPUMalloc(conv7_2_mbox_loc);
		KERNELCPU2GPU(conv7_2_mbox_loc);

		// Layer conv8_2_mbox
		conv8_2_mbox_conf = new DNN_KERNEL(4, 3, 3, 256);
		conv8_2_mbox_conf->SetWeight(path + "/conv8_2_mbox_confp_w.txt");
		conv8_2_mbox_conf->SetBias(path + "/conv8_2_mbox_confp_b.txt");
		KERNELGPUMalloc(conv8_2_mbox_conf);
		KERNELCPU2GPU(conv8_2_mbox_conf);

		conv8_2_mbox_loc = new DNN_KERNEL(8, 3, 3, 256);
		conv8_2_mbox_loc->SetWeight(path + "/conv8_2_mbox_locp_w.txt");
		conv8_2_mbox_loc->SetBias(path + "/conv8_2_mbox_locp_b.txt");
		KERNELGPUMalloc(conv8_2_mbox_loc);
		KERNELCPU2GPU(conv8_2_mbox_loc);

		// Layer conv9_2_mbox
		conv9_2_mbox_conf = new DNN_KERNEL(4, 3, 3, 256);
		conv9_2_mbox_conf->SetWeight(path + "/conv9_2_mbox_confp_w.txt");
		conv9_2_mbox_conf->SetBias(path + "/conv9_2_mbox_confp_b.txt");
		KERNELGPUMalloc(conv9_2_mbox_conf);
		KERNELCPU2GPU(conv9_2_mbox_conf);

		conv9_2_mbox_loc = new DNN_KERNEL(8, 3, 3, 256);
		conv9_2_mbox_loc->SetWeight(path + "/conv9_2_mbox_locp_w.txt");
		conv9_2_mbox_loc->SetBias(path + "/conv9_2_mbox_locp_b.txt");
		KERNELGPUMalloc(conv9_2_mbox_loc);
		KERNELCPU2GPU(conv9_2_mbox_loc);

		// Layer conv10_2_mbox
		conv10_2_mbox_conf = new DNN_KERNEL(4, 3, 3, 256);
		conv10_2_mbox_conf->SetWeight(path + "/conv10_2_mbox_confp_w.txt");
		conv10_2_mbox_conf->SetBias(path + "/conv10_2_mbox_confp_b.txt");
		KERNELGPUMalloc(conv10_2_mbox_conf);
		KERNELCPU2GPU(conv10_2_mbox_conf);

		conv10_2_mbox_loc = new DNN_KERNEL(8, 3, 3, 256);
		conv10_2_mbox_loc->SetWeight(path + "/conv10_2_mbox_locp_w.txt");
		conv10_2_mbox_loc->SetBias(path + "/conv10_2_mbox_locp_b.txt");
		KERNELGPUMalloc(conv10_2_mbox_loc);
		KERNELCPU2GPU(conv10_2_mbox_loc);

		// Layer conv1
		conv1_1_fm = new DNN_MAT(512, 512, 64);
		MATGPUMalloc(conv1_1_fm);
		conv1_2_fm = new DNN_MAT(512, 512, 64);
		MATGPUMalloc(conv1_2_fm);
		pool1_fm = new DNN_MAT(256, 256, 64);
		MATGPUMalloc(pool1_fm);

		// Layer conv2
		conv2_1_fm = new DNN_MAT(256, 256, 128);
		MATGPUMalloc(conv2_1_fm);
		conv2_2_fm = new DNN_MAT(256, 256, 128);
		MATGPUMalloc(conv2_2_fm);
		pool2_fm = new DNN_MAT(128, 128, 128);
		MATGPUMalloc(pool2_fm);

		// Layer conv3
		conv3_1_fm = new DNN_MAT(128, 128, 256);
		MATGPUMalloc(conv3_1_fm);
		conv3_2_fm = new DNN_MAT(128, 128, 256);
		MATGPUMalloc(conv3_2_fm);
		conv3_3_fm = new DNN_MAT(128, 128, 256);
		MATGPUMalloc(conv3_3_fm);
		pool3_fm = new DNN_MAT(64, 64, 256);
		MATGPUMalloc(pool3_fm);

		// Layer conv4
		conv4_1_fm = new DNN_MAT(64, 64, 512);
		MATGPUMalloc(conv4_1_fm);
		conv4_2_fm = new DNN_MAT(64, 64, 512);
		MATGPUMalloc(conv4_2_fm);
		conv4_3_fm = new DNN_MAT(64, 64, 512);
		MATGPUMalloc(conv4_3_fm);
		pool4_fm = new DNN_MAT(32, 32, 512);
		MATGPUMalloc(pool4_fm);

		// Layer conv5
		conv5_1_fm = new DNN_MAT(32, 32, 512);
		MATGPUMalloc(conv5_1_fm);
		conv5_2_fm = new DNN_MAT(32, 32, 512);
		MATGPUMalloc(conv5_2_fm);
		conv5_3_fm = new DNN_MAT(32, 32, 512);
		MATGPUMalloc(conv5_3_fm);
		pool5_fm = new DNN_MAT(32, 32, 512);
		MATGPUMalloc(pool5_fm);

		// Layer fc6
		fc6_fm = new DNN_MAT(32, 32, 1024);
		MATGPUMalloc(fc6_fm);

		// Layer fc7
		fc7_fm = new DNN_MAT(32, 32, 1024);
		MATGPUMalloc(fc7_fm);

		// Layer conv6
		conv6_1_fm = new DNN_MAT(32, 32, 256);
		MATGPUMalloc(conv6_1_fm);
		conv6_2_fm = new DNN_MAT(16, 16, 512);
		MATGPUMalloc(conv6_2_fm);

		// Layer conv7
		conv7_1_fm = new DNN_MAT(16, 16, 128);
		MATGPUMalloc(conv7_1_fm);
		conv7_2_fm = new DNN_MAT(8, 8, 256);
		MATGPUMalloc(conv7_2_fm);

		// Layer conv8
		conv8_1_fm = new DNN_MAT(8, 8, 128);
		MATGPUMalloc(conv8_1_fm);
		conv8_2_fm = new DNN_MAT(4, 4, 256);
		MATGPUMalloc(conv8_2_fm);

		// Layer conv9
		conv9_1_fm = new DNN_MAT(4, 4, 128);
		MATGPUMalloc(conv9_1_fm);
		conv9_2_fm = new DNN_MAT(2, 2, 256);
		MATGPUMalloc(conv9_2_fm);

		// Layer conv10
		conv10_1_fm = new DNN_MAT(2, 2, 128);
		MATGPUMalloc(conv10_1_fm);
		conv10_2_fm = new DNN_MAT(1, 1, 256);
		MATGPUMalloc(conv10_2_fm);

		// Layer conv4_3_norm_mbox
		conv4_3_norm_mbox_conf_fm = new DNN_MAT(64, 64, 6);
		MATGPUMalloc(conv4_3_norm_mbox_conf_fm);
		conv4_3_norm_mbox_loc_fm = new DNN_MAT(64, 64, 12);
		MATGPUMalloc(conv4_3_norm_mbox_loc_fm);

		// Layer fc7_mbox
		fc7_mbox_conf_fm = new DNN_MAT(32, 32, 4);
		MATGPUMalloc(fc7_mbox_conf_fm);
		fc7_mbox_loc_fm = new DNN_MAT(32, 32, 8);
		MATGPUMalloc(fc7_mbox_loc_fm);
		
		// Layer conv6_2_mbox
		conv6_2_mbox_conf_fm = new DNN_MAT(16, 16, 4);
		MATGPUMalloc(conv6_2_mbox_conf_fm);
		conv6_2_mbox_loc_fm = new DNN_MAT(16, 16, 8);
		MATGPUMalloc(conv6_2_mbox_loc_fm);

		// Layer conv7_2_mbox
		conv7_2_mbox_conf_fm = new DNN_MAT(8, 8, 4);
		MATGPUMalloc(conv7_2_mbox_conf_fm);
		conv7_2_mbox_loc_fm = new DNN_MAT(8, 8, 8);
		MATGPUMalloc(conv7_2_mbox_loc_fm);

		// Layer conv8_2_mbox
		conv8_2_mbox_conf_fm = new DNN_MAT(4, 4, 4);
		MATGPUMalloc(conv8_2_mbox_conf_fm);
		conv8_2_mbox_loc_fm = new DNN_MAT(4, 4, 8);
		MATGPUMalloc(conv8_2_mbox_loc_fm);

		// Layer conv9_2_mbox
		conv9_2_mbox_conf_fm = new DNN_MAT(2, 2, 4);
		MATGPUMalloc(conv9_2_mbox_conf_fm);
		conv9_2_mbox_loc_fm = new DNN_MAT(2, 2, 8);
		MATGPUMalloc(conv9_2_mbox_loc_fm);

		// Layer conv10_2_mbox
		conv10_2_mbox_conf_fm = new DNN_MAT(1, 1, 4);
		MATGPUMalloc(conv10_2_mbox_conf_fm);
		conv10_2_mbox_loc_fm = new DNN_MAT(1, 1, 8);
		MATGPUMalloc(conv10_2_mbox_loc_fm);

		// Prior box.
		conv4_3_norm_mbox_prior = new DNN_MAT(64, 64, 12);
		fc7_mbox_prior = new DNN_MAT(32, 32, 8);
		conv6_2_mbox_prior = new DNN_MAT(16, 16, 8);
		conv7_2_mbox_prior = new DNN_MAT(8, 8, 8);
		conv8_2_mbox_prior = new DNN_MAT(4, 4, 8);
		conv9_2_mbox_prior = new DNN_MAT(2, 2, 8);
		conv10_2_mbox_prior = new DNN_MAT(1, 1, 8);

		std::vector<float> stmp,artmp;
		stmp.push_back(0.04);
		stmp.push_back(0.07);
		stmp.push_back(0.08);
		artmp.push_back(0.41);
		DNN_PriorboxLayer(conv4_3_norm_mbox_prior, stmp, artmp);
		prior.push_back(conv4_3_norm_mbox_prior);

		stmp.clear();
		artmp.clear();
		stmp.push_back(0.10);
		artmp.push_back(0.30);
		artmp.push_back(0.41);
		DNN_PriorboxLayer(fc7_mbox_prior, stmp, artmp);
		prior.push_back(fc7_mbox_prior);

		stmp.clear();
		stmp.push_back(0.25);
		DNN_PriorboxLayer(conv6_2_mbox_prior, stmp, artmp);
		prior.push_back(conv6_2_mbox_prior);

		stmp.clear();
		stmp.push_back(0.40);
		DNN_PriorboxLayer(conv7_2_mbox_prior, stmp, artmp);
		prior.push_back(conv7_2_mbox_prior);

		stmp.clear();
		stmp.push_back(0.55);
		DNN_PriorboxLayer(conv8_2_mbox_prior, stmp, artmp);
		prior.push_back(conv8_2_mbox_prior);

		stmp.clear();
		stmp.push_back(0.70);
		DNN_PriorboxLayer(conv9_2_mbox_prior, stmp, artmp);
		prior.push_back(conv9_2_mbox_prior);

		stmp.clear();
		stmp.push_back(0.85);
		DNN_PriorboxLayer(conv10_2_mbox_prior, stmp, artmp);
		prior.push_back(conv10_2_mbox_prior);
}
	void Forward(DNN_MAT *in)
	{
		// Layer conv1
		MATCPU2GPU(in);
		output.clear();
		conf.clear();
		loc.clear();
		DNN_ConvLayer(in, conv1_1_fm, conv1_1, 1, 1, 1, true);

		DNN_ConvLayer(conv1_1_fm, conv1_2_fm, conv1_2, 1, 1, 1, true);	

		DNN_MaxPoolLayer(conv1_2_fm, pool1_fm, 2, 0, 2);

		// Layer conv2
		DNN_ConvLayer(pool1_fm, conv2_1_fm, conv2_1, 1, 1, 1, true);

		DNN_ConvLayer(conv2_1_fm, conv2_2_fm, conv2_2, 1, 1, 1, true);

		DNN_MaxPoolLayer(conv2_2_fm, pool2_fm, 2, 0, 2);

		// Layer conv3
		DNN_ConvLayer(pool2_fm, conv3_1_fm, conv3_1, 1, 1, 1, true);

		DNN_ConvLayer(conv3_1_fm, conv3_2_fm, conv3_2, 1, 1, 1, true);

		DNN_ConvLayer(conv3_2_fm, conv3_3_fm, conv3_3, 1, 1, 1, true);

		DNN_MaxPoolLayer(conv3_3_fm, pool3_fm, 2, 0, 2);

		// Layer conv4
		DNN_ConvLayer(pool3_fm, conv4_1_fm, conv4_1, 1, 1, 1, true);

		DNN_ConvLayer(conv4_1_fm, conv4_2_fm, conv4_2, 1, 1, 1, true);

		DNN_ConvLayer(conv4_2_fm, conv4_3_fm, conv4_3, 1, 1, 1, true);

		DNN_MaxPoolLayer(conv4_3_fm, pool4_fm, 2, 0, 2);

		// Layer conv5
		DNN_ConvLayer(pool4_fm, conv5_1_fm, conv5_1, 1, 1, 1, true);

		DNN_ConvLayer(conv5_1_fm, conv5_2_fm, conv5_2, 1, 1, 1, true);

		DNN_ConvLayer(conv5_2_fm, conv5_3_fm, conv5_3, 1, 1, 1, true);

		DNN_MaxPoolLayer(conv5_3_fm, pool5_fm, 3, 1, 1);

		// Layer fc6
		DNN_ConvLayer(pool5_fm, fc6_fm, fc6, 6, 1, 6, true);

		// Layer fc7
		DNN_ConvLayer(fc6_fm, fc7_fm, fc7, 0, 1, 1, true);

		// Layer conv6
		DNN_ConvLayer(fc7_fm, conv6_1_fm, conv6_1, 0, 1, 1, true);

		DNN_ConvLayer(conv6_1_fm, conv6_2_fm, conv6_2, 1, 2, 1, true);

		// Layer conv7
		DNN_ConvLayer(conv6_2_fm, conv7_1_fm, conv7_1, 0, 1, 1, true);

		DNN_ConvLayer(conv7_1_fm, conv7_2_fm, conv7_2, 1, 2, 1, true);

		// Layer conv8
		DNN_ConvLayer(conv7_2_fm, conv8_1_fm, conv8_1, 0, 1, 1, true);

		DNN_ConvLayer(conv8_1_fm, conv8_2_fm, conv8_2, 1, 2, 1, true);

		// Layer conv9
		DNN_ConvLayer(conv8_2_fm, conv9_1_fm, conv9_1, 0, 1, 1, true);

		DNN_ConvLayer(conv9_1_fm, conv9_2_fm, conv9_2, 1, 2, 1, true);

		// Layer conv10
		DNN_ConvLayer(conv9_2_fm, conv10_1_fm, conv10_1, 0, 1, 1, true);

		DNN_ConvLayer(conv10_1_fm, conv10_2_fm, conv10_2, 1, 2, 1, true);

		// Layer norm4_3
		DNN_NormLayer(conv4_3_fm);

		// Layer conv4_3_norm_mbox
		DNN_ConvLayer(conv4_3_fm, conv4_3_norm_mbox_conf_fm, conv4_3_norm_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv4_3_norm_mbox_conf_fm, 2);
		MATGPU2CPU(conv4_3_norm_mbox_conf_fm);

		DNN_ConvLayer(conv4_3_fm, conv4_3_norm_mbox_loc_fm, conv4_3_norm_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv4_3_norm_mbox_loc_fm);
		conf.push_back(conv4_3_norm_mbox_conf_fm);
		loc.push_back(conv4_3_norm_mbox_loc_fm);

		// Layer fc7_mbox
		DNN_ConvLayer(fc7_fm, fc7_mbox_conf_fm, fc7_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(fc7_mbox_conf_fm, 2);
		MATGPU2CPU(fc7_mbox_conf_fm);

		DNN_ConvLayer(fc7_fm, fc7_mbox_loc_fm, fc7_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(fc7_mbox_loc_fm);
		conf.push_back(fc7_mbox_conf_fm);
		loc.push_back(fc7_mbox_loc_fm);

		// Layer conv6_2_mbox
		DNN_ConvLayer(conv6_2_fm, conv6_2_mbox_conf_fm, conv6_2_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv6_2_mbox_conf_fm, 2);
		MATGPU2CPU(conv6_2_mbox_conf_fm);

		DNN_ConvLayer(conv6_2_fm, conv6_2_mbox_loc_fm, conv6_2_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv6_2_mbox_loc_fm);
		conf.push_back(conv6_2_mbox_conf_fm);
		loc.push_back(conv6_2_mbox_loc_fm);

		// Layer conv7_2_mbox
		DNN_ConvLayer(conv7_2_fm, conv7_2_mbox_conf_fm, conv7_2_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv7_2_mbox_conf_fm, 2);
		MATGPU2CPU(conv7_2_mbox_conf_fm);

		DNN_ConvLayer(conv7_2_fm, conv7_2_mbox_loc_fm, conv7_2_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv7_2_mbox_loc_fm);
		conf.push_back(conv7_2_mbox_conf_fm);
		loc.push_back(conv7_2_mbox_loc_fm);

		// Layer conv8_2_mbox
		DNN_ConvLayer(conv8_2_fm, conv8_2_mbox_conf_fm, conv8_2_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv8_2_mbox_conf_fm, 2);
		MATGPU2CPU(conv8_2_mbox_conf_fm);

		DNN_ConvLayer(conv8_2_fm, conv8_2_mbox_loc_fm, conv8_2_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv8_2_mbox_loc_fm);
		conf.push_back(conv8_2_mbox_conf_fm);
		loc.push_back(conv8_2_mbox_loc_fm);

		// Layer conv9_2_mbox
		DNN_ConvLayer(conv9_2_fm, conv9_2_mbox_conf_fm, conv9_2_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv9_2_mbox_conf_fm, 2);
		MATGPU2CPU(conv9_2_mbox_conf_fm);

		DNN_ConvLayer(conv9_2_fm, conv9_2_mbox_loc_fm, conv9_2_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv9_2_mbox_loc_fm);
		conf.push_back(conv9_2_mbox_conf_fm);
		loc.push_back(conv9_2_mbox_loc_fm);

		// Layer conv10_2_mbox
		DNN_ConvLayer(conv10_2_fm, conv10_2_mbox_conf_fm, conv10_2_mbox_conf, 1, 1, 1, false);
		DNN_SoftmaxLayer(conv10_2_mbox_conf_fm, 2);
		MATGPU2CPU(conv10_2_mbox_conf_fm);

		DNN_ConvLayer(conv10_2_fm, conv10_2_mbox_loc_fm, conv10_2_mbox_loc, 1, 1, 1, false);
		MATGPU2CPU(conv10_2_mbox_loc_fm);
		conf.push_back(conv10_2_mbox_conf_fm);
		loc.push_back(conv10_2_mbox_loc_fm);
		
		// Post-processing.

		DNN_PostprocLayer(conf, loc, prior, output); 
		
		for (int i = 0; i < output.size(); i++)
		{
			std::cout << "Bounding Box: " << i << std::endl;
			std::cout << "Top-left : (" << output[i].lx << "," << output[i].ly << ")" << std::endl;
			std::cout << "Bottom-right : (" << output[i].rx << "," << output[i].ry << ")" << std::endl;
		}
	}
	~Net(void)
	{
		delete conv1_1;
		delete conv1_2;

		delete conv2_1;
		delete conv2_2;

		delete conv3_1;
		delete conv3_2;
		delete conv3_3;

		delete conv4_1;
		delete conv4_2;
		delete conv4_3;

		delete conv5_1;
		delete conv5_2;
		delete conv5_3;

		delete fc6;
		delete fc7;

		delete conv6_1;
		delete conv6_2;

		delete conv7_1;
		delete conv7_2;

		delete conv8_1;
		delete conv8_2;

		delete conv9_1;
		delete conv9_2;

		delete conv10_1;
		delete conv10_2;

		delete conv4_3_norm_mbox_conf;
		delete conv4_3_norm_mbox_loc;

		delete fc7_mbox_conf;
		delete fc7_mbox_loc;

		delete conv6_2_mbox_conf;
		delete conv6_2_mbox_loc;

		delete conv7_2_mbox_conf;
		delete conv7_2_mbox_loc;

		delete conv8_2_mbox_conf;
		delete conv8_2_mbox_loc;

		delete conv9_2_mbox_conf;
		delete conv9_2_mbox_loc;

		delete conv10_2_mbox_conf;
		delete conv10_2_mbox_loc;

		delete conv1_1_fm;
		delete conv1_2_fm;
		delete pool1_fm;

		delete conv2_1_fm;
		delete conv2_2_fm;
		delete pool2_fm;

		delete conv3_1_fm;
		delete conv3_2_fm;
		delete conv3_3_fm;
		delete pool3_fm;

		delete conv4_1_fm;
		delete conv4_2_fm;
		delete conv4_3_fm;
		delete pool4_fm;

		delete conv5_1_fm;
		delete conv5_2_fm;
		delete conv5_3_fm;
		delete pool5_fm;

		delete fc6_fm;
		delete fc7_fm;

		delete conv6_1_fm;
		delete conv6_2_fm;

		delete conv7_1_fm;
		delete conv7_2_fm;

		delete conv8_1_fm;
		delete conv8_2_fm;

		delete conv9_1_fm;
		delete conv9_2_fm;

		delete conv10_1_fm;
		delete conv10_2_fm;

		delete conv4_3_norm_mbox_conf_fm;
		delete conv4_3_norm_mbox_loc_fm;

		delete fc7_mbox_conf_fm;
		delete fc7_mbox_loc_fm;

		delete conv6_2_mbox_conf_fm;
		delete conv6_2_mbox_loc_fm;

		delete conv7_2_mbox_conf_fm;
		delete conv7_2_mbox_loc_fm;

		delete conv8_2_mbox_conf_fm;
		delete conv8_2_mbox_loc_fm;

		delete conv9_2_mbox_conf_fm;
		delete conv9_2_mbox_loc_fm;

		delete conv10_2_mbox_conf_fm;
		delete conv10_2_mbox_loc_fm;

		delete conv4_3_norm_mbox_prior;
		delete fc7_mbox_prior;
		delete conv6_2_mbox_prior;
		delete conv7_2_mbox_prior;
		delete conv8_2_mbox_prior;
		delete conv9_2_mbox_prior;
		delete conv10_2_mbox_prior;
	}
};

int main()
{
	// Reset GPU.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	cudaSetDevice(0);
	cudaDeviceReset();

	// Load kernel.
	Net SSD("E:/2017_10_19_ 毕业设计/Code/SSD Matlab Emulation/ssd512_weights_txt");

	// Load color image.
	DNN_MAT *input = new DNN_MAT(512, 512, 3);
	MATGPUMalloc(input);
	input->ReadBMP("E:/2017_10_19_ 毕业设计/Code/SSD C Emulation/lena512color.bmp", 115, 117, 123);
	std::cout << "read image done" << std::endl;

	clock_t start = clock();
	SSD.Forward(input);
	clock_t end = clock();
	std::cout << "Time= " << (end - start) / CLOCKS_PER_SEC << std::endl;

	return 0;

}