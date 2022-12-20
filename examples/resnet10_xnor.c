/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      resnet18_xnor.c
 * \brief     XNOR implementation of ResNet18 
 * \author    Ravit Sharma 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

//////////////////////////////
// General Headers
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include "conv2_1.h"
#include "conv2_2.h"
#include "conv3_1.h"
#include "conv3_2.h"
#include "conv4_1.h"
#include "conv4_2.h"
#include "conv4_d.h"
#include "conv5_1.h"
#include "conv5_2.h"
#include "fc.h"

// Datatypes
#include "datatypes.h"
// NN functions
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "xnor_cn.h"

#include "special.h"

#define act_bw 1
#define weight_bw 1

static bnDtype conv2_1_act_unpacked[C2_1XY*C2_1XY*C2_1Z] = input_2_1;
static pckDtype conv2_1_act[act_bw*C2_1XY*C2_1XY*C2_1Z/pckWdt]; // = input_2_1_pack;
static pckDtype conv2_1_wgt[weight_bw*C2_1KZ*C2_1KXY*C2_1KXY*C2_1Z] = weight_2_1_pack;
static bnDtype conv2_1_thresh[C2_1KZ] = thr_2_1;
static pckDtype conv2_1_sign[C2_1KZ/pckWdt] = sign_2_1_pack;
//static pckDtype conv2_2_act[act_bw*C2_1OXY*C2_1OXY*C2_1KZ/pckWdt];
static bnDtype conv2_2_act_unpacked[act_bw*C2_1OXY*C2_1OXY*C2_1KZ];
static pckDtype conv2_2_act[act_bw*C2_2XY*C2_2XY*C2_2Z/pckWdt];// = input_2_2_pack;
static pckDtype conv2_2_wgt[weight_bw*C2_2KZ*C2_2KXY*C2_2KXY*C2_2Z] = weight_2_2_pack;
static bnDtype conv2_2_thresh[C2_2KZ] = thr_2_2;
static pckDtype conv2_2_sign[C2_2KZ/pckWdt] = sign_2_2_pack;
static pckDtype conv2_3_act[act_bw*C2_2OXY*C2_2OXY*C2_2KZ/pckWdt] = input_3_1_pack;// = output_2_2_pack;
static bnDtype conv2_3_act_unpacked[C2_2OXY*C2_2OXY*C2_2KZ] = input_3_1;// = output_2_2;
static bnDtype conv2_2_mean[C2_2KZ] = mu_2_2;
static bnDtype conv2_2_var[C2_2KZ] = sigma_2_2;
static bnDtype conv2_2_gamma[C2_2KZ] = gamma_2_2;
static bnDtype conv2_2_beta[C2_2KZ] = beta_2_2;

//static bnDtype conv2_3_act_unpacked[C3_1XY*C3_1XY*C3_1Z] = input_3_1;
//static pckDtype conv2_3_act[act_bw*C3_1XY*C3_1XY*C3_1Z/pckWdt]; // = input_3_1_pack;
static pckDtype conv3_1_wgt[weight_bw*C3_1KZ*C3_1KXY*C3_1KXY*C3_1Z] = weight_3_1_pack;
static bnDtype conv3_1_thresh[C3_1KZ] = thr_3_1;
static pckDtype conv3_1_sign[C3_1KZ/pckWdt] = sign_3_1_pack;
//static pckDtype conv3_2_act[act_bw*C3_1OXY*C3_1OXY*C3_1KZ/pckWdt];
static bnDtype conv3_2_act_unpacked[act_bw*C3_1OXY*C3_1OXY*C3_1KZ];
static pckDtype conv3_2_act[act_bw*C3_2XY*C3_2XY*C3_2Z/pckWdt];// = input_3_2_pack;
static pckDtype conv3_2_wgt[weight_bw*C3_2KZ*C3_2KXY*C3_2KXY*C3_2Z] = weight_3_2_pack;
static bnDtype conv3_2_thresh[C3_2KZ] = thr_3_2;
static pckDtype conv3_2_sign[C3_2KZ/pckWdt] = sign_3_2_pack;
static pckDtype conv3_3_act[act_bw*C3_2OXY*C3_2OXY*C3_2KZ/pckWdt];// = output_3_2_pack;
static bnDtype conv3_3_act_unpacked[C3_2OXY*C3_2OXY*C3_2KZ];// = output_3_2;
static bnDtype conv3_2_mean[C3_2KZ] = mu_3_2;
static bnDtype conv3_2_var[C3_2KZ] = sigma_3_2;
static bnDtype conv3_2_gamma[C3_2KZ] = gamma_3_2;
static bnDtype conv3_2_beta[C3_2KZ] = beta_3_2;

//static pckDtype conv3_3_act[act_bw*C4_1XY*C4_1XY*C4_1Z/pckWdt]; // = input_4_1_pack;
//static bnDtype conv3_3_act_unpacked[act_bw*C4_1XY*C4_1XY*C4_1Z] = input_4_1;
static pckDtype conv4_1_wgt[weight_bw*C4_1KZ*C4_1KXY*C4_1KXY*C4_1Z] = weight_4_1_pack;
static bnDtype conv4_1_thresh[C4_1KZ] = thr_4_1;
static pckDtype conv4_1_sign[C4_1KZ/pckWdt] = sign_4_1_pack;
//static pckDtype conv4_2_act[act_bw*C4_1OXY*C4_1OXY*C4_1KZ/pckWdt];
static bnDtype conv4_2_act_unpacked[C4_1OXY*C4_1OXY*C4_1KZ];
static pckDtype conv4_2_act[act_bw*C4_2XY*C4_2XY*C4_2Z/pckWdt];// = input_4_2_pack;
static pckDtype conv4_2_wgt[weight_bw*C4_2KZ*C4_2KXY*C4_2KXY*C4_2Z] = weight_4_2_pack;
static bnDtype conv4_2_thresh[C4_2KZ] = thr_4_2;
static pckDtype conv4_2_sign[C4_2KZ/pckWdt] = sign_4_2_pack;
static pckDtype conv4_3_act[act_bw*C4_2OXY*C4_2OXY*C4_2KZ/pckWdt];// = output_4_2_pack;
static bnDtype conv4_3_act_unpacked[C4_2OXY*C4_2OXY*C4_2KZ];// = output_4_2;
static bnDtype conv4_2_mean[C4_2KZ] = mu_4_2;
static bnDtype conv4_2_var[C4_2KZ] = sigma_4_2;
static bnDtype conv4_2_gamma[C4_2KZ] = gamma_4_2;
static bnDtype conv4_2_beta[C4_2KZ] = beta_4_2;
static bnDtype conv4_d_act_unpacked[C4_dOXY*C4_dOXY*C4_dKZ];// = output_4_2;
static pckDtype conv4_d_wgt[weight_bw*C4_dKZ*C4_dKXY*C4_dKXY*C4_dZ] = weight_4_d_pack;
static bnDtype conv4_d_mean[C4_dKZ] = mu_4_d;
static bnDtype conv4_d_var[C4_dKZ] = sigma_4_d;
static bnDtype conv4_d_gamma[C4_dKZ] = gamma_4_d;
static bnDtype conv4_d_beta[C4_dKZ] = beta_4_d;

//static bnDtype conv4_3_act_unpacked[C5_1XY*C5_1XY*C5_1Z] = input_5_1;
//static pckDtype conv4_3_act[act_bw*C5_1XY*C5_1XY*C5_1Z/pckWdt]; // = input_5_1_pack;
static pckDtype conv5_1_wgt[weight_bw*C5_1KZ*C5_1KXY*C5_1KXY*C5_1Z] = weight_5_1_pack;
static bnDtype conv5_1_thresh[C5_1KZ] = thr_5_1;
static pckDtype conv5_1_sign[C5_1KZ/pckWdt] = sign_5_1_pack;
//static pckDtype conv5_2_act[act_bw*C5_1OXY*C5_1OXY*C5_1KZ/pckWdt];
static bnDtype conv5_2_act_unpacked[act_bw*C5_1OXY*C5_1OXY*C5_1KZ];
static pckDtype conv5_2_act[act_bw*C5_2XY*C5_2XY*C5_2Z/pckWdt];// = input_5_2_pack;
static pckDtype conv5_2_wgt[weight_bw*C5_2KZ*C5_2KXY*C5_2KXY*C5_2Z] = weight_5_2_pack;
static bnDtype conv5_2_thresh[C5_2KZ] = thr_5_2;
static pckDtype conv5_2_sign[C5_2KZ/pckWdt] = sign_5_2_pack;
static pckDtype conv5_3_act[act_bw*C5_2OXY*C5_2OXY*C5_2KZ/pckWdt];// = output_5_2_pack;
static bnDtype conv5_3_act_unpacked[C5_2OXY*C5_2OXY*C5_2KZ];// = output_5_2;
static bnDtype conv5_2_mean[C5_2KZ] = mu_5_2;
static bnDtype conv5_2_var[C5_2KZ] = sigma_5_2;
static bnDtype conv5_2_gamma[C5_2KZ] = gamma_5_2;
static bnDtype conv5_2_beta[C5_2KZ] = beta_5_2;

static bnDtype fc_in[FI]; //C5_2KZ
static pckDtype fc_wgt[FI*FO/pckWdt] = weight__pack;
static pckDtype fc_wgt_unpacked[FI*FO] = weight_;
static bnDtype fc_out[FO];

int main(void) {
   struct identity_block_conf block2_conf = {.C_1_act_unpacked= conv2_1_act_unpacked, .C_1_act= conv2_1_act, .C_2_act= conv2_2_act, .C_3_act_unpacked= conv2_3_act_unpacked, .C_1_wgt= conv2_1_wgt, .C_2_wgt= conv2_2_wgt, .C_2_mean= conv2_2_mean, .C_2_var= conv2_2_var, .C_2_gamma= conv2_2_gamma, .C_2_beta= conv2_2_beta, .C_1_thresh= conv2_1_thresh, .C_1_sign= conv2_1_sign, .C_1XY= C2_1XY, .C_1Z= C2_1Z, .C_1KXY= C2_1KXY, .C_1KZ= C2_1KZ, .C_1PD= C2_1PD, .C_1PL= C2_1PL, .C_2XY= C2_2XY, .C_2Z= C2_2Z, .C_2KXY= C2_2KXY, .C_2KZ= C2_2KZ, .C_2PD= C2_2PD, .C_2PL= C2_2PL, .C_2OXY= C2_2OXY};
   identity_block(block2_conf);

   struct identity_block_conf block3_conf = {.C_1_act_unpacked= conv2_3_act_unpacked, .C_1_act= conv2_3_act, .C_2_act= conv3_2_act, .C_3_act_unpacked= conv3_3_act_unpacked, .C_1_wgt= conv3_1_wgt, .C_2_wgt= conv3_2_wgt, .C_2_mean= conv3_2_mean, .C_2_var= conv3_2_var, .C_2_gamma= conv3_2_gamma, .C_2_beta= conv3_2_beta, .C_1_thresh= conv3_1_thresh, .C_1_sign= conv3_1_sign, .C_1XY= C3_1XY, .C_1Z= C3_1Z, .C_1KXY= C3_1KXY, .C_1KZ= C3_1KZ, .C_1PD= C3_1PD, .C_1PL= C3_1PL, .C_2XY= C3_2XY, .C_2Z= C3_2Z, .C_2KXY= C3_2KXY, .C_2KZ= C3_2KZ, .C_2PD= C3_2PD, .C_2PL= C3_2PL, .C_2OXY= C3_2OXY};
   identity_block(block3_conf);

   struct convolutional_block_conf block4_conf = {.C_1_act_unpacked= conv3_3_act_unpacked, .C_1_act= conv3_3_act, .C_2_act= conv4_2_act, .C_3_act_unpacked= conv4_3_act_unpacked, .C_1_wgt= conv4_1_wgt, .C_2_wgt= conv4_2_wgt, .C_2_mean= conv4_2_mean, .C_2_var= conv4_2_var, .C_2_gamma= conv4_2_gamma, .C_2_beta= conv4_2_beta, .C_1_thresh= conv4_1_thresh, .C_1_sign= conv4_1_sign, .C_1XY= C4_1XY, .C_1Z= C4_1Z, .C_1KXY= C4_1KXY, .C_1KZ= C4_1KZ, .C_1PD= C4_1PD, .C_1PL= C4_1PL, .C_2XY= C4_2XY, .C_2Z= C4_2Z, .C_2KXY= C4_2KXY, .C_2KZ= C4_2KZ, .C_2PD= C4_2PD, .C_2PL= C4_2PL, .C_2OXY= C4_2OXY, .C_d_act_unpacked=conv4_d_act_unpacked, .C_d_wgt=conv4_d_wgt, .C_d_mean=conv4_d_mean, .C_d_var=conv4_d_var, .C_d_gamma=conv4_d_gamma, .C_d_beta=conv4_d_beta, .C_dKXY=C4_dKXY,  .C_dKZ=C4_dKZ,  .C_dPD=C4_dPD,  .C_dPL=C4_dPL,  .C_dOXY=C4_dOXY};
   convolutional_block(block4_conf);

   struct identity_block_conf block5_conf = {.C_1_act_unpacked= conv4_3_act_unpacked, .C_1_act= conv4_3_act, .C_2_act= conv5_2_act, .C_3_act_unpacked= conv5_3_act_unpacked, .C_1_wgt= conv5_1_wgt, .C_2_wgt= conv5_2_wgt, .C_2_mean= conv5_2_mean, .C_2_var= conv5_2_var, .C_2_gamma= conv5_2_gamma, .C_2_beta= conv5_2_beta, .C_1_thresh= conv5_1_thresh, .C_1_sign= conv5_1_sign, .C_1XY= C5_1XY, .C_1Z= C5_1Z, .C_1KXY= C5_1KXY, .C_1KZ= C5_1KZ, .C_1PD= C5_1PD, .C_1PL= C5_1PL, .C_2XY= C5_2XY, .C_2Z= C5_2Z, .C_2KXY= C5_2KXY, .C_2KZ= C5_2KZ, .C_2PD= C5_2PD, .C_2PL= C5_2PL, .C_2OXY= C5_2OXY};
   identity_block(block5_conf);

   //printFloatArray(conv5_3_act_unpacked+C5_2OXY*C5_2OXY*C5_2Z-20, 20);
   //pack(conv5_3_act_unpacked, conv5_3_act, C5_2OXY*C5_2OXY*C5_2Z);
   //printPackedArray(conv5_3_act, C5_2OXY*C5_2OXY*C5_2Z/pckWdt);

   averagePool1_1(conv5_3_act_unpacked, fc_in, C5_1KZ, C5_2OXY, C5_2OXY);
   
   bwn_fc(fc_in, fc_wgt, fc_out, FI, FO);
   normalize(fc_out, FO);
   
   printFloatArray(fc_out, FO);
   
   return (EXIT_SUCCESS);
}
