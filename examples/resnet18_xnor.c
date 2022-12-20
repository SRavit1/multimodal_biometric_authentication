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


// Datatypes
#include "datatypes.h"
// NN functions
#include "utils.h"
#include "xnor_base.h"
#include "xnor_fc.h"
#include "bwn_dense_cn.h"
#include "xnor_cn.h"

#define act_bw 1
#define weight_bw 1

int main( void) {
   static pckDtype   conv1_wgt[weight_bw*64*3*7*7];
   static pckDtype   conv2_1_wgt[weight_bw*64*64*3*3/pckWdt];
   static pckDtype   conv2_2_wgt[weight_bw*64*64*3*3/pckWdt];
   static pckDtype   conv3_1_wgt[weight_bw*64*64*3*3/pckWdt];
   static pckDtype   conv3_2_wgt[weight_bw*64*64*3*3/pckWdt];
   static pckDtype   conv4_1_wgt[weight_bw*128*128*3*3/pckWdt];
   static pckDtype   conv4_2_wgt[weight_bw*128*128*3*3/pckWdt];
   static pckDtype   conv4_d_wgt[weight_bw*128*64*1*1/pckWdt];
   static pckDtype   conv5_1_wgt[weight_bw*128*128*3*3/pckWdt];
   static pckDtype   conv5_2_wgt[weight_bw*128*128*3*3/pckWdt];
   static pckDtype   conv6_1_wgt[weight_bw*256*256*3*3/pckWdt];
   static pckDtype   conv6_2_wgt[weight_bw*256*256*3*3/pckWdt];
   static pckDtype   conv6_d_wgt[weight_bw*256*128*1*1/pckWdt];
   static pckDtype   conv7_1_wgt[weight_bw*256*256*3*3/pckWdt];
   static pckDtype   conv7_2_wgt[weight_bw*256*256*3*3/pckWdt];
   static pckDtype   conv8_1_wgt[weight_bw*512*512*3*3/pckWdt];
   static pckDtype   conv8_2_wgt[weight_bw*512*512*3*3/pckWdt];
   static pckDtype   conv8_d_wgt[weight_bw*512*256*1*1/pckWdt];
   static pckDtype   conv9_1_wgt[weight_bw*512*512*3*3/pckWdt];
   static pckDtype   conv9_2_wgt[weight_bw*512*512*3*3/pckWdt];
   static int8_t     fc1_wgt[weight_bw*512*512];

   static int8_t     inp_act[act_bw*224*224*3];
   static pckDtype   conv1_act[act_bw*56*56*64/pckWdt];
   static pckDtype   conv2_1_act[act_bw*56*56*64/pckWdt];
   static pckDtype   conv2_2_act[act_bw*56*56*64/pckWdt];
   static pckDtype   conv3_1_act[act_bw*56*56*64/pckWdt];
   static pckDtype   conv3_2_act[act_bw*56*56*64/pckWdt];
   static pckDtype   conv4_1_act[act_bw*28*28*128/pckWdt];
   static pckDtype   conv4_2_act[act_bw*28*28*128/pckWdt];
   static pckDtype   conv4_d_act[act_bw*28*28*128/pckWdt];
   static pckDtype   conv5_1_act[act_bw*28*28*128/pckWdt];
   static pckDtype   conv5_2_act[act_bw*28*28*128/pckWdt];
   static pckDtype   conv6_1_act[act_bw*14*14*256/pckWdt];
   static pckDtype   conv6_2_act[act_bw*14*14*256/pckWdt];
   static pckDtype   conv6_d_act[act_bw*14*14*256/pckWdt];
   static pckDtype   conv7_1_act[act_bw*14*14*256/pckWdt];
   static pckDtype   conv7_2_act[act_bw*14*14*256/pckWdt];
   static pckDtype   conv8_1_act[act_bw*7*7*512/pckWdt];
   static pckDtype   conv8_2_act[act_bw*7*7*512/pckWdt];
   static pckDtype   conv8_d_act[act_bw*7*7*512/pckWdt];
   static pckDtype   conv9_1_act[act_bw*7*7*512/pckWdt];
   static pckDtype   conv9_2_act[act_bw*7*7*512/pckWdt];
   static pckDtype   fc1_act[act_bw*512];

   /*
      problems
      - pooling and stride different
   */

   clock_t t;
   const int rep = 100;
   t = clock();
   for (int i = 0; i < rep; i++) {
      
      //conv1
      CnXnorWrap(inp_act, conv1_wgt, 3, 256, 256, 3, 7, 7, 64, conv1_act, 1, 4, 0, 0);

      //conv2
      CnXnorWrap(conv1_act, conv2_1_wgt, 64, 56, 56, 64, 3, 3, 64, conv2_1_act, 1, 1, 0, 0);
      CnXnorWrap(conv2_1_act, conv2_2_wgt, 64, 56, 56, 64, 3, 3, 64, conv2_2_act, 1, 1, 0, 0);
      
      //conv3
      CnXnorWrap(conv2_2_act, conv3_1_wgt, 64, 56, 56, 64, 3, 3, 64, conv3_1_act, 1, 1, 0, 0);
      CnXnorWrap(conv3_1_act, conv3_2_wgt, 64, 56, 56, 64, 3, 3, 64, conv3_2_act, 1, 1, 0, 0);

      //conv4
      CnXnorWrap(conv3_2_act, conv1_wgt, 64, 56, 56, 64, 3, 3, 128, conv4_1_act, 1, 2, 0, 0);
      CnXnorWrap(conv4_1_act, conv1_wgt, 128, 28, 28, 128, 3, 3, 128, conv4_2_act, 1, 1, 0, 0);
      CnXnorWrap(conv3_2_act, conv1_wgt, 64, 56, 56, 64, 1, 1, 128, conv4_d_act, 1, 2, 0, 0);

      //conv5
      CnXnorWrap(conv4_2_act, conv1_wgt, 128, 28, 28, 128, 3, 3, 128, conv5_1_act, 1, 1, 0, 0);
      CnXnorWrap(conv5_1_act, conv1_wgt, 128, 28, 28, 128, 3, 3, 128, conv5_2_act, 1, 1, 0, 0);

      //conv6
      CnXnorWrap(conv5_2_act, conv6_1_wgt, 128, 28, 28, 128, 3, 3, 256, conv6_1_act, 1, 2, 0, 0);
      CnXnorWrap(conv6_1_act, conv6_2_wgt, 256, 14, 14, 256, 3, 3, 256, conv6_2_act, 1, 1, 0, 0);
      CnXnorWrap(conv5_2_act, conv6_d_wgt, 128, 28, 28, 128, 1, 1, 256, conv6_d_act, 0, 2, 0, 0);

      //conv7
      CnXnorWrap(conv6_d_act, conv7_1_wgt, 256, 14, 14, 256, 3, 3, 256, conv7_1_act, 1, 1, 0, 0);
      CnXnorWrap(conv7_1_act, conv7_2_wgt, 256, 14, 14, 256, 3, 3, 256, conv7_2_act, 1, 1, 0, 0);

      //conv8
      CnXnorWrap(conv7_2_act, conv8_1_wgt, 256, 14, 14, 256, 3, 3, 512, conv8_1_act, 1, 2, 0, 0);
      CnXnorWrap(conv8_1_act, conv8_2_wgt, 512, 7, 7, 512, 3, 3, 512, conv8_2_act, 1, 1, 0, 0);
      CnXnorWrap(conv7_2_act, conv8_d_wgt, 256, 14, 14, 512, 1, 1, 512, conv8_d_act, 1, 2, 0, 0);

      //conv9
      CnXnorWrap(conv8_d_act, conv9_1_wgt, 512, 7, 7, 512, 3, 3, 512, conv9_1_act, 1, 1, 0, 0);
      CnXnorWrap(conv9_1_act, conv9_2_wgt, 512, 7, 7, 512, 3, 3, 512, conv9_2_act, 1, 1, 0, 0);

      //fc1
      FcXnorWrap(conv9_2_act, fc1_wgt, 512, 512, fc1_act, 0, 0);
   }
   t = clock() - t;
   printf ("CPU runtime %f ms.\n",((float)t*1000)/(CLOCKS_PER_SEC*rep));
   
   return (EXIT_SUCCESS);
}
