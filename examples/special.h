#include <datatypes.h>
#include <math.h>
#include <stdio.h>

void printPackedArray(pckDtype *arr, int n) {
   for (int i = 0; i < n; i++)
      printf("%x ", arr[i]);
   printf("\n");
}

void printFloatArray(float *arr, int n) {
   for (int i = 0; i < n; i++)
      printf("%f ", arr[i]);
   printf("\n");
}

void pack(bnDtype* src, pckDtype* target, int n) {
   //assert n%pckWdt == 0
   for (int i = 0; i < n/pckWdt; i++) {
      for (int j = 0; j < pckWdt; j++) {
         int value = src[i*pckWdt + j]>=0 ? 1 : 0;
         target[i] &= ~(1 << (pckWdt-j-1)); //all 1s except 0 at index j
         target[i] |= (value << (pckWdt-j-1)); //all 0s except sign(src[i*pcWdt+j]) at index j
         //printf("%f %d %x\n", src[i*pckWdt + j], value, target[i]);
      }
   }
}

void reluPack(pckDtype *arr, int n) {
   for (int i = 0; i < n; i++)
      *arr++ = -1;
}

void reluFloat(bnDtype *arr, int n) {
   for (int i = 0; i < n; i++)
      *arr++ = *arr < 0 ? 0 : *arr;
}

//target += src;
void addFloat(bnDtype *src, bnDtype *target, int n) {
   for (int i = 0; i < n; i++) {
      target[i] += src[i];
   }
}

//height, width, depth
void averagePool1_1(bnDtype *in, bnDtype *out, const uint16_t dpth, const uint16_t wdth, const uint16_t hght) {
   for (uint16_t i = 0; i < dpth; i++) out[i]=0;

   const uint16_t yCoeff = wdth*dpth;
   const uint16_t xCoeff = dpth;
   
   for (uint16_t i = 0; i < hght; i++) {
      for (uint16_t j = 0; j < wdth; j++) {
         for (uint16_t k = 0; k < dpth; k++) {
            out[k] += in[i*yCoeff + j*xCoeff  + k];
         }
      }
   }

   for (uint16_t i = 0; i < dpth; i++) out[i]/=hght*wdth;
}

void bwn_fc(bnDtype *in, pckDtype *krn, bnDtype *out, const uint16_t numIn, const uint16_t numOut) {
   for (int i = 0; i < numOut; i++) {
      out[i] = 0;
      for (int j = 0; j < numIn; j++) {
         if ((krn[(i*numIn + j)/pckWdt]&(1<<(pckWdt - 1 - (i*numIn + j)%pckWdt)))!=0) out[i] += in[j];
         else out[i] -= in[j];
      }
   }
}

void bwn_fc_unpacked(bnDtype *in, bnDtype *krn, bnDtype *out, const uint16_t numIn, const uint16_t numOut) {
   for (int i = 0; i < numOut; i++) {
      out[i] = 0;
      for (int j = 0; j < numIn; j++) {
         if (krn[i*numIn + j]) out[i] += in[j];
         else out[i] -= in[j];
      }
   }
}

void normalize(bnDtype *in, int n) {
   float magnitude = 1e-3;
   for (int i = 0; i < n; i++) {
      magnitude += in[i]*in[i];
   }
   magnitude = sqrt(magnitude);

   for (int i = 0; i < n; i++) {
      in[i] /= magnitude;
   }
}

struct identity_block_conf {
   bnDtype *C_1_act_unpacked; pckDtype *C_1_act; pckDtype *C_2_act;bnDtype *C_3_act_unpacked; pckDtype *C_1_wgt; pckDtype *C_2_wgt; 
   bnDtype *C_2_mean; bnDtype *C_2_var; bnDtype *C_2_gamma; bnDtype *C_2_beta; bnDtype *C_1_thresh; pckDtype *C_1_sign;
   uint8_t C_1XY; uint8_t C_1Z; uint8_t C_1KXY; uint8_t C_1KZ; uint8_t C_1PD; uint8_t C_1PL;
   uint8_t C_2XY; uint8_t C_2Z; uint8_t C_2KXY; uint8_t C_2KZ; uint8_t C_2PD; uint8_t C_2PL; uint8_t C_2OXY;
};

struct convolutional_block_conf {
   bnDtype *C_1_act_unpacked; pckDtype *C_1_act; pckDtype *C_2_act; bnDtype *C_3_act_unpacked; pckDtype *C_1_wgt; pckDtype *C_2_wgt; 
   bnDtype *C_2_mean; bnDtype *C_2_var; bnDtype *C_2_gamma; bnDtype *C_2_beta; bnDtype *C_1_thresh; pckDtype *C_1_sign;
   uint8_t C_1XY; uint8_t C_1Z; uint8_t C_1KXY; uint8_t C_1KZ; uint8_t C_1PD; uint8_t C_1PL;
   uint8_t C_2XY; uint8_t C_2Z; uint8_t C_2KXY; uint8_t C_2KZ; uint8_t C_2PD; uint8_t C_2PL; uint8_t C_2OXY;
   bnDtype *C_d_act_unpacked; pckDtype *C_d_wgt; bnDtype *C_d_mean; bnDtype *C_d_var; bnDtype *C_d_gamma; bnDtype *C_d_beta;
   uint8_t C_dKXY; uint8_t C_dKZ; uint8_t C_dPD; uint8_t C_dPL; uint8_t C_dOXY;
};

void identity_block(struct identity_block_conf s)
{
   pack(s.C_1_act_unpacked, s.C_1_act, s.C_1XY*s.C_1XY*s.C_1Z);
   //printPackedArray(s.C_1_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);
   CnXnorWrap(s.C_1_act, s.C_1_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_1KXY, s.C_1KXY, s.C_1KZ, s.C_2_act, s.C_1PD, s.C_1PL, s.C_1_thresh, s.C_1_sign);
   //printPackedArray(C_2_act, sizeof(C_2_act)/sizeof(pckDtype));
   reluPack(s.C_2_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);
   CnXnorNoBinWrap(s.C_2_act, s.C_2_wgt, s.C_2Z, s.C_2XY, s.C_2XY, s.C_2Z, s.C_2KXY, s.C_2KXY, s.C_2KZ, s.C_3_act_unpacked, s.C_2PD, s.C_2PL, s.C_2_mean, s.C_2_var, s.C_2_gamma, s.C_2_beta);
   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_1_act_unpacked, s.C_1XY*s.C_1XY*s.C_1Z);

   //printFloatArray(s.C_1_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   addFloat(s.C_1_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   reluFloat(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
}

void convolutional_block(struct convolutional_block_conf s)
{
   pack(s.C_1_act_unpacked, s.C_1_act, s.C_1XY*s.C_1XY*s.C_1Z);
   //printPackedArray(C_1_act, sizeof(C_1_act)/sizeof(pckDtype));
   CnXnorWrap(s.C_1_act, s.C_1_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_1KXY, s.C_1KXY, s.C_1KZ, s.C_2_act, s.C_1PD, s.C_1PL, s.C_1_thresh, s.C_1_sign);
   //printPackedArray(C_2_act, sizeof(C_2_act)/sizeof(pckDtype));
   reluPack(s.C_2_act, s.C_2XY*s.C_2XY*s.C_2Z/pckWdt);
   CnXnorNoBinWrap(s.C_2_act, s.C_2_wgt, s.C_2Z, s.C_2XY, s.C_2XY, s.C_2Z, s.C_2KXY, s.C_2KXY, s.C_2KZ, s.C_3_act_unpacked, s.C_2PD, s.C_2PL, s.C_2_mean, s.C_2_var, s.C_2_gamma, s.C_2_beta);
   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   
   //printPackedArray(s.C_1_act, C4_1XY*C4_1XY*C4_1Z/pckWdt);
   CnXnorNoBinWrap(s.C_1_act, s.C_d_wgt, s.C_1Z, s.C_1XY, s.C_1XY, s.C_1Z, s.C_dKXY, s.C_dKXY, s.C_dKZ, s.C_d_act_unpacked, s.C_dPD, s.C_dPL, s.C_d_mean, s.C_d_var, s.C_d_gamma, s.C_d_beta);
   //printFloatArray(s.C_d_act_unpacked, s.C_dOXY*s.C_dOXY*s.C_dKZ);
   
   //printFloatArray(s.C_d_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
   addFloat(s.C_d_act_unpacked, s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   reluFloat(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked, s.C_2OXY*s.C_2OXY*s.C_2KZ);
   //printFloatArray(s.C_3_act_unpacked+s.C_2OXY*s.C_2OXY*s.C_2KZ-20, 20);
}