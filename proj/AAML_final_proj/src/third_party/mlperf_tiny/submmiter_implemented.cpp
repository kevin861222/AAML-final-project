/*
Copyright 2020 EEMBC and The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file reflects a modified version of th_lib from EEMBC. The reporting logic
in th_results is copied from the original in EEMBC.
==============================================================================*/
/// \file
/// \brief C++ implementations of submitter_implemented.h

#include "third_party/mlperf_tiny/api/submitter_implemented.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "third_party/mlperf_tiny/api/internally_implemented.h"
#include "third_party/mlperf_tiny/util/quantization_helpers.h"

#include <stdio.h>

#include "tiny/v0.1/training/image_classification/trained_models/pretrainedResnet_quant.h"

#include "menu.h"
#include "tflite.h"
#include "perf.h"

const int kIcInputSize = 32*32*3;


// Implement this method to prepare for inference and preprocess inputs.
void th_load_tensor() {
  uint8_t input_quantized[kIcInputSize];
  int8_t input_asint[kIcInputSize];

  size_t bytes = ee_get_buffer(reinterpret_cast<uint8_t *>(input_quantized),
                               kIcInputSize * sizeof(uint8_t));
  if (bytes / sizeof(uint8_t) != kIcInputSize) {
    th_printf("Input db has %d elemented, expected %d\n", bytes / sizeof(uint8_t),
              kIcInputSize);
    return;
  }
  uint16_t i = 0;
  for(i=0; i<kIcInputSize;i++)
  {
	  if(input_quantized[i]<=127)
	    input_asint[i] = ((int8_t)input_quantized[i]) - 128;
	  else
	    input_asint[i] = (int8_t)(input_quantized[i] - 128);
  }
 
  tflite_set_input(input_asint);
  tflite_invoke_pre();
}

// Add to this method to return real inference results.
void th_results() {
  const int nresults = 10;
  /**
   * The results need to be printed back in exactly this format; if easier
   * to just modify this loop than copy to results[] above, do that.
   */
  th_printf("m-results-[");
  int kCategoryCount = 10;

  int8_t* output_data = tflite_get_output();

  for (int i = 0; i < kCategoryCount; i++) {
    th_printf("%d", output_data[i]);
    if (i < (nresults - 1)) {
      th_printf(",");
    }
  }
  th_printf("]\r\n");
}

// Implement this method with the logic to perform one inference cycle.
void th_infer() { tflite_invoke(); }

/// \brief optional API.
void th_final_initialize(void) {
  tflite_load_model(pretrainedResnet_quant, pretrainedResnet_quant_len);
}
void th_pre() {}
void th_post() {}

void th_command_ready(char volatile *p_command) {
  p_command = p_command;
  ee_serial_command_parser_callback((char *)p_command);
}

// th_libc implementations.
int th_strncmp(const char *str1, const char *str2, size_t n) {
  return strncmp(str1, str2, n);
}

char *th_strncpy(char *dest, const char *src, size_t n) {
  return strncpy(dest, src, n);
}

size_t th_strnlen(const char *str, size_t maxlen) {
  size_t i = 0;
  while(str[i] != '\0' && i < maxlen){
    i++;
  }
  return i;
}

char *th_strcat(char *dest, const char *src) { return strcat(dest, src); }

char *th_strtok(char *str1, const char *sep) { return strtok(str1, sep); }

int th_atoi(const char *str) { return atoi(str); }

void *th_memset(void *b, int c, size_t len) { return memset(b, c, len); }

void *th_memcpy(void *dst, const void *src, size_t n) {
  return memcpy(dst, src, n);
}

/* N.B.: Many embedded *printf SDKs do not support all format specifiers. */
int th_vprintf(const char *format, va_list ap) { return vprintf(format, ap); }
void th_printf(const char *p_fmt, ...) {
  va_list args;
  va_start(args, p_fmt);
  (void)th_vprintf(p_fmt, args); /* ignore return */
  va_end(args);
}

char th_getchar() { return getchar(); }

void th_serialport_initialize(void) {
// Already initialized in main(), skipped.
}

void th_timestamp(void) {
  # if EE_CFG_ENERGY_MODE==1
  timestampPin = 0;
  for (int i=0; i<100000; ++i) {
    asm("nop");
  }
  timestampPin = 1;
 #else
  unsigned long microSeconds = 0ul;
  /* USER CODE 2 BEGIN */
  microSeconds = static_cast<unsigned long>(perf_get_mcycle64()/75);
  /* USER CODE 2 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP, microSeconds);
  #endif
}

void th_timestamp_initialize(void) {
  /* USER CODE 1 BEGIN */
  // Setting up BOTH perf and energy here
  /* USER CODE 1 END */
  /* This message must NOT be changed. */
  th_printf(EE_MSG_TIMESTAMP_MODE);
  /* Always call the timestamp on initialize so that the open-drain output
     is set to "1" (so that we catch a falling edge) */
  th_timestamp();
}