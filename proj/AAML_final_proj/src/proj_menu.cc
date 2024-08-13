/*
 * Copyright 2021 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "proj_menu.h"

#include <stdio.h>

// #include<ctime>
#include "cfu.h"
#include "menu.h"
#include "perf.h"
#include "third_party/mlperf_tiny/api/internally_implemented.h"
#include "third_party/mlperf_tiny/api/submitter_implemented.h"

namespace {

// Template Fn

void do_enter_mlperf_tiny(void){
  ee_benchmark_initialize();
  while (1) {
    int c;
    c = th_getchar();
    #ifndef MLPERF_TINY_NO_ECHO
      putchar(c);
    #endif
    ee_serial_callback(c);
  }
}

struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('0', "Enter MLPerf Tiny Benchmark Interface", do_enter_mlperf_tiny),
        MENU_END,
    },
};

};  // anonymous namespace

extern "C" void do_proj_menu() { menu_run(&MENU); }
