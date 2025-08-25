#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tsavorite.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>

#define NUM_INPUT_TENSORS 2
#define NUM_INPUT_URINARY_TENSORS 1
#define  NUM_ELEMENTS 32
#define  NUM_ELEMENTS_SCALE 32*4 + 25

// index 0 for addition, index 1 for subtraction, index 2 for multiplication, index 3 for division
float test_input_1[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{1.1,  2.3,  3.2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SUB KERNEL
	{2.2,  10.3,  10.4,  2.2,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//MULT KERNEL
	{1.1,  2.3,  3.2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//DIV KERNEL
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	// SQRT Kernel
	{1,  4,  9.6,  16,  25,  36,  49,  64,  81,  100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024},
	//SQR Kernel
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//NEG Kernel
	{1.1,  -4.4,  10,  -5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -23, 24, 25, -26, 27, -28, 29, -30, 31, -32.6},
	//ABS Kernel
	{1.1,  -4.4,  10,  -5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -23, 24, 25, -26, 27, -28, 29, -30, 31, -32.6},
	//SIN Kernel
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6}
	
};
float test_input_2[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{1.1,  2.2,  3.3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SUB KERNEL
	{1.1,  2.2,  3.0,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//MULT KERNEL
	{1.1,  2.2,  3.3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//DIV KERNEL
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//Below ROW value not used for Unary OPS-SQRT, NEG, ABS, SIN
	//SQRT KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//NEG KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//ABS KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SIN Kernel input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
};

float test_result[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{2.20, 4.50, 6.50, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 24.00, 26.00, 28.00, 30.00, 32.00, 34.00, 36.00, 38.00, 40.00, 42.00, 44.00, 46.00, 48.00, 50.00, 52.00, 54.00, 56.00, 58.00, 60.00, 62.00, 64.00},
	//SUB KERNEL
	{1.1, 8.1, 7.4, -1.8, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
	//MULT KERNEL
	{1.21, 5.06, 10.56, 16.00, 25.00, 36.00, 49.00, 64.00, 81.00, 100.00, 121.00, 144.00, 169.00, 196.00, 225.00, 256.00, 289.00, 324.00, 361.00, 400.00, 441.00, 484.00, 529.00, 576.00, 625.00, 676.00, 729.00, 784.00, 841.00, 900.00, 961.00, 1024.00},
	//DIV KERNEL
	{1.0, 2.0, 2, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	//SQRT Kernel
	{1,  2,  3.098387,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SQR Kernel
	{1,  4,  9,  16,  25,  36,  49,  64,  81,  100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024},
	//NEG Kernel
	{-1.1,  4.4,  -10,  5,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, 23, -24, -25, 26, -27, 28, -29, 30, -31, 32.6},
	//ABS Kernel
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6},
	//SIN Kernel
	{0.891207,  -0.951602,  -0.544021,  -0.958924,  -0.958924,  -0.279416,  0.656987,  0.989358,  0.412118,  -0.544021, -0.999990, -0.536573, 0.420167, 0.990607, 0.650288, -0.287903, -0.961398, -0.750987, 0.149877, 0.912945, 0.912945, 0.912945, -0.846220, -0.905578, -0.132352, 0.762559, 0.956376, 0.270906, -0.663634, -0.988032, -0.404039, 0.926149}
	
};

float test_input_scale_1[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	//ADD KERNEL
	{1.3, 2.3, 3.3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//SUB KERNEL
	{8.5, 2.5, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 64,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 2, 4, 8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//MULT KERNEL
	{1.5, 2.5, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//DIV KERNEL
	{4.2, 8.4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SQRT KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//NEG KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//ABS KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SIN KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}
};

float test_input_scale_2[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	// ADD KERNEL
	{1.3, 2.3, 3.3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	// SUB KERNEL
	{1, 8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 6, 8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	// MULT KERNEL
	{2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// DIV KERNEL
	{2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//Below ROW value not used for Unary OPS-SQRT, NEG, ABS, SIN
	//SQRT KERNEL input not used
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//NEG KERNEL input not used
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//ABS KERNEL input not used
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SIN KERNEL input not used
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}
};
float test_result_scale[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	// ADD KERNEL
	{2.6, 4.6, 6.6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50},
	// SUB KERNEL
	{7.5, -5.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  32,
	 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
        -5, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
	 3, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
	 1, 2,  5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	// MULT KERNEL
	{3, 5,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	// DIV KERNEL
	{2.1, 4.2,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	// SQRT KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 3, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 4, 5, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// NEG KERNEL
	{1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 9, -4, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 16, -25, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	// ABS KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// SIN KERNEL
	{-0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.412118,-0.756802, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.287903,-0.132352, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471}
};

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the backend to perform the computation (TSAVORITE)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static bool ggml_tsi_compare_two_float(float a, float b) {
    float epsilon = 1e-5;
    float absA = abs(a);
    float absB = abs(b);
    float diff = abs(a - b);
    float minV = std::numeric_limits<float>::min();
    float maxV = std::numeric_limits<float>::max();

    if (a == b) { // shortcut, handles infinities
        return true;
    } else if (a == 0 || b == 0 || (absA + absB < minV)) {
                        // a or b is zero or both are extremely close to it
                        // relative error is less meaningful here
        return diff < (epsilon * minV);
    }
    // use relative error
    return diff /std::min((absA + absB), maxV) < epsilon;
}


static bool load_model(simple_model & model, float * a, float * b, enum ggml_type data_type, int elements_A, int elements_B) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
    fprintf(stderr, "%s: using TSavorite backend \n", __func__);
    model.backend = ggml_backend_tsavorite_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_tsavorite_init() failed\n", __func__);
	return false;
    }

    int num_tensors;

    if (!b)
        num_tensors = NUM_INPUT_URINARY_TENSORS;
    else
        num_tensors = NUM_INPUT_TENSORS;

    // Since we are not passing the mem_buffer ggml context will create
    /* .mem_buffer = params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size) */
    // mem_buffer for ctx is used for any object creation and used for tensor data if
    // backend doesnt have own memory
    // Since we are using backend memory hence i have removed extra bytes: 100, removed from mem_size at below
    struct ggml_init_params params {
            /*.mem_size   =*/ (ggml_tensor_overhead() * num_tensors),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    fprintf(stderr, "\n Calculating mem_size %ld  %d  and creating ggml context \n", ggml_tensor_overhead(), num_tensors); 

    // create context
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", __func__);
	return false;
    }

    // create tensors
    // //  BELOW CODE NO CHANGE FOR tsavorite Backend
    // Tensor just created with OBJ(Structure)+Tensor(structure)
    // Still Buffer need to attached to Tensor since we are using Backend
    // We will using tsi_alloc called under tsavorite-backend

    fprintf(stderr, "\n Creating input Tensor \n");

    //int64_t ne[GGML_MAX_DIMS]; // number of elements
    //size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
    model.a = ggml_new_tensor_1d(model.ctx, data_type, elements_A);
    if (b)
        model.b = ggml_new_tensor_1d(model.ctx, data_type, elements_B);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    fprintf(stderr, "\n Creating Backend Buffer \n");

    // Here at ggml Context we have only two input tensors, hence backend memory is
    // created for two input tensors
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    fprintf(stderr, "\n Loading Input Tensor Data to Backend Buffer \n");

    // loading the data to tensor
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    if (b)
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));

    // create a array to print input tensor
    std::vector<float> out_data(ggml_nelements(model.a));
    // bring the data from the backend memory
    ggml_backend_tensor_get(model.a, out_data.data(), 0, ggml_nbytes(model.a));


    fprintf(stderr, "\nBringing  tensor data from Backend buffer and printing %d  tensor data:\n[", (int) model.a->ne[0]);

    for (int i = 0; i < model.a->ne[0] /* cols */; i++) {
        fprintf(stderr, " %.2f", out_data[i]);
    }
    fprintf(stderr, " ]\n");
    return true;
}

// build the compute graph
static struct ggml_cgraph * build_graph(const simple_model& model, enum ggml_tsavorite_kernel_type ops_type) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor * result;
    switch(ops_type) {
	    case GGML_TSAVORITE_KERNEL_TYPE_ADD:
    		result = ggml_add(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SUB:
    		result = ggml_sub(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_MULT:
    		result = ggml_mul(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_DIV:
    		result = ggml_div(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SQRT:
    		result = ggml_sqrt(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SQR:
    		result = ggml_sqr(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_NEG:
                result = ggml_neg(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_ABS:
                result = ggml_abs(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SIN:
                result = ggml_sin(ctx0, model.a);
		break;
	     default:
    		ggml_free(ctx0);
    		fprintf(stderr, "\n Non Supported Operation \n");
		return NULL;
    }
    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
static struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr, enum ggml_tsavorite_kernel_type ops_type) {
    // reset the allocator to free all the memory allocated during the previous inference

    fprintf(stderr, "\n Under Test case for  compute API creating  build_graph  \n");
    struct ggml_cgraph * gf = build_graph(model, ops_type);
    if (!gf) { 
	    fprintf(stderr, "\ncompute failed\n");
	    return NULL;
    }
	   
    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

enum ggml_tsavorite_kernel_type convert_testcase_to_ops_type (const char *testCase) {
        if (!strcmp(testCase,"add"))
            return GGML_TSAVORITE_KERNEL_TYPE_ADD;
        else if (!strcmp(testCase,"sub"))
            return GGML_TSAVORITE_KERNEL_TYPE_SUB;
        else if (!strcmp(testCase,"mult"))
            return GGML_TSAVORITE_KERNEL_TYPE_MULT;
        else if (!strcmp(testCase,"div"))
            return GGML_TSAVORITE_KERNEL_TYPE_DIV;
        else if (!strcmp(testCase,"sqrt"))
            return GGML_TSAVORITE_KERNEL_TYPE_SQRT;
        else if (!strcmp(testCase,"sqr"))
            return GGML_TSAVORITE_KERNEL_TYPE_SQR;
        else if (!strcmp(testCase,"neg"))
            return GGML_TSAVORITE_KERNEL_TYPE_NEG;
        else if (!strcmp(testCase,"abs"))
            return GGML_TSAVORITE_KERNEL_TYPE_ABS;
        else if (!strcmp(testCase,"sin"))
            return GGML_TSAVORITE_KERNEL_TYPE_SIN;

    	fprintf(stderr, "\n un-supported test case %s hence running default test case which is add operation  \n", testCase);
	return GGML_TSAVORITE_KERNEL_TYPE_ADD;
}

int main(int argc, char *argv[]) {
    ggml_time_init();
    bool test_case_flag = true;
    enum ggml_tsavorite_kernel_type ops_type;
    simple_model model;
    float *input1[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    float *input2[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    float *result_data[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    bool data_scale = false;

    int elements_A=0, elements_B=0;
    int num_of_input_tensors;

    if (argc > 1) {
    	ops_type = convert_testcase_to_ops_type(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "scale"))
		data_scale = true;
    } else {
	// Default Case
    	ops_type = convert_testcase_to_ops_type("add");
    }
    if (ops_type == GGML_TSAVORITE_KERNEL_TYPE_SQRT ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SQR ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_NEG ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_ABS ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SIN)
	    num_of_input_tensors = NUM_INPUT_URINARY_TENSORS;
    else 
	    num_of_input_tensors = NUM_INPUT_TENSORS;

    if (data_scale) {
	    input1[ops_type]      = test_input_scale_1[ops_type];
	    elements_A            = NUM_ELEMENTS_SCALE; 
	    if (num_of_input_tensors != NUM_INPUT_URINARY_TENSORS) {
	        input2[ops_type]      = test_input_scale_2[ops_type];
	        elements_B            = NUM_ELEMENTS_SCALE; 
	    }
	    result_data[ops_type] = test_result_scale[ops_type];
    } else {
	    input1[ops_type]      = test_input_1[ops_type];
	    elements_A            = NUM_ELEMENTS; 
	    if (num_of_input_tensors != NUM_INPUT_URINARY_TENSORS) {
	        input2[ops_type]      = test_input_2[ops_type];
	        elements_B            = NUM_ELEMENTS; 
	    }
	    result_data[ops_type] = test_result[ops_type];
    }

    if(!load_model(model, input1[ops_type], input2[ops_type], GGML_TYPE_F32, elements_A, elements_B)) {
	    fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	    return -1;
    }
    // since tsavorite-backend init set the debug level to none, we are overwritting here
    ggml_tsavorite_log_type_val = GGML_TSAVORITE_LOG_DEBUG;

    ggml_gallocr_t allocr = NULL;

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    if (!allocr) {
    	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }

    // create the worst case graph for memory usage estimation
    struct ggml_cgraph * gf = build_graph(model, ops_type);
    if (!gf) {
    	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }
    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

    fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);

    // perform computation
    struct ggml_tensor * result = compute(model, allocr, ops_type);
    if (!result) {
	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }
    fprintf(stderr, "\n Compute Done \n");

    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:

    fprintf(stderr, "\n operation type: %d, num of elements %d  \n", ops_type, (int) result->ne[0]);

    fprintf(stderr, "\n compute is also done \n");
    for (int i = 0; i < result->ne[0] /* cols */; i++) {
	if (ggml_tsi_compare_two_float(out_data[i], result_data[ops_type][i])) {
		continue;
	}
	test_case_flag = false;
    	fprintf(stderr, "\n result for index %d is not matching expected %f got %f \n", i, result_data[ops_type][i], out_data[i]);
    }

    if (test_case_flag == false) {
	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }
    fprintf(stderr, "\n\n TEST CASE PASSED \n\n");

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    //ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
