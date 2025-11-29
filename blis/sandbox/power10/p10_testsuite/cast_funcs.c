#include "cast_funcs.h"
#include "../bli_sandbox.h"

// bit map used for casting float to bfloat16
typedef union
{
    float v;
    struct
    {
        uint32_t m:23;
        uint32_t e:8;
        uint32_t s:1;
    } bits;
} float32_s;


// cast float16 into float
float cast_f16_to_f32(float16 val) 
{
    uint16_t in = val.v;
    float out;
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = in & 0x7fff;                       // Non-sign bits
    t2 = in & 0x8000;                       // Sign bit
    t3 = in & 0x7c00;                       // Exponent
    
    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000;                       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit

    *((uint32_t*)&out) = t1;
    return out;
}

// cast float to float16
float16 cast_f32_to_f16(const float in) 
{
    float16 f16_out;

    uint32_t inu = *((uint32_t*)&in);
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = inu & 0x7fffffff;                 // Non-sign bits
    t2 = inu & 0x80000000;                 // Sign bit
    t3 = inu & 0x7f800000;                 // Exponent
    
    t1 >>= 13;                             // Align mantissa on MSB
    t2 >>= 16;                             // Shift sign bit into position

    t1 -= 0x1c000;                         // Adjust bias

    t1 = (t3 < 0x38800000) ? 0 : t1; 
    t1 = (t3 > 0x47000000) ? 0x7bff : t1;
    t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

    t1 |= t2;                              // Re-insert sign bit

    f16_out.v = t1;
    return f16_out;
}


// cast float to bfloat16
bfloat16 cast_f32_to_bf16 (float val)
{
    bfloat16 bf16;
    float32_s f32;
    f32.v = val;
    bf16.bits.s = f32.bits.s;
    bf16.bits.e = f32.bits.e;
    bf16.bits.m = f32.bits.m >> 16;
    return bf16;
}

// cast bfloat16 to float
float cast_bf16_to_f32(bfloat16 val)
{
    float32_s f32;
    f32.bits.s = val.bits.s;
    f32.bits.e = val.bits.e;
    f32.bits.m = val.bits.m << 16;
    return f32.v;
}

// cast a nibbles struct to a float array
void cast_i4_to_f32(float *fvals, nibbles vals)
{
    int8_t val0 = vals.bits.nib1;
    int8_t val1 = vals.bits.nib2;

    val0 = (val0 >= 8 ? val0 - 16 : val0);
    val1 = (val1 >= 8 ? val1 - 16 : val1);

    fvals[0] = (float) val0;
    fvals[1] = (float) val1;
}

// condense two float vals to a nibbles struct
nibbles cast_f32_to_i4(float val0, float val1)
{
    nibbles vals;

    int8_t val0_ = ((int8_t)val0) & 0xf0;
    int8_t val1_ = ((int8_t)val1) & 0xf0;

    vals.bits.nib1 = val0_;
    vals.bits.nib2 = val1_;

    return vals;
}

// cast float matrix to float nibbles
void cast_f32_to_i4m(float *a_float, nibbles *a, int num_elems)
{
    int j=0;
    for(int i=0; i<num_elems; i+=2)
    {
        float val1 = a_float[i];
        float val0 = a_float[i+1];

        a[j] = cast_f32_to_i4(val0, val1);
        j++;
    }
}

// cast nibbles matrix to float matrix
void cast_i4_to_f32m(nibbles *a, float *a_float, int num_elems)
{
    int j=0;
    float *fvals = (float *)malloc(2*sizeof(float));
    for(int i=0; i<num_elems; i+=2)
    {
        nibbles vals = a[j];
        j++;
        cast_i4_to_f32(fvals, vals);
        a_float[i] = fvals[0];
        a_float[i+1] = fvals[1];
    }
    free(fvals);
}



// cast single element using C casting

EASY_CAST_FUNC(f32, f32, float,   float);
EASY_CAST_FUNC(f32, i32, float,     int);
EASY_CAST_FUNC(f32, i16, float, int16_t);
EASY_CAST_FUNC(f32,  i8, float,  int8_t);

EASY_CAST_FUNC(i32, f32,     int, float);
EASY_CAST_FUNC(i16, f32, int16_t, float);
EASY_CAST_FUNC( i8, f32,  int8_t, float);


// cast entire matrix buffer

CASTING_MATRIX_FUNC(f32,  f32, float,    float,  cast_f32_to_f32);
CASTING_MATRIX_FUNC(f32, bf16, float, bfloat16, cast_f32_to_bf16);
CASTING_MATRIX_FUNC(f32,  f16, float,  float16,  cast_f32_to_f16);
CASTING_MATRIX_FUNC(f32,  i32, float,      int,  cast_f32_to_i32);
CASTING_MATRIX_FUNC(f32,  i16, float,  int16_t,  cast_f32_to_i16);
CASTING_MATRIX_FUNC(f32,   i8, float,   int8_t,   cast_f32_to_i8);

CASTING_MATRIX_FUNC(bf16, f32, bfloat16, float, cast_bf16_to_f32);
CASTING_MATRIX_FUNC( f16, f32,  float16, float,  cast_f16_to_f32);
CASTING_MATRIX_FUNC( i32, f32,      int, float,  cast_i32_to_f32);
CASTING_MATRIX_FUNC( i16, f32,  int16_t, float,  cast_i16_to_f32);
CASTING_MATRIX_FUNC(  i8, f32,   int8_t, float,   cast_i8_to_f32);

