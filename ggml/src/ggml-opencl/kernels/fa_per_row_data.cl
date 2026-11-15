#ifdef __OPENCL_VERSION__
#define TF4 float4
#else
#define TF4 cl_float4
#endif

struct fa_per_row_data
{
    TF4 m_i;
    TF4 d_i;
    TF4 m_i_minus_1;
    TF4 d_i_minus_1;
};
