#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// solve_tri
//------------------------------------------------------------------------------
kernel void kernel_solve_tri_f32(
        global uchar * src0,
        ulong offset0,
        global uchar * src1,
        ulong offset1,
        global uchar * dst,
        ulong offsetd,
        int n,
        int k,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    int col = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    global const uchar * Lb = src0 + offset0 + i2 * nb02 + i3 * nb03;
    global const uchar * Bb = src1 + offset1 + i2 * nb12 + i3 * nb13;
    global       uchar * Xb = dst + offsetd + i2 * nb2 + i3 * nb3;

    for(int row = 0; row < n; ++row){
        global const float *pB = (global const float *)(Bb + row * nb11 + col * nb10);

        float sum = 0.0f;
        for(int j = 0; j < row; ++j){
            global const float *pL = (global const float *)(Lb + row * nb01 + j * nb00);
            global const float *pX = (global const float *)(Xb + j * nb1 + col * nb0);
            sum += (*pL) * (*pX);
        }

        global const float * pDiag = (global const float *)(Lb + row * nb01 + row *nb00);
        global float * pOut = (global float *)(Xb + row * nb1 + col *nb0);

        *pOut = ((* pB) - sum) / (*pDiag);
    }
}
