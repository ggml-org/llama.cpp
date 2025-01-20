// 32-bit transpose, loading/storing a 4x4 tile of elements

kernel void kernel_transpose_32(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;

    float4 temp0 = read_imagef(input, (j_2+0)*cols+i);
    float4 temp1 = read_imagef(input, (j_2+1)*cols+i);
    float4 temp2 = read_imagef(input, (j_2+2)*cols+i);
    float4 temp3 = read_imagef(input, (j_2+3)*cols+i);

    write_imagef(output, (i_2+0)*rows+j, (float4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
    write_imagef(output, (i_2+1)*rows+j, (float4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imagef(output, (i_2+2)*rows+j, (float4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imagef(output, (i_2+3)*rows+j, (float4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));

}
