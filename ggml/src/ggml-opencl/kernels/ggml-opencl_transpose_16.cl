// 16-bit transpose, loading/storing an 8x8 tile of elements

kernel void kernel_transpose_16(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_3 = i<<3;
    const int j_3 = j<<3;

    ushort8 temp0 = as_ushort8(read_imagef(input, (j_3+0)*cols+i));
    ushort8 temp1 = as_ushort8(read_imagef(input, (j_3+1)*cols+i));
    ushort8 temp2 = as_ushort8(read_imagef(input, (j_3+2)*cols+i));
    ushort8 temp3 = as_ushort8(read_imagef(input, (j_3+3)*cols+i));
    ushort8 temp4 = as_ushort8(read_imagef(input, (j_3+4)*cols+i));
    ushort8 temp5 = as_ushort8(read_imagef(input, (j_3+5)*cols+i));
    ushort8 temp6 = as_ushort8(read_imagef(input, (j_3+6)*cols+i));
    ushort8 temp7 = as_ushort8(read_imagef(input, (j_3+7)*cols+i));

    write_imagef(output, (i_3+0)*rows+j, as_float4((ushort8)(temp0.s0, temp1.s0, temp2.s0, temp3.s0, temp4.s0, temp5.s0, temp6.s0, temp7.s0)));
    write_imagef(output, (i_3+1)*rows+j, as_float4((ushort8)(temp0.s1, temp1.s1, temp2.s1, temp3.s1, temp4.s1, temp5.s1, temp6.s1, temp7.s1)));
    write_imagef(output, (i_3+2)*rows+j, as_float4((ushort8)(temp0.s2, temp1.s2, temp2.s2, temp3.s2, temp4.s2, temp5.s2, temp6.s2, temp7.s2)));
    write_imagef(output, (i_3+3)*rows+j, as_float4((ushort8)(temp0.s3, temp1.s3, temp2.s3, temp3.s3, temp4.s3, temp5.s3, temp6.s3, temp7.s3)));
    write_imagef(output, (i_3+4)*rows+j, as_float4((ushort8)(temp0.s4, temp1.s4, temp2.s4, temp3.s4, temp4.s4, temp5.s4, temp6.s4, temp7.s4)));
    write_imagef(output, (i_3+5)*rows+j, as_float4((ushort8)(temp0.s5, temp1.s5, temp2.s5, temp3.s5, temp4.s5, temp5.s5, temp6.s5, temp7.s5)));
    write_imagef(output, (i_3+6)*rows+j, as_float4((ushort8)(temp0.s6, temp1.s6, temp2.s6, temp3.s6, temp4.s6, temp5.s6, temp6.s6, temp7.s6)));
    write_imagef(output, (i_3+7)*rows+j, as_float4((ushort8)(temp0.s7, temp1.s7, temp2.s7, temp3.s7, temp4.s7, temp5.s7, temp6.s7, temp7.s7)));
}
