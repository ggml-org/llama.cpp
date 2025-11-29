
echo Build BLIS CPP Template tests
make clean
make

echo Run tests
./test_asum_blis.x
./test_axpy_blis.x
./test_copy_blis.x
./test_dot_blis.x
./test_dotc_blis.x
./test_gbmv_blis.x
./test_gemm_blis.x
./test_gemv_blis.x
./test_ger_blis.x
./test_gerc_blis.x
./test_geru_blis.x
./test_hemm_blis.x
./test_hemv_blis.x
./test_her2_blis.x
./test_her_blis.x
./test_herk_blis.x
./test_hpr2_blis.x
./test_hpr_blis.x
./test_nrm2_blis.x
./test_rot_blis.x
./test_rotg_blis.x
./test_rotm_blis.x
./test_rotmg_blis.x
./test_scal_blis.x
./test_sdsdot_blis.x
./test_spr2_blis.x
./test_spr_blis.x
./test_swap_blis.x
./test_symm_blis.x
./test_syr2_blis.x
./test_syr2k_blis.x
./test_syr_blis.x
./test_syrk_blis.x
./test_tbmv_blis.x
./test_tbsv_blis.x
./test_tpmv_blis.x
./test_tpsv_blis.x
./test_trmm_blis.x
./test_trsm_blis.x
./test_trsv_blis.x
