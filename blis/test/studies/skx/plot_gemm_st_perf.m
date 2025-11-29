addpath(pathname)

output_st_sgemm_asm_blis
output_st_dgemm_asm_blis
output_st_cgemm_1m_blis
output_st_zgemm_1m_blis

output_st_sgemm_openblas
output_st_dgemm_openblas
output_st_cgemm_openblas
output_st_zgemm_openblas

output_st_sgemm_mkl
output_st_dgemm_mkl
output_st_cgemm_mkl
output_st_zgemm_mkl

% SGEMM Single threaded

axes1 = subplot(4, 4, 1);
hold(axes1,'on');
plot(data_st_sgemm_asm_blis(:,1), data_st_sgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_sgemm_openblas(:,1), data_st_sgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_sgemm_mkl(:,1), data_st_sgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SGEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% DGEMM Single threaded

axes1 = subplot(4, 4, 5);
hold(axes1,'on');
plot(data_st_dgemm_asm_blis(:,1), data_st_dgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_dgemm_openblas(:,1), data_st_dgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_dgemm_mkl(:,1), data_st_dgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DGEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

% CGEMM Single threaded

axes1 = subplot(4, 4, 9);
hold(axes1,'on');
plot(data_st_cgemm_1m_blis(:,1), data_st_cgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_cgemm_openblas(:,1), data_st_cgemm_openblas(:,4),'--', 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_cgemm_mkl(:,1), data_st_cgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CGEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% ZGEMM Single threaded

axes1 = subplot(4, 4, 13);
hold(axes1,'on');
plot(data_st_zgemm_1m_blis(:,1), data_st_zgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_zgemm_openblas(:,1), data_st_zgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_zgemm_mkl(:,1), data_st_zgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZGEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

clear *gemm*
rmpath(pathname)
