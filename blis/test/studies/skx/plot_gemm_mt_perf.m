addpath(pathname)

output_mt_sgemm_asm_blis
output_mt_dgemm_asm_blis
output_mt_cgemm_1m_blis
output_mt_zgemm_1m_blis

output_mt_sgemm_openblas
output_mt_dgemm_openblas
output_mt_cgemm_openblas
output_mt_zgemm_openblas

output_mt_sgemm_mkl
output_mt_dgemm_mkl
output_mt_cgemm_mkl
output_mt_zgemm_mkl


% SGEMM multi threaded

axes1 = subplot(4, 4, 1);
hold(axes1,'on');
plot(data_mt_sgemm_asm_blis(:,1), data_mt_sgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_sgemm_openblas(:,1), data_mt_sgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_sgemm_mkl(:,1), data_mt_sgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores] )


% DGEMM multi threaded

axes1 = subplot(4, 4, 5);
hold(axes1,'on');
plot(data_mt_dgemm_asm_blis(:,1), data_mt_dgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_dgemm_openblas(:,1), data_mt_dgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_dgemm_mkl(:,1), data_mt_dgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'bemt');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CGEMM multi threaded

axes1 = subplot(4, 4, 9);
hold(axes1,'on');
plot(data_mt_cgemm_1m_blis(:,1), data_mt_cgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_cgemm_openblas(:,1), data_mt_cgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_cgemm_mkl(:,1), data_mt_cgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% ZGEMM multi threaded

axes1 = subplot(4, 4, 13);
hold(axes1,'on');
plot(data_mt_zgemm_1m_blis(:,1), data_mt_zgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_zgemm_openblas(:,1), data_mt_zgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_zgemm_mkl(:,1), data_mt_zgemm_mkl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
legend({'BLIS', 'OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

clear *gemm*
rmpath(pathname)
