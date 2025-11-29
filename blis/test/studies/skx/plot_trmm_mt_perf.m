addpath(pathname)

output_mt_strmm_asm_blis
output_mt_dtrmm_asm_blis
output_mt_ctrmm_1m_blis
output_mt_ztrmm_1m_blis

output_mt_strmm_openblas
output_mt_dtrmm_openblas
output_mt_ctrmm_openblas
output_mt_ztrmm_openblas

output_mt_strmm_mkl
output_mt_dtrmm_mkl
output_mt_ctrmm_mkl
output_mt_ztrmm_mkl

% mtRMM multi threaded

axes1 = subplot(4, 4, 4);
hold(axes1,'on');
plot(data_mt_strmm_asm_blis(:,1), data_mt_strmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_strmm_openblas(:,1), data_mt_strmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_strmm_mkl(:,1), data_mt_strmm_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('STRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% DTRMM multi threaded

axes1 = subplot(4, 4, 8);
hold(axes1,'on');
plot(data_mt_dtrmm_asm_blis(:,1), data_mt_dtrmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_dtrmm_openblas(:,1), data_mt_dtrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_dtrmm_mkl(:,1), data_mt_dtrmm_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'bemt');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CTRMM multi threaded

axes1 = subplot(4, 4, 12);
hold(axes1,'on');
plot(data_mt_ctrmm_1m_blis(:,1), data_mt_ctrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_ctrmm_openblas(:,1), data_mt_ctrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_ctrmm_mkl(:,1), data_mt_ctrmm_mkl(:,3),'--',  'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% ZTRMM multi threaded

axes1 = subplot(4, 4, 16);
hold(axes1,'on');
plot(data_mt_ztrmm_1m_blis(:,1), data_mt_ztrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_ztrmm_openblas(:,1), data_mt_ztrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_ztrmm_mkl(:,1), data_mt_ztrmm_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

clear *trmm*
rmpath(pathname)

