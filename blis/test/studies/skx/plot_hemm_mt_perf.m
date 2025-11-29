addpath(pathname)


output_mt_shemm_asm_blis
output_mt_dhemm_asm_blis
output_mt_chemm_1m_blis
output_mt_zhemm_1m_blis

output_mt_shemm_openblas
output_mt_dhemm_openblas
output_mt_chemm_openblas
output_mt_zhemm_openblas

output_mt_shemm_mkl
output_mt_dhemm_mkl
output_mt_chemm_mkl
output_mt_zhemm_mkl


% SSYMM multi threaded

axes1 = subplot(4, 4, 3);
hold(axes1,'on');
plot(data_mt_shemm_asm_blis(:,1), data_mt_shemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_shemm_openblas(:,1), data_mt_shemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_shemm_mkl(:,1), data_mt_shemm_mkl(:,3), '--',  'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% DSYMM multi threaded

axes1 = subplot(4, 4, 7);
hold(axes1,'on');
plot(data_mt_dhemm_asm_blis(:,1), data_mt_dhemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_dhemm_openblas(:,1), data_mt_dhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_dhemm_mkl(:,1), data_mt_dhemm_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'bemt');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CHEMM multi threaded

axes1 = subplot(4, 4, 11);
hold(axes1,'on');
plot(data_mt_chemm_1m_blis(:,1), data_mt_chemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_chemm_openblas(:,1), data_mt_chemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_chemm_mkl(:,1), data_mt_chemm_mkl(:,3), '--',  'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CHEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% ZHEMM multi threaded

axes1 = subplot(4, 4, 15);
hold(axes1,'on');
plot(data_mt_zhemm_1m_blis(:,1), data_mt_zhemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_zhemm_openblas(:,1), data_mt_zhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_zhemm_mkl(:,1), data_mt_zhemm_mkl(:,3),'--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZHEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

clear *hemm*
rmpath(pathname)
