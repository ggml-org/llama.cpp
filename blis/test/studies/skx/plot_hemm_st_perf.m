addpath(pathname)


output_st_shemm_asm_blis
output_st_dhemm_asm_blis
output_st_chemm_1m_blis
output_st_zhemm_1m_blis

output_st_shemm_openblas
output_st_dhemm_openblas
output_st_chemm_openblas
output_st_zhemm_openblas

output_st_shemm_mkl
output_st_dhemm_mkl
output_st_chemm_mkl
output_st_zhemm_mkl


% SSYMM Single threaded

axes1 = subplot(4, 4, 3);
hold(axes1,'on');
plot(data_st_shemm_asm_blis(:,1), data_st_shemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_shemm_openblas(:,1), data_st_shemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_shemm_mkl(:,1), data_st_shemm_mkl(:,3), '--',  'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% DSYMM Single threaded

axes1 = subplot(4, 4, 7);
hold(axes1,'on');
plot(data_st_dhemm_asm_blis(:,1), data_st_dhemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_dhemm_openblas(:,1), data_st_dhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_dhemm_mkl(:,1), data_st_dhemm_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

% CHEMM Single threaded

axes1 = subplot(4, 4, 11);
hold(axes1,'on');
plot(data_st_chemm_1m_blis(:,1), data_st_chemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_chemm_openblas(:,1), data_st_chemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_chemm_mkl(:,1), data_st_chemm_mkl(:,3), '--',  'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CHEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% ZHEMM Single threaded

axes1 = subplot(4, 4, 15);
hold(axes1,'on');
plot(data_st_zhemm_1m_blis(:,1), data_st_zhemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_zhemm_openblas(:,1), data_st_zhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_st_zhemm_mkl(:,1), data_st_zhemm_mkl(:,3),'--', 'LineWidth', 1.25,'Color', [1 0 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZHEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
% legend({'BLIS', 'BLIS (AVX2)', 'OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

clear *hemm*
rmpath(pathname)
