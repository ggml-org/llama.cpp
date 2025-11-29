addpath(pathname)


output_st_shemm_asm_blis
output_st_dhemm_asm_blis
output_st_chemm_1m_blis
output_st_zhemm_1m_blis

output_st_shemm_openblas
output_st_dhemm_openblas
output_st_chemm_openblas
output_st_zhemm_openblas

% SSYMM Single threaded

axes1 = subplot(4, 4, 3);
hold(axes1,'on');
plot(data_st_shemm_asm_blis(:,1), data_st_shemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_shemm_openblas(:,1), data_st_shemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
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
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZHEMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

clear *hemm*
rmpath(pathname)

addpath(pathname_armpl)

output_st_shemm_armpl
output_st_dhemm_armpl
output_st_chemm_armpl
output_st_zhemm_armpl

% Shemm Single threaded

subplot(4, 4, 3);
plot(data_st_shemm_armpl(:,1), data_st_shemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 7);
plot(data_st_dhemm_armpl(:,1), data_st_dhemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 11);
plot(data_st_chemm_armpl(:,1), data_st_chemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 15);
plot(data_st_zhemm_armpl(:,1), data_st_zhemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

clear *hemm*
rmpath(pathname_armpl)
