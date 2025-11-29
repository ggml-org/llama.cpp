addpath(pathname)

output_st_ssyrk_asm_blis
output_st_dsyrk_asm_blis
output_st_csyrk_1m_blis
output_st_zsyrk_1m_blis

output_st_ssyrk_openblas
output_st_dsyrk_openblas
output_st_csyrk_openblas
output_st_zsyrk_openblas


plot_lower = 0;


% SSYRK Single threaded

axes1 = subplot(4, 4, 2);
hold(axes1,'on');
plot(data_st_ssyrk_asm_blis(:,1), data_st_ssyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_ssyrk_openblas(:,1), data_st_ssyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYRK (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )

% DSYRK single threaded

axes1 = subplot(4, 4, 6);
hold(axes1,'on');
plot(data_st_dsyrk_asm_blis(:,1), data_st_dsyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_dsyrk_openblas(:,1), data_st_dsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYRK (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');

v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

% CSYRK single threaded

axes1 = subplot(4, 4, 10);
hold(axes1,'on');
plot(data_st_csyrk_1m_blis(:,1), data_st_csyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_csyrk_openblas(:,1), data_st_csyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CSYRK (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )

% ZSYRK single threaded

axes1 = subplot(4, 4, 14);
hold(axes1,'on');
plot(data_st_zsyrk_1m_blis(:,1), data_st_zsyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_zsyrk_openblas(:,1), data_st_zsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZSYRK (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
% legend({'BLIS', 'BLIS (AVX2)','OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');

v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

clear *syrk*
rmpath(pathname)

addpath(pathname_armpl)

output_st_ssyrk_armpl
output_st_dsyrk_armpl
output_st_csyrk_armpl
output_st_zsyrk_armpl

% Ssyrk Single threaded

subplot(4, 4, 2);
plot(data_st_ssyrk_armpl(:,1), data_st_ssyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 6);
plot(data_st_dsyrk_armpl(:,1), data_st_dsyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 10);
plot(data_st_csyrk_armpl(:,1), data_st_csyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 14);
plot(data_st_zsyrk_armpl(:,1), data_st_zsyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

clear *syrk*
rmpath(pathname_armpl)

