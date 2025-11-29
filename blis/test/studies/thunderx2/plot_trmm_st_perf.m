addpath(pathname)

output_st_strmm_asm_blis
output_st_dtrmm_asm_blis
output_st_ctrmm_1m_blis
output_st_ztrmm_1m_blis

output_st_strmm_openblas
output_st_dtrmm_openblas
output_st_ctrmm_openblas
output_st_ztrmm_openblas


% STRMM Single threaded

axes1 = subplot(4, 4, 4);
hold(axes1,'on');
plot(data_st_strmm_asm_blis(:,1), data_st_strmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_strmm_openblas(:,1), data_st_strmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('STRMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% DTRMM Single threaded

axes1 = subplot(4, 4, 8);
hold(axes1,'on');
plot(data_st_dtrmm_asm_blis(:,1), data_st_dtrmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_dtrmm_openblas(:,1), data_st_dtrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DTRMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

% CTRMM Single threaded

axes1 = subplot(4, 4, 12);
hold(axes1,'on');
plot(data_st_ctrmm_1m_blis(:,1), data_st_ctrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_ctrmm_openblas(:,1), data_st_ctrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CTRMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 speak ] )


% ZTRMM Single threaded

axes1 = subplot(4, 4, 16);
hold(axes1,'on');
plot(data_st_ztrmm_1m_blis(:,1), data_st_ztrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_st_ztrmm_openblas(:,1), data_st_ztrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZTRMM (single-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 v(2) 0 dpeak ] )

clear *trmm*
rmpath(pathname)


addpath(pathname_armpl)

output_st_strmm_armpl
output_st_dtrmm_armpl
output_st_ctrmm_armpl
output_st_ztrmm_armpl

% Strmm Single threaded

subplot(4, 4, 4);
plot(data_st_strmm_armpl(:,1), data_st_strmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 8);
plot(data_st_dtrmm_armpl(:,1), data_st_dtrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 12);
plot(data_st_ctrmm_armpl(:,1), data_st_ctrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);

subplot(4, 4, 16);
plot(data_st_ztrmm_armpl(:,1), data_st_ztrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
legend({'BLIS','OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');


clear *trmm*
rmpath(pathname_armpl)
