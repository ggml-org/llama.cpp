addpath(pathname)
output_mt_ssyrk_asm_blis
output_mt_dsyrk_asm_blis
output_mt_csyrk_1m_blis
output_mt_zsyrk_1m_blis

output_mt_ssyrk_openblas
output_mt_dsyrk_openblas
output_mt_csyrk_openblas
output_mt_zsyrk_openblas

output_mt_ssyrk_mkl
output_mt_dsyrk_mkl
output_mt_csyrk_mkl
output_mt_zsyrk_mkl

plot_lower=0;

if(plot_lower)
    output_mt_ssyrk_asm_blis
    output_mt_dsyrk_asm_blis
    output_mt_csyrk_1m_blis
    output_mt_zsyrk_1m_blis
    
    output_mt_ssyrk_openblas
    output_mt_dsyrk_openblas
    output_mt_csyrk_openblas
    output_mt_zsyrk_openblas
    
    output_mt_ssyrk_mkl
    output_mt_dsyrk_mkl
    output_mt_csyrk_mkl
    output_mt_zsyrk_mkl
end


% SSYRK multi threaded

axes1 = subplot(4, 4, 2);
hold(axes1,'on');
plot(data_mt_ssyrk_asm_blis(:,1), data_mt_ssyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_ssyrk_openblas(:,1), data_mt_ssyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_ssyrk_mkl(:,1), data_mt_ssyrk_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);

if(plot_lower)
    
    plot(data_mt_ssyrk_l_asm_blis(:,1), data_mt_ssyrk_l_asm_blis(:,3), '-.','LineWidth', 1.25,'Color', [0 0 1]);
    plot(data_mt_ssyrk_l_openblas(:,1), data_mt_ssyrk_l_openblas(:,3), '-.', 'LineWidth', 1.25,'Color', [0 1 0]);
    plot(data_mt_ssyrk_l_mkl(:,1), data_mt_ssyrk_l_mkl(:,3), '-.', 'LineWidth', 1.25,'Color', [1 0 0]);
end

ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

% DSYRK multi threaded

axes1 = subplot(4, 4, 6);
hold(axes1,'on');
plot(data_mt_dsyrk_asm_blis(:,1), data_mt_dsyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_dsyrk_openblas(:,1), data_mt_dsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_dsyrk_mkl(:,1), data_mt_dsyrk_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);

if(plot_lower)
    
    plot(data_mt_dsyrk_l_asm_blis(:,1), data_mt_dsyrk_l_asm_blis(:,3), '-.', 'LineWidth', 1.25,'Color', [0 0 1]);
    plot(data_mt_dsyrk_l_openblas(:,1), data_mt_dsyrk_l_openblas(:,3), '-.', 'LineWidth', 1.25,'Color', [0 1 0]);
    plot(data_mt_dsyrk_l_mkl(:,1), data_mt_dsyrk_l_mkl(:,3), '-.', 'LineWidth', 1.25,'Color', [1 0 0]);
end

ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'bemt');

v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CSYRK multi threaded

axes1 = subplot(4, 4, 10);
hold(axes1,'on');
plot(data_mt_csyrk_1m_blis(:,1), data_mt_csyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_csyrk_openblas(:,1), data_mt_csyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_csyrk_mkl(:,1), data_mt_csyrk_mkl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 0]);

if(plot_lower)
    plot(data_mt_csyrk_l_1m_blis(:,1), data_mt_csyrk_l_1m_blis(:,3),'-.', 'LineWidth', 1.25,'Color', [0 0 1]);
    plot(data_mt_csyrk_l_openblas(:,1), data_mt_csyrk_l_openblas(:,3), '-.', 'LineWidth', 1.25,'Color', [0 1 0]);
    plot(data_mt_csyrk_l_mkl(:,1), data_mt_csyrk_l_mkl(:,3), '-.', 'LineWidth', 1.25,'Color', [1 0 0]);
end

ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

% ZSYRK multi threaded

axes1 = subplot(4, 4, 14);
hold(axes1,'on');
plot(data_mt_zsyrk_1m_blis(:,1), data_mt_zsyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
plot(data_mt_zsyrk_openblas(:,1), data_mt_zsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
plot(data_mt_zsyrk_mkl(:,1), data_mt_zsyrk_mkl(:,3), '--',  'LineWidth', 1.25,'Color', [1 0 0]);

if(plot_lower)
    plot(data_mt_zsyrk_l_1m_blis(:,1), data_mt_zsyrk_l_1m_blis(:,3), '-.', 'LineWidth', 1.25,'Color', [0 0 1]);
    plot(data_mt_zsyrk_l_openblas(:,1), data_mt_zsyrk_l_openblas(:,3), '-.', 'LineWidth', 1.25,'Color', [0 1 0]);
    plot(data_mt_zsyrk_l_mkl(:,1), data_mt_zsyrk_l_mkl(:,3), '-.', 'LineWidth', 1.25,'Color', [1 0 0]);
end
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
legend({'BLIS (Upper)', 'OpenBLAS (Upper)', 'ARMPL (Upper)', 'BLIS (Lower)', 'OpenBLAS (Lower) ', 'ARMPL (Lower)'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');

v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

clear *syrk*
rmpath(pathname)

