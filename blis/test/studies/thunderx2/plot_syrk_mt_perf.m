
axes2 = subplot(4, 4, 2);
hold(axes2,'on');

axes6 = subplot(4, 4, 6);
hold(axes6,'on');

axes10 = subplot(4, 4, 10);
hold(axes10,'on');

axes14 = subplot(4, 4, 14);
hold(axes14,'on');

addpath(pathname_blis)

if(plot_s)
axes(axes2);
output_mt_ssyrk_asm_blis
plot(data_mt_ssyrk_asm_blis(:,1), data_mt_ssyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_d)
axes(axes6);
output_mt_dsyrk_asm_blis
plot(data_mt_dsyrk_asm_blis(:,1), data_mt_dsyrk_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_c)
axes(axes10);
output_mt_csyrk_1m_blis
plot(data_mt_csyrk_1m_blis(:,1), data_mt_csyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_z)
axes(axes14);
output_mt_zsyrk_1m_blis
plot(data_mt_zsyrk_1m_blis(:,1), data_mt_zsyrk_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end
clear *syrk*
rmpath(pathname_blis)


% OpenBLAS
addpath(pathname_openblas)

if(plot_s)
axes(axes2);
output_mt_ssyrk_openblas
plot(data_mt_ssyrk_openblas(:,1), data_mt_ssyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_d)
axes(axes6);
output_mt_dsyrk_openblas
plot(data_mt_dsyrk_openblas(:,1), data_mt_dsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_c)
axes(axes10);
output_mt_csyrk_openblas
plot(data_mt_csyrk_openblas(:,1), data_mt_csyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_z)
axes(axes14);
output_mt_zsyrk_openblas
plot(data_mt_zsyrk_openblas(:,1), data_mt_zsyrk_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

clear *syrk*
rmpath(pathname_openblas)


% ARMPL
addpath(pathname_armpl)
if(plot_s)
axes(axes2);
output_mt_ssyrk_armpl
plot(data_mt_ssyrk_armpl(:,1), data_mt_ssyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_d)
axes(axes6);
output_mt_dsyrk_armpl
plot(data_mt_dsyrk_armpl(:,1), data_mt_dsyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_c)
axes(axes10);
output_mt_csyrk_armpl
plot(data_mt_csyrk_armpl(:,1), data_mt_csyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_z)
axes(axes14);
output_mt_zsyrk_armpl
plot(data_mt_zsyrk_armpl(:,1), data_mt_zsyrk_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

clear *syrk*
rmpath(pathname_armpl)

axes(axes2);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes2,'on');
set(axes2,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

% DSYRK multi threaded

axes(axes6);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes6,'on');
set(axes6,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CSYRK multi threaded

axes(axes10);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes10,'on');
set(axes10,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

% ZSYRK multi threaded

axes(axes14);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZSYRK (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes14,'on');
set(axes14,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
% legend({'BLIS', 'BLIS (AVX2)','OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

