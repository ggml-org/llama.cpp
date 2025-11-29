axes3 = subplot(4, 4, 3);
hold(axes3,'on');

axes7 = subplot(4, 4, 7);
hold(axes7,'on');

axes11 = subplot(4, 4, 11);
hold(axes11,'on');

axes15 = subplot(4, 4, 15);
hold(axes15,'on');

addpath(pathname_blis)

if(plot_s)
    axes(axes3);
    output_mt_shemm_asm_blis
    plot(data_mt_shemm_asm_blis(:,1), data_mt_shemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_d)
    axes(axes7);
    output_mt_dhemm_asm_blis
    plot(data_mt_dhemm_asm_blis(:,1), data_mt_dhemm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_c)
    axes(axes11);
    output_mt_chemm_1m_blis
    plot(data_mt_chemm_1m_blis(:,1), data_mt_chemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_z)
    axes(axes15);
    output_mt_zhemm_1m_blis
    plot(data_mt_zhemm_1m_blis(:,1), data_mt_zhemm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

clear *hemm*
rmpath(pathname_blis)

addpath(pathname_openblas)

if(plot_s)
    axes(axes3);
    output_mt_shemm_openblas
    plot(data_mt_shemm_openblas(:,1), data_mt_shemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_d)
    axes(axes7);
    output_mt_dhemm_openblas
    plot(data_mt_dhemm_openblas(:,1), data_mt_dhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_c)
    axes(axes11);
    output_mt_chemm_openblas
    plot(data_mt_chemm_openblas(:,1), data_mt_chemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_z)
    axes(axes15);
    output_mt_zhemm_openblas
    plot(data_mt_zhemm_openblas(:,1), data_mt_zhemm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

clear *hemm*
rmpath(pathname_openblas)

addpath(pathname_armpl);

if(plot_s)
    axes(axes3);
    output_mt_shemm_armpl
    plot(data_mt_shemm_armpl(:,1), data_mt_shemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_d)
    axes(axes7);
    output_mt_dhemm_armpl
    plot(data_mt_dhemm_armpl(:,1), data_mt_dhemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_c)
    axes(axes11);
    output_mt_chemm_armpl
    plot(data_mt_chemm_armpl(:,1), data_mt_chemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_z)
    axes(axes15);
    output_mt_zhemm_armpl
    plot(data_mt_zhemm_armpl(:,1), data_mt_zhemm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

clear *hemm*
rmpath(pathname_armpl)

% SSYMM multi threaded

axes(axes3);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SSYMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes3,'on');
set(axes3,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


axes(axes7);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DSYMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes7,'on');
set(axes7,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

% CHEMM multi threaded

axes(axes11);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CHEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes11,'on');
set(axes11,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )


% ZHEMM multi threaded
axes(axes15);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZHEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes15,'on');
set(axes15,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
% legend({'BLIS', 'BLIS (AVX2)', 'OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )



