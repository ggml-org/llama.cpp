axes1 = subplot(4, 4, 1);
hold(axes1,'on');

axes2 = subplot(4, 4, 5);
hold(axes2,'on');

axes3 = subplot(4, 4, 9);
hold(axes3,'on');

axes4 = subplot(4, 4, 13);
hold(axes4,'on');

addpath(pathname_blis)


if(plot_s)
    % SGEMM multi threaded
    axes(axes1);
    output_mt_sgemm_asm_blis
    plot(data_mt_sgemm_asm_blis(:,1), data_mt_sgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
end
% DGEMM multi threaded

if(plot_d)
    
    axes(axes2);
    output_mt_dgemm_asm_blis
    plot(data_mt_dgemm_asm_blis(:,1), data_mt_dgemm_asm_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
end

% CGEMM multi threaded

if(plot_c)
    axes(axes3);
    output_mt_cgemm_1m_blis
    plot(data_mt_cgemm_1m_blis(:,1), data_mt_cgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
end

% ZGEMM multi threaded

if(plot_z)
    axes(axes4);
    output_mt_zgemm_1m_blis
    plot(data_mt_zgemm_1m_blis(:,1), data_mt_zgemm_1m_blis(:,4), 'LineWidth', 1.25,'Color', [0 0 1]);
end

clear *gemm*
rmpath(pathname_blis)


% OpenBLAS

addpath(pathname_openblas)

if(plot_s)
    axes(axes1);
    output_mt_sgemm_openblas
    plot(data_mt_sgemm_openblas(:,1), data_mt_sgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_d)
    axes(axes2);
    output_mt_dgemm_openblas
    plot(data_mt_dgemm_openblas(:,1), data_mt_dgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_c)
    axes(axes3);
    output_mt_cgemm_openblas
    plot(data_mt_cgemm_openblas(:,1), data_mt_cgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_z)
    axes(axes4);
    output_mt_zgemm_openblas
    plot(data_mt_zgemm_openblas(:,1), data_mt_zgemm_openblas(:,4), 'LineWidth', 1.25,'Color', [0 1 0]);
end

clear *gemm*
rmpath(pathname_openblas)

% ARMPL

addpath(pathname_armpl)

if(plot_s)
    axes(axes1);
    output_mt_sgemm_armpl
    plot(data_mt_sgemm_armpl(:,1), data_mt_sgemm_armpl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end


if(plot_d)
    axes(axes2);
    output_mt_dgemm_armpl
    plot(data_mt_dgemm_armpl(:,1), data_mt_dgemm_armpl(:,4), '--',  'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_c)
    axes(axes3);
    output_mt_cgemm_armpl
    plot(data_mt_cgemm_armpl(:,1), data_mt_cgemm_armpl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

if(plot_z)
    axes(axes4);
    output_mt_zgemm_armpl
    plot(data_mt_zgemm_armpl(:,1), data_mt_zgemm_armpl(:,4), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
end

clear *gemm*
rmpath(pathname_armpl)

axes(axes1);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('SGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes1,'on');
set(axes1,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores] )

axes(axes2);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes2,'on');
set(axes2,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

axes(axes3);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes3,'on');
set(axes3,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

axes(axes4);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZGEMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes4,'on');
set(axes4,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'MKL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )



