axes4 = subplot(4, 4, 4);
hold(axes4,'on');

axes8 = subplot(4, 4, 8);
hold(axes8,'on');

axes12 = subplot(4, 4, 12);
hold(axes12,'on');

axes16 = subplot(4, 4, 16);
hold(axes16,'on');

addpath(pathname_blis)

if(plot_s)
    axes(axes4);
    output_mt_strmm_asm_blis
    plot(data_mt_strmm_asm_blis(:,1), data_mt_strmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_d)
    axes(axes8);
    output_mt_dtrmm_asm_blis
    plot(data_mt_dtrmm_asm_blis(:,1), data_mt_dtrmm_asm_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_c)
    axes(axes12);
    output_mt_ctrmm_1m_blis
    plot(data_mt_ctrmm_1m_blis(:,1), data_mt_ctrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

if(plot_z)
    axes(axes16);
    output_mt_ztrmm_1m_blis
    plot(data_mt_ztrmm_1m_blis(:,1), data_mt_ztrmm_1m_blis(:,3), 'LineWidth', 1.25,'Color', [0 0 1]);
end

clear *trmm*
rmpath(pathname_blis)

addpath(pathname_openblas)

if(plot_s)
    axes(axes4);
    output_mt_strmm_openblas
    plot(data_mt_strmm_openblas(:,1), data_mt_strmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_d)
    axes(axes8);
    output_mt_dtrmm_openblas
    plot(data_mt_dtrmm_openblas(:,1), data_mt_dtrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_c)
    axes(axes12);
    output_mt_ctrmm_openblas
    plot(data_mt_ctrmm_openblas(:,1), data_mt_ctrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end

if(plot_z)
    axes(axes16);
    output_mt_ztrmm_openblas
    plot(data_mt_ztrmm_openblas(:,1), data_mt_ztrmm_openblas(:,3), 'LineWidth', 1.25,'Color', [0 1 0]);
end
clear *trmm*
rmpath(pathname_openblas)





if(plot_armpl)
    
    addpath(pathname_armpl)
    
    if(plot_s)
        axes(axes4);
        output_mt_strmm_armpl
        plot(data_mt_strmm_armpl(:,1), data_mt_strmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
    end
    
    if(plot_d)
        axes(axes8);
        output_mt_dtrmm_armpl
        plot(data_mt_dtrmm_armpl(:,1), data_mt_dtrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
    end
    
    if(plot_c)
        axes(axes12);
        output_mt_ctrmm_armpl
        plot(data_mt_ctrmm_armpl(:,1), data_mt_ctrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
    end
    
    if(plot_z)
        axes(axes16);
        output_mt_ztrmm_armpl
        plot(data_mt_ztrmm_armpl(:,1), data_mt_ztrmm_armpl(:,3), '--', 'LineWidth', 1.25,'Color', [1 0 1]);
    end
    
    clear *trmm*
    rmpath(pathname_armpl)
end

axes(axes4);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('STRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes4,'on');
set(axes4,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

axes(axes8);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('DTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes8,'on');
set(axes8,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%legend({'BLIS', 'OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'best');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

axes(axes12);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
%xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('CTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes12,'on');
set(axes12,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 speak*numcores ] )

axes(axes16);
ylabel( 'GFLOPS', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
xlabel( 'matrix dimension m=n=k', 'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue' );
title('ZTRMM (multi-threaded)','FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
box(axes16,'on');
set(axes16,'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue');
legend({'BLIS','OpenBLAS'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');
v = axis;     % extract the current ranges
axis( [ 0 xmax_mt 0 dpeak*numcores ] )

legend({'BLIS','OpenBLAS', 'ARMPL'},'FontSize', fontsize, 'FontWeight', 'bold', 'FontName', 'Helvetica Neue', 'Location', 'South');

