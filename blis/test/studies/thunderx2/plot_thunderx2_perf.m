plot_st = 1;
plot_1s = 1;
plot_2s = 1;


plot_s = 1;
plot_d = 1;
plot_c = 1;
plot_z = 1;

plot_armpl = 1;

fontsize = 6;

freq = 2;
sflopspercycle = 16;
dflopspercycle = 8;

speak = sflopspercycle*freq;
dpeak = dflopspercycle*freq;

xmax_mt = 5000;

if(plot_st)
    numcores = 1;
    
    fig1 = figure(1);
    clf(fig1)
    %
    pathname = './20180824/';
    pathname_armpl = './20180829/';
    plot_gemm_st_perf
    plot_syrk_st_perf
    plot_hemm_st_perf
    plot_trmm_st_perf
    
    %fig1.PaperPositionMode = 'auto';
    orient(fig1,'landscape')
    set(fig1,'PaperUnits','normalized');
    set(fig1,'PaperPosition', [0 0 1 1]);
    print(fig1, 'thunderx2-st-20180829', '-dpdf')
    
    clear pathname pathname_armpl
    
    
end

if (plot_1s)
    fig1 = figure(2);
    clf;
    
    
    numcores = 28;
    pathname_blis = './20180830/1socket';
    pathname_armpl = './20180830/1socket';
    pathname_openblas = './20180830/1socket';
    
    %JC = 2, IC = 14
    
    plot_gemm_mt_perf
    plot_syrk_mt_perf
    plot_hemm_mt_perf
    plot_trmm_mt_perf
    
    %fig1.PaperPositionMode = 'auto';
    orient(fig1,'landscape')
    set(fig1,'PaperUnits','normalized');
    set(fig1,'PaperPosition', [0 0 1 1]);
    print(fig1, 'thunderx2-mt-28cores-20180830', '-dpdf')
end

if(plot_2s)
    
    numcores = 56;
    
    %JC = 4, IC = 14
    
    fig1 = figure(3);
    clf;
    
    plot_gemm = 1;
    plot_syrk = 1;
    plot_hemm = 1;
    plot_trmm = 1;
    
    plot_s = 1;
    plot_d = 1;
    plot_c = 1;
    plot_z = 1;
    pathname_blis = './20180830/2sockets';
    pathname_openblas = './20180830/2sockets';
    pathname_armpl = './20180830/2sockets';
    
    
    if(plot_gemm)
        plot_gemm_mt_perf
    end
    if(plot_syrk)
        plot_syrk_mt_perf
    end
    if(plot_hemm)
        plot_hemm_mt_perf
    end
    if(plot_trmm)
        plot_trmm_mt_perf
    end
    
    %fig1.PaperPositionMode = 'auto';
    orient(fig1,'landscape')
    set(fig1,'PaperUnits','normalized');
    set(fig1,'PaperPosition', [0 0 1 1]);
    print(fig1, 'thunderx2-mt-56cores-20180830', '-dpdf')
end