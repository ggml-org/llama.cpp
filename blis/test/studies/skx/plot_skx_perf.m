fontsize = 6;
numcores = 4;
freq = 3.5;
sflopspercycle = 64;
dflopspercycle = 32;

speak = sflopspercycle*freq;
dpeak = dflopspercycle*freq;

xmax_mt = 5000;

fig1 = figure(1);
clf(fig1)
% 
pathname = './20180711/';
plot_gemm_st_perf
plot_syrk_st_perf
plot_hemm_st_perf
plot_trmm_st_perf
 
fig1.PaperPositionMode = 'auto';
orient(fig1,'landscape')
print(fig1, 'skx-st', '-dpdf','-fillpage')

% fig1 = figure(2);
% clf;
% 
% plot_gemm_mt_perf
% plot_syrk_mt_perf
% plot_hemm_mt_perf
% plot_trmm_mt_perf
% 
% fig1.PaperPositionMode = 'auto';
% orient(fig1,'landscape')
% print(fig1, 'A57-mt', '-dpdf','-fillpage')
