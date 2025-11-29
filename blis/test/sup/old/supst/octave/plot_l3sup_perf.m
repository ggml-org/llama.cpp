function r_val = plot_l3sup_perf( opname, ...
                                  data_blissup, ...
                                  data_blislpab, ...
                                  data_eigen, ...
                                  data_open, ...
                                  data_bfeo, ...
                                  data_xsmm, ...
                                  data_vend, vend_str, ...
                                  nth, ...
                                  rows, cols, ...
                                  cfreq, ...
                                  dfps, ...
                                  theid, impl )

%if ... %mod(theid-1,cols) == 2 || ...
%   ... %mod(theid-1,cols) == 3 || ...
%   ... %mod(theid-1,cols) == 4 || ...
%   0 == 1 ... %theid >= 19
%	show_plot = 0;
%else
	show_plot = 1;
%end

%legend_plot_id = 11;
legend_plot_id = 2*cols + 1*5;

if 1
	ax1 = subplot( rows, cols, theid );
	hold( ax1, 'on' );
end

% Set line properties.
color_blissup  = 'k'; lines_blissup  = '-';  markr_blissup  = '';
color_blislpab = 'k'; lines_blislpab = ':';  markr_blislpab = '';
color_eigen    = 'm'; lines_eigen    = '-.'; markr_eigen    = 'o';
color_open     = 'r'; lines_open     = '--'; markr_open     = 'o';
color_bfeo     = 'c'; lines_bfeo     = '-';  markr_bfeo     = 'o';
color_xsmm     = 'g'; lines_xsmm     = '-';  markr_xsmm     = 'o';
color_vend     = 'b'; lines_vend     = '-.'; markr_vend     = '.';

% Compute the peak performance in terms of the number of double flops
% executable per cycle and the clock rate.
if opname(1) == 's' || opname(1) == 'c'
	flopspercycle = dfps * 2;
else
	flopspercycle = dfps;
end
max_perf_core = (flopspercycle * cfreq) * 1;

% Escape underscores in the title.
title_opname = strrep( opname, '_', '\_' );

% Print the title to a string.
titlename = '%s';
titlename = sprintf( titlename, title_opname );

% Set the legend strings.
blissup_legend  = sprintf( 'BLIS sup' );
blislpab_legend = sprintf( 'BLIS conv' );
eigen_legend    = sprintf( 'Eigen' );
open_legend     = sprintf( 'OpenBLAS' );
bfeo_legend     = sprintf( 'BLASFEO' );
xsmm_legend     = sprintf( 'libxsmm' );
%vend_legend     = sprintf( 'MKL' );
%vend_legend     = sprintf( 'ARMPL' );
vend_legend     = vend_str;

% Set axes range values.
y_scale = 1.00;
x_begin = 0;
%x_end is set below.
y_begin = 0;
y_end   = max_perf_core * y_scale;

% Set axes names.
if nth == 1
	yaxisname = 'GFLOPS';
else
	yaxisname = 'GFLOPS/core';
end


%flopscol = 4;
flopscol = size( data_blissup, 2 );
msize = 5;
if 1
	fontsize = 12;
else
	fontsize = 16;
end
linesize = 0.5;
legend_loc = 'southeast';

% --------------------------------------------------------------------

% Automatically detect a column with the increasing problem size.
% Then set the maximum x-axis value.
for psize_col = 1:3
	if data_blissup( 1, psize_col ) ~= data_blissup( 2, psize_col )
		break;
	end
end
x_axis( :, 1 ) = data_blissup( :, psize_col );

% Compute the number of data points we have in the x-axis. Note that
% we only use half the data points for the m = n = k column of graphs.
if mod(theid-1,cols) == 6
	np = size( data_blissup, 1 ) / 2;
else
	np = size( data_blissup, 1 );
end

has_xsmm = 1;
if data_xsmm( 1, flopscol ) == 0.0
	has_xsmm = 0;
end

% Grab the last x-axis value.
x_end = data_blissup( np, psize_col );

%data_peak( 1, 1:2 ) = [     0 max_perf_core ];
%data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

if show_plot == 1
blissup_ln  = line( x_axis( 1:np, 1 ), data_blissup( 1:np, flopscol ) / nth, ...
                    'Color',color_blissup, 'LineStyle',lines_blissup, ...
                    'LineWidth',linesize );
blislpab_ln = line( x_axis( 1:np, 1 ), data_blislpab( 1:np, flopscol ) / nth, ...
                    'Color',color_blislpab, 'LineStyle',lines_blislpab, ...
                    'LineWidth',linesize );
eigen_ln    = line( x_axis( 1:np, 1 ), data_eigen( 1:np, flopscol ) / nth, ...
                    'Color',color_eigen, 'LineStyle',lines_eigen, ...
                    'LineWidth',linesize );
open_ln     = line( x_axis( 1:np, 1 ), data_open( 1:np, flopscol ) / nth, ...
                    'Color',color_open, 'LineStyle',lines_open, ...
                    'LineWidth',linesize );
bfeo_ln     = line( x_axis( 1:np, 1 ), data_bfeo( 1:np, flopscol ) / nth, ...
                    'Color',color_bfeo, 'LineStyle',lines_bfeo, ...
                    'LineWidth',linesize );
if has_xsmm == 1
xsmm_ln     = line( x_axis( 1:np, 1 ), data_xsmm( 1:np, flopscol ) / nth, ...
                    'Color',color_xsmm, 'LineStyle',lines_xsmm, ...
                    'LineWidth',linesize );
else
xsmm_ln     = line( nan, nan, ...
                    'Color',color_xsmm, 'LineStyle',lines_xsmm, ...
                    'LineWidth',linesize );
end
vend_ln     = line( x_axis( 1:np, 1 ), data_vend( 1:np, flopscol ) / nth, ...
                    'Color',color_vend, 'LineStyle',lines_vend, ...
                    'LineWidth',linesize );
elseif theid == legend_plot_id
blissup_ln  = line( nan, nan, ...
                    'Color',color_blissup, 'LineStyle',lines_blissup, ...
                    'LineWidth',linesize );
blislpab_ln = line( nan, nan, ...
                    'Color',color_blislpab, 'LineStyle',lines_blislpab, ...
                    'LineWidth',linesize );
eigen_ln    = line( nan, nan, ...
                    'Color',color_eigen, 'LineStyle',lines_eigen, ...
                    'LineWidth',linesize );
open_ln     = line( nan, nan, ...
                    'Color',color_open, 'LineStyle',lines_open, ...
                    'LineWidth',linesize );
bfeo_ln     = line( nan, nan, ...
                    'Color',color_bfeo, 'LineStyle',lines_bfeo, ...
                    'LineWidth',linesize );
xsmm_ln     = line( nan, nan, ...
                    'Color',color_xsmm, 'LineStyle',lines_xsmm, ...
                    'LineWidth',linesize );
vend_ln     = line( nan, nan, ...
                    'Color',color_vend, 'LineStyle',lines_vend, ...
                    'LineWidth',linesize );
end


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if     6000 <= x_end && x_end < 10000
	x_tick2 = x_end - 2000;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
elseif 4000 <= x_end && x_end < 6000
	x_tick2 = x_end - 1000;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
elseif 2000 <= x_end && x_end < 3000
	x_tick2 = x_end - 400;
	x_tick1 = x_tick2/2;
	xticks( ax1, [ x_tick1 x_tick2 ] );
elseif 500 <= x_end && x_end < 1000
	x_tick3 = x_end*(3/4);
	x_tick2 = x_end*(2/4);
	x_tick1 = x_end*(1/4);
	xticks( ax1, [ x_tick1 x_tick2 x_tick3 ] );
end

if show_plot == 1 || theid == legend_plot_id
	if nth == 1 && theid == legend_plot_id
		if has_xsmm == 1
			leg = legend( ...
			[ ...
			  blissup_ln ...
			  blislpab_ln ...
			  eigen_ln ...
			  open_ln ...
			  bfeo_ln ...
			  xsmm_ln ...
			  vend_ln ...
			], ...
			blissup_legend, ...
			blislpab_legend, ...
			eigen_legend, ...
			open_legend, ...
			bfeo_legend, ...
			xsmm_legend, ...
			vend_legend, ...
			'Location', legend_loc );
			set( leg,'Box','off' );
			set( leg,'Color','none' );
			set( leg,'Units','inches' );
			if impl == 'octave'
				set( leg,'FontSize',fontsize );
				set( leg,'Position',[15.40 4.75 1.9 1.20] ); % (1,4tl)
			else
				set( leg,'FontSize',fontsize-3 );
				set( leg,'Position',[18.20 10.20 1.15 0.7 ] ); % (1,4tl)
			end
		else
			leg = legend( ...
			[ ...
			  blissup_ln ...
			  blislpab_ln ...
			  eigen_ln ...
			  open_ln ...
			  bfeo_ln ...
			  vend_ln ...
			], ...
			blissup_legend, ...
			blislpab_legend, ...
			eigen_legend, ...
			open_legend, ...
			bfeo_legend, ...
			vend_legend, ...
			'Location', legend_loc );
			set( leg,'Box','off' );
			set( leg,'Color','none' );
			set( leg,'Units','inches' );
			if impl == 'octave'
				set( leg,'FontSize',fontsize );
				set( leg,'Position',[15.40 7.65 1.9 1.10] ); % (1,4tl)
			else
				set( leg,'FontSize',fontsize-1 );
				set( leg,'Position',[18.24 10.15 1.15 0.7] ); % (1,4tl)
			end
		end
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'Units','inches' );
		%                    xpos ypos
		%set( leg,'Position',[11.32 6.36 1.15 0.7 ] ); % (1,4tl)
	elseif nth > 1 && theid == legend_plot_id
	end
end

set( ax1,'FontSize',fontsize );
set( ax1,'TitleFontSizeMultiplier',1.0 ); % default is 1.1.
box( ax1, 'on' );

titl = title( titlename );
set( titl, 'FontWeight', 'normal' ); % default font style is now 'bold'.

% The default is to align the plot title across whole figure, not the box.
% This is a hack to nudge the title back to the center of the box.
if impl == 'octave'
	tpos = get( titl, 'Position' );
	% For some reason, the titles in the graphs in the last column start
	% off in a different relative position than the graphs in the other
	% columns. Here, we manually account for that.
	if mod(theid-1,cols) == 6
		tpos(1) = tpos(1) + -10;
	else
		tpos(1) = tpos(1) + -40;
	end
	set( titl, 'Position', tpos );
	set( titl, 'FontSize', fontsize );
else % impl == 'matlab'
	tpos = get( titl, 'Position' );
	tpos(1) = tpos(1) + 90;
	set( titl, 'Position', tpos );
end

if theid > (rows-1)*cols
	%xlab = xlabel( ax1,xaxisname );
	%tpos = get( xlab, 'Position' )
	%tpos(2) = tpos(2) + 10;
	%set( xlab, 'Position', tpos );
	if     theid == rows*cols - 6
	xlab = xlabel( ax1, 'm = 6; n = k' );
	elseif theid == rows*cols - 5
	xlab = xlabel( ax1, 'n = 8; m = k' );
	elseif theid == rows*cols - 4
	xlab = xlabel( ax1, 'k = 4; m = n' );
	elseif theid == rows*cols - 3
	xlab = xlabel( ax1, 'm; n = 8, k = 4' );
	elseif theid == rows*cols - 2
	xlab = xlabel( ax1, 'n; m = 6, k = 4' );
	elseif theid == rows*cols - 1
	xlab = xlabel( ax1, 'k; m = 6, n = 8' );
	elseif theid == rows*cols - 0
	xlab = xlabel( ax1, 'm = n = k' );
	end
end

if mod(theid-1,cols) == 0
	ylab = ylabel( ax1,yaxisname );
end

r_val = 0;

