function r_val = plot_l3sup_perf( opname, ...
                                  smalldims, ...
                                  data_blissup, ...
                                  data_blisconv, ...
                                  data_eigen, ...
                                  data_open, ...
                                  data_vend, vend_str, ...
                                  data_bfeo, ...
                                  data_xsmm, ...
                                  nth, ...
                                  rows, cols, ...
                                  cfreq, ...
                                  dfps, ...
                                  theid, impl )

% Define a single-/multithreadedness predicate for convenience.
if nth == 1
	is_st = 1;
else
	is_st = 0;
end

% Define the column in which the performance rates are found.
flopscol = size( data_blissup, 2 );

% Check if blasfeo data is available.
has_bfeo = 1;
if data_bfeo( 1, flopscol ) == 0.0
	has_bfeo = 0;
end

% Check if libxsmm data is available.
has_xsmm = 1;
if data_xsmm( 1, flopscol ) == 0.0
	has_xsmm = 0;
end

% Define which plot id will have the legend.
%if is_st
%	if has_xsmm == 1
%		legend_plot_id = 2*cols + 1*5;
%	else
%		legend_plot_id = 1*cols + 1*5;
%	end
%else
%	legend_plot_id = 0*cols + 1*6;
%end
legend_plot_id = cols*rows;

% Hold the axes.
if 1
	ax1 = subplot( rows, cols, theid );
	hold( ax1, 'on' );
end

% Set line properties.
color_blissup  = 'k'; lines_blissup  = '-';  markr_blissup  = '';
color_blisconv = 'k'; lines_blisconv = ':';  markr_blisconv = '';
color_eigen    = 'm'; lines_eigen    = '-.'; markr_eigen    = 'o';
color_open     = 'r'; lines_open     = '--'; markr_open     = 'o';
color_vend     = 'b'; lines_vend     = '-.'; markr_vend     = '.';
color_bfeo     = 'c'; lines_bfeo     = '-';  markr_bfeo     = 'o';
color_xsmm     = 'g'; lines_xsmm     = '-';  markr_xsmm     = 'o';

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
blissup_lg  = sprintf( 'BLIS sup' );
blisconv_lg = sprintf( 'BLIS conv' );
eigen_lg    = sprintf( 'Eigen' );
open_lg     = sprintf( 'OpenBLAS' );
vend_lg     = vend_str;
bfeo_lg     = sprintf( 'BLASFEO' );
xsmm_lg     = sprintf( 'libxsmm' );

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

% Compute the number of data points we have in the x-axis. Note that we
% only use half the data points for the m = n = k column of graphs.
%if mod(theid-1,cols) == 6
%	np = size( data_blissup, 1 ) / 2;
%else
%	np = size( data_blissup, 1 );
%end
np = size( data_blissup, 1 );

% Grab the last x-axis value.
x_end = data_blissup( np, psize_col );

%data_peak( 1, 1:2 ) = [     0 max_perf_core ];
%data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

blissup_ln  = line( x_axis( 1:np, 1 ), data_blissup( 1:np, flopscol ) / nth, ...
					'Color',color_blissup, 'LineStyle',lines_blissup, ...
					'LineWidth',linesize );
blisconv_ln = line( x_axis( 1:np, 1 ), data_blisconv( 1:np, flopscol ) / nth, ...
					'Color',color_blisconv, 'LineStyle',lines_blisconv, ...
					'LineWidth',linesize );
eigen_ln    = line( x_axis( 1:np, 1 ), data_eigen( 1:np, flopscol ) / nth, ...
					'Color',color_eigen, 'LineStyle',lines_eigen, ...
					'LineWidth',linesize );
open_ln     = line( x_axis( 1:np, 1 ), data_open( 1:np, flopscol ) / nth, ...
					'Color',color_open, 'LineStyle',lines_open, ...
					'LineWidth',linesize );
vend_ln     = line( x_axis( 1:np, 1 ), data_vend( 1:np, flopscol ) / nth, ...
					'Color',color_vend, 'LineStyle',lines_vend, ...
					'LineWidth',linesize );
if has_bfeo == 1
	bfeo_ln     = line( x_axis( 1:np, 1 ), data_bfeo( 1:np, flopscol ) / nth, ...
						'Color',color_bfeo, 'LineStyle',lines_bfeo, ...
						'LineWidth',linesize );
else
	bfeo_ln     = line( nan, nan, ...
						'Color',color_bfeo, 'LineStyle',lines_bfeo, ...
						'LineWidth',linesize );
end
if has_xsmm == 1
	xsmm_ln     = line( x_axis( 1:np, 1 ), data_xsmm( 1:np, flopscol ) / nth, ...
						'Color',color_xsmm, 'LineStyle',lines_xsmm, ...
						'LineWidth',linesize );
else
	xsmm_ln     = line( nan, nan, ...
						'Color',color_xsmm, 'LineStyle',lines_xsmm, ...
						'LineWidth',linesize );
end


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if    10000 <= x_end && x_end < 15000
	x_tick2 = x_end - 2000;
	x_tick1 = x_tick2/2;
	%xticks( ax1, [ x_tick1 x_tick2 ] );
	xticks( ax1, [ 3000 6000 9000 12000 ] );
elseif 6000 <= x_end && x_end < 10000
	x_tick2 = x_end - 2000;
	x_tick1 = x_tick2/2;
	%xticks( ax1, [ x_tick1 x_tick2 ] );
	xticks( ax1, [ 2000 4000 6000 8000 ] );
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

	%                    xpos ypos
	%set( leg,'Position',[11.32 6.36 1.15 0.7 ] ); % (1,4tl)
if nth == 1 && theid == legend_plot_id
	if has_xsmm == 1
		% single-threaded, with libxsmm (ccc)
		leg = legend( ...
		[ blissup_ln  blisconv_ln  eigen_ln  open_ln  vend_ln  bfeo_ln  xsmm_ln ], ...
		  blissup_lg, blisconv_lg, eigen_lg, open_lg, vend_lg, bfeo_lg, xsmm_lg, ...
		'Location', legend_loc );
		set( leg,'Box','off','Color','none','Units','inches' );
		if impl == 'octave'
			set( leg,'FontSize',fontsize );
			set( leg,'Position',[15.35 4.62 1.9 1.20] ); % (1,4tl)
		else
			set( leg,'FontSize',fontsize-3 );
			set( leg,'Position',[18.20 10.20 1.15 0.7 ] ); % (1,4tl)
		end
	else
		% single-threaded, without libxsmm (rrr, or other)
		leg = legend( ...
		[ blissup_ln  blisconv_ln  eigen_ln  open_ln  vend_ln  bfeo_ln  ], ...
		  blissup_lg, blisconv_lg, eigen_lg, open_lg, vend_lg, bfeo_lg, ...
		'Location', legend_loc );
		set( leg,'Box','off','Color','none','Units','inches' );
		if impl == 'octave'
			set( leg,'FontSize',fontsize );
			set( leg,'Position',[15.35 7.40 1.9 1.10] ); % (1,4tl)
		else
			set( leg,'FontSize',fontsize-1 );
			set( leg,'Position',[18.24 10.15 1.15 0.7] ); % (1,4tl)
		end
	end
elseif nth > 1 && theid == legend_plot_id
	% multithreaded
	leg = legend( ...
	[ blissup_ln  blisconv_ln  eigen_ln  open_ln  vend_ln ], ...
	  blissup_lg, blisconv_lg, eigen_lg, open_lg, vend_lg, ...
	'Location', legend_loc );
	set( leg,'Box','off','Color','none','Units','inches' );
	if impl == 'octave'
		set( leg,'FontSize',fontsize );
		set( leg,'Position',[18.20 10.30 1.9 0.95] ); % (1,4tl)
	else
		set( leg,'FontSize',fontsize-1 );
		set( leg,'Position',[18.24 10.15 1.15 0.7] ); % (1,4tl)
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

sll_str = sprintf( 'm = %u; n = k', smalldims(1) );
lsl_str = sprintf( 'n = %u; m = k', smalldims(2) );
lls_str = sprintf( 'k = %u; m = n', smalldims(3) );
lss_str = sprintf( 'm; n = %u, k = %u', smalldims(2), smalldims(3) );
sls_str = sprintf( 'n; m = %u, k = %u', smalldims(1), smalldims(3) );
ssl_str = sprintf( 'k; m = %u, n = %u', smalldims(1), smalldims(2) );
lll_str = sprintf( 'm = n = k' );

% Place labels on the bottom row of graphs.
if theid > (rows-1)*cols
	%xlab = xlabel( ax1,xaxisname );
	%tpos = get( xlab, 'Position' )
	%tpos(2) = tpos(2) + 10;
	%set( xlab, 'Position', tpos );
	if     theid == rows*cols - 6
		xlab = xlabel( ax1, sll_str );
	elseif theid == rows*cols - 5
		xlab = xlabel( ax1, lsl_str );
	elseif theid == rows*cols - 4
		xlab = xlabel( ax1, lls_str );
	elseif theid == rows*cols - 3
		xlab = xlabel( ax1, lss_str );
	elseif theid == rows*cols - 2
		xlab = xlabel( ax1, sls_str );
	elseif theid == rows*cols - 1
		xlab = xlabel( ax1, ssl_str );
	elseif theid == rows*cols - 0
		xlab = xlabel( ax1, lll_str );
	end
end

% Place labels on the left-hand column of graphs.
if mod(theid-1,cols) == 0
	ylab = ylabel( ax1,yaxisname );
end

r_val = 0;

