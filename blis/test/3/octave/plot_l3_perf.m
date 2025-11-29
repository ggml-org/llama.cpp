function r_val = plot_l3_perf( opname, ...
                               data_blis, ...
                               data_open, ...
                               data_eige, ...
                               data_vend, vend_str, ...
                               nth, ...
                               rows, cols, ...
                               cfreq, ...
                               dfps, ...
                               theid, ...
                               leg_pos_st, leg_pos_mt, ...
                               sp_margins )

% Define the column in which the performance rates are found.
flopscol = size( data_blis, 2 );

% Define which plot id will have the legend.
% NOTE: We can draw the legend on any graph as long as it has already been
% rendered. Since the coordinates are global, we can simply always wait until
% the final graph to draw the legend.
legend_plot_id = cols*rows;

% Set line properties.
color_blis = 'k'; lines_blis = '-';  markr_blis = '';
color_open = 'r'; lines_open = '--'; markr_open = 'o';
color_eige = 'm'; lines_eige = '-.'; markr_eige = 'x';
color_vend = 'b'; lines_vend = '-.'; markr_vend = '.';

% Compute the peak performance in terms of the number of double flops
% executable per cycle and the clock rate.
if opname(1) == 's' || opname(1) == 'c'
	flopspercycle = dfps * 2;
else
	flopspercycle = dfps;
end
max_perf_core = (flopspercycle * cfreq) * 1;

% Adjust title for real domain hemm and herk.
title_opname = opname;
if opname(1) == 's' || opname(1) == 'd' 
%	if strcmp( extractAfter( opname, 1 ), 'hemm' ) || ...
%	   strcmp( extractAfter( opname, 1 ), 'herk' )
%		title_opname(2:3) = 'sy';
%	end
	opname_u = opname; opname_u(1) = '_';
	if strcmp( opname_u, '_hemm' ) || ...
	   strcmp( opname_u, '_herk' )
		title_opname(2:3) = 'sy';
	end
end

% Print the title to a string.
titlename = '%s';
titlename = sprintf( titlename, title_opname );

% Set the legend strings.
blis_legend = sprintf( 'BLIS' );
open_legend = sprintf( 'OpenBLAS' );
eige_legend = sprintf( 'Eigen' );
%vend_legend  = sprintf( 'MKL' );
%vend_legend  = sprintf( 'ARMPL' );
vend_legend = vend_str;

% Set axes range values.
y_scale = 1.00;
x_begin = 0;
x_end   = data_blis( size( data_blis, 1 ), 1 );
y_begin = 0;
y_end   = max_perf_core * y_scale;

% Set axes names.
if nth == 1
	yaxisname = 'GFLOPS';
else
	yaxisname = 'GFLOPS/core';
end

% Set the marker size, font size, and other items.
msize = 5;
if 0
	xaxisname = '      m = n = k';
	fontsize = 12;
else
	xaxisname = 'm = n = k';
	fontsize = 20;
end
linesize = 0.8;
legend_loc = 'southeast';

%ax1 = subplot( rows, cols, theid );
ax1 = subplot_tight( rows, cols, theid, sp_margins );

% Hold the axes.
hold( ax1, 'on' );

% --------------------------------------------------------------------

x_axis( :, 1 ) = data_blis( :, 1 );

data_peak( 1, 1:2 ) = [     0 max_perf_core ];
data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

% Plot the data series for BLIS, which is required.
blis_ln  = line( x_axis( :, 1 ), data_blis( :, flopscol ) / nth, ...
				 'Color',color_blis, 'LineStyle',lines_blis, ...
				 'LineWidth',linesize );

% Plot the data series for OpenBLAS, if applicable.
if data_open(1,1) ~= -1
	open_ln  = line( x_axis( :, 1 ), data_open( :, flopscol ) / nth, ...
					 'Color',color_open, 'LineStyle',lines_open, ...
					 'LineWidth',linesize );
else
	open_ln  = line( nan, nan, ...
					 'Color',color_open, 'LineStyle',lines_open, ...
					 'LineWidth',linesize );
end

% Plot the data series for a vendor library, if applicable.
if data_vend(1,1) ~= -1
	vend_ln  = line( x_axis( :, 1 ), data_vend( :, flopscol ) / nth, ...
					 'Color',color_vend, 'LineStyle',lines_vend, ...
					 'LineWidth',linesize );
else
	vend_ln  = line( nan, nan, ...
					 'Color',color_vend, 'LineStyle',lines_vend, ...
					 'LineWidth',linesize );
end

% Plot the data series for Eigen, if applicable.
if data_eige(1,1) ~= -1
	eige_ln  = line( x_axis( :, 1 ), data_eige( :, flopscol ) / nth, ...
					 'Color',color_eige, 'LineStyle',lines_eige, ...
					 'LineWidth',linesize );
else
	eige_ln  = line( nan, nan, ...
					 'Color',color_eige, 'LineStyle',lines_eige, ...
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
end

if rows == 4 && cols == 5

	if nth == 1 && theid == legend_plot_id

		leg = legend( [ blis_ln vend_ln open_ln eige_ln ], ...
		              blis_legend, vend_legend, open_legend, eige_legend, ...
		              'Location', legend_loc );
		set( leg,'Box','off','Color','none','Units','inches','FontSize',fontsize );
		set( leg,'Position',leg_pos_st );

	elseif nth > 1 && theid == legend_plot_id

		leg = legend( [ blis_ln vend_ln open_ln eige_ln ], ...
		              blis_legend, vend_legend, open_legend, eige_legend, ...
		              'Location', legend_loc );
		set( leg,'Box','off','Color','none','Units','inches','FontSize',fontsize );
		set( leg,'Position',leg_pos_mt );
	end
end

set( ax1,'FontSize',fontsize );
set( ax1,'TitleFontSizeMultiplier',1.0 ); % default is 1.1.
box( ax1, 'on' );

titl = title( titlename );
set( titl, 'FontWeight', 'normal' ); % default font style is now 'bold'.

tpos = get( titl, 'Position' ); % default is to align across whole figure, not box.
%tpos(1) = tpos(1) + 100;
tpos(1) = tpos(1) + 40;
set( titl, 'Position', tpos ); % here we nudge it back to centered with box.
set( titl, 'FontSize', fontsize );

if theid > (rows-1)*cols
	%tpos = get( xlab, 'Position' )
	%tpos(2) = tpos(2) + 10;
	%set( xlab, 'Position', tpos );
	xlab = xlabel( ax1,xaxisname );
end

if mod(theid-1,cols) == 0
	ylab = ylabel( ax1,yaxisname );
end

r_val = 0;

