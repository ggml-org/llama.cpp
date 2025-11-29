function r_val = plot_gemm_perf( dt_str, ...
                                 data_ref, ...
                                 data_intern_w, ...
                                 data_intern_wo, ...
                                 data_ad_hoc, ...
                                 refch, ...
                                 nth, ...
                                 rows, cols, ...
                                 cfreq, ...
                                 dfps, ...
                                 theid )

if 1
ax1 = subplot( rows, cols, theid );
hold( ax1, 'on' );
end

% Set line properties.
color_ref       = 'b'; lines_ref       = ':';  markr_ref       = '';
color_intern_w  = 'r'; lines_intern_w  = '-';  markr_intern_w  = '';
color_intern_wo = 'b'; lines_intern_wo = '--'; markr_intern_wo = '.';
color_ad_hoc    = 'k'; lines_ad_hoc    = '-.'; markr_ad_hoc    = '';

% Compute the peak performance in terms of the number of double flops
% executable per cycle and the clock rate.
if dt_str(4) == 's'
	flopspercycle = dfps * 2;
else
	flopspercycle = dfps;
end
max_perf_core = (flopspercycle * cfreq) * 1;

% Print the title to a string.
%titlename = '%sgemm';
titlename = '%s';
titlename = sprintf( titlename, dt_str );

% Set the legend strings.
if refch == 's'
ref_legend      = sprintf( 'Ref (sgemm)' );
elseif refch == 'd'
ref_legend      = sprintf( 'Ref (dgemm)' );
elseif refch == 'c'
ref_legend      = sprintf( 'Ref (cgemm)' );
elseif refch == 'z'
ref_legend      = sprintf( 'Ref (zgemm)' );
end
internw_legend  = sprintf( 'Intern (+xm)' );
internwo_legend = sprintf( 'Intern (-xm)' );
ad_hoc_legend   = sprintf( 'Ad-hoc' );

% Set axes range values.
y_scale   = 1.00;
x_begin   = 0;
x_end     = data_ref( size( data_ref, 1 ), 1 );
y_begin   = 0;
y_end     = max_perf_core * y_scale;

% Set axes names.
xaxisname = '     m = n = k';
if nth == 1
	yaxisname = 'GFLOPS';
else
	yaxisname = 'GFLOPS/core';
end


flopscol = 4;
msize = 5;
if 1
fontsize = 13;
else
fontsize = 16;
end
linesize = 0.5;
legend_loc = 'southeast';

% --------------------------------------------------------------------

x_axis( :, 1 ) = data_intern_w( :, 1 );

data_peak( 1, 1:2 ) = [     0 max_perf_core ];
data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

ref       = line( x_axis( :, 1 ), data_ref( :, flopscol ) / nth, ...
                  'Color',color_ref, 'LineStyle',lines_ref, ...
                  'LineWidth',linesize );
if ( uses_xmem( dt_str ) )
intern_w  = line( x_axis( :, 1 ), data_intern_w( :, flopscol ) / nth, ...
                  'Color',color_intern_w, 'LineStyle',lines_intern_w, ...
                  'LineWidth',linesize );
else
%set( intern_w, 'visible', 'off' );
intern_w  = line( nan, nan, ...
                  'Color',color_intern_w, 'LineStyle',lines_intern_w, ...
                  'LineWidth',linesize );
end
intern_wo = line( x_axis( :, 1 ), data_intern_wo( :, flopscol ) / nth, ...
                  'Color',color_intern_wo, 'LineStyle',lines_intern_wo, ...
                  'LineWidth',linesize );
ad_hoc    = line( x_axis( :, 1 ), data_ad_hoc( :, flopscol ) / nth, ...
                  'Color',color_ad_hoc, 'LineStyle',lines_ad_hoc, ...
                  'LineWidth',linesize );


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if x_end == 6000
    x_tick2 = x_end - 1000;
    x_tick1 = x_tick2/2;
    xticks( ax1, [ x_tick1 x_tick2 ] );
elseif x_end == 2000
    x_tick2 = x_end - 400;
    x_tick1 = x_tick2/2;
    xticks( ax1, [ x_tick1 x_tick2 ] );
end

% full domain case
if rows == 4 && cols == 4

	if theid == 2 || theid == 4

		leg = legend( ...
		[ ...
		  ref ...
		  intern_w ...
		  intern_wo ...
		  ad_hoc ...
		], ...
		ref_legend, ...
		internw_legend, ...
		internwo_legend, ...
		ad_hoc_legend, ...
		'Location', legend_loc );
		%'Location', 'best' );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-2 );
		set( leg,'Units','inches' );
		if theid == 2
			set( leg,'Position',[2.31 3.52 0.7 0.3 ] );
		elseif theid == 4
			set( leg,'Position',[4.80 3.52 0.7 0.3 ] );
		end
	end

% select graphs
elseif rows == 3 && cols == 4

	if theid == 2 || theid == 4

		leg = legend( ...
		[ ...
		  ref ...
		  intern_w ...
		  intern_wo ...
		  ad_hoc ...
		], ...
		ref_legend, ...
		internw_legend, ...
		internwo_legend, ...
		ad_hoc_legend, ...
		'Location', legend_loc );
		%'Location', 'best' );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-2 );
		set( leg,'Units','inches' );
		if theid == 2
			set( leg,'Position',[4.38 4.78 0.7 0.3 ] );
		elseif theid == 4
			set( leg,'Position',[8.82 4.78 0.7 0.3 ] );
		end
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

if theid > (rows-1)*cols
xlab = xlabel( ax1,xaxisname );
%tpos = get( xlab, 'Position' )
%tpos(2) = tpos(2) + 10;
%set( xlab, 'Position', tpos );
end

if mod(theid-1,cols) == 0
ylab = ylabel( ax1,yaxisname );
end


r_val = 0;

end
