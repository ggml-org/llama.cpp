function r_val = plot_gemm_perf( dt_str, ...
                                 data_ref, ...
                                 data_intern, ...
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
color_ref    = 'b'; lines_ref    = ':';  markr_ref    = '';
color_intern = 'b'; lines_intern = '-';  markr_intern = '';
color_ad_hoc = 'k'; lines_ad_hoc = '-.'; markr_ad_hoc = '';

% Compute the peak performance in terms of the number of double flops
% executable per cycle and the clock rate.
if dt_str(4) == 's'
	flopspercycle = dfps * 2;
else
	flopspercycle = dfps;
end
max_perf_core = (flopspercycle * cfreq) * 1;

% Print the title to a string.
titlename = '%s';
titlename = sprintf( titlename, dt_str );

% Set the legend strings.
if refch == 's'
ref_legend    = sprintf( 'Ref (sgemm)' );
elseif refch == 'd'
ref_legend    = sprintf( 'Ref (dgemm)' );
elseif refch == 'c'
ref_legend    = sprintf( 'Ref (cgemm)' );
elseif refch == 'z'
ref_legend    = sprintf( 'Ref (zgemm)' );
end
intern_legend = sprintf( 'Internal' );
ad_hoc_legend = sprintf( 'Ad-hoc' );

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

x_axis( :, 1 ) = data_intern( :, 1 );

data_peak( 1, 1:2 ) = [     0 max_perf_core ];
data_peak( 2, 1:2 ) = [ x_end max_perf_core ];

ref     = line( x_axis( :, 1 ), data_ref( :, flopscol ) / nth, ...
                'Color',color_ref, 'LineStyle',lines_ref, ...
                'LineWidth',linesize );
intern  = line( x_axis( :, 1 ), data_intern( :, flopscol ) / nth, ...
                'Color',color_intern, 'LineStyle',lines_intern, ...
                'LineWidth',linesize );
ad_hoc  = line( x_axis( :, 1 ), data_ad_hoc( :, flopscol ) / nth, ...
                'Color',color_ad_hoc, 'LineStyle',lines_ad_hoc, ...
                'LineWidth',linesize );


xlim( ax1, [x_begin x_end] );
ylim( ax1, [y_begin y_end] );

if rows == 8 && cols == 16

	refs_legend = sprintf( 'Ref [sc]gemm' );
	refd_legend = sprintf( 'Ref [dz]gemm' );

	if theid == 1

		leg = legend( ...
		[ ...
		  ref ...
		  intern ...
		  ad_hoc ...
		], ...
		refs_legend, ...
		intern_legend, ...
		ad_hoc_legend, ...
		'Location', 'best' );
		%'Location', legend_loc );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-2 );
		set( leg,'Units','inches' );

	elseif theid == 9

		leg = legend( ...
		[ ...
		  ref ...
		  intern ...
		  ad_hoc ...
		], ...
		refd_legend, ...
		intern_legend, ...
		ad_hoc_legend, ...
		'Location', 'best' );
		%'Location', legend_loc );
		set( leg,'Box','off' );
		set( leg,'Color','none' );
		set( leg,'FontSize',fontsize-2 );
		set( leg,'Units','inches' );
	end

elseif rows == 4 && cols == 4

	if theid == 2 || theid == 4

		leg = legend( ...
		[ ...
		  ref ...
		  intern ...
		  ad_hoc ...
		], ...
		ref_legend, ...
		intern_legend, ...
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
		%set( leg,'Position',[1.03 3.46 0.7 0.3 ] );
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
