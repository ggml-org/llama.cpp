function r_val = plot_panel_4x5 ...
     ( ...
       cfreq, ...
       dflopspercycle, ...
       nth, ...
       thr_str, ...
       dirpath, ...
       arch_str, ...
       vend_leg_str  ...
     )

impl = 'octave';
%impl = 'matlab';

%sp = 'default';
subp = 'tight';

if strcmp( subp, 'default' )
	position  = [100 100 2000 1500];
	papersize = [14.2 19.0];
	leg_pos_st = [3.40 8.70 1.9 1.0 ]; % (0,2br)
	leg_pos_mt = [13.08 13.09 1.9 1.0 ]; % (0,3tr)
	sp_margins = [ 0.070 0.049 ];
else
	position  = [100 100 1864 1540];
	papersize = [15.6 19.4];
	%leg_pos_st = [1.15 8.70 2.1 1.2 ]; % (dgemm)
	%leg_pos_st = [1.60 8.80 2.1 1.2 ]; % (dgemm)
	leg_pos_st = [15.90 13.60 2.1 1.2 ]; % (strsm)
	%leg_pos_mt = [12.20 13.60 2.1 1.2 ]; % (strmm)
	%leg_pos_mt = [5.30 12.60 2.1 1.2 ]; % (ssymm)
	%leg_pos_mt = [8.50 13.62 2.1 1.2 ]; % (ssyrk)
	%leg_pos_mt = [5.30 5.10 2.1 1.2 ]; % (chemm)
	leg_pos_mt = [15.90 13.60 2.1 1.2 ]; % (strsm)
	sp_margins = [ 0.068 0.051 ];
end

%fig = figure('Position', [100, 100, 2000, 1500]);
fig = figure('Position', position);
orient( fig, 'portrait' );
set(gcf,'PaperUnits', 'inches');
if strcmp( impl, 'octave' )
	%set(gcf,'PaperSize', [14.2 19.0]);
	set(gcf,'PaperSize', papersize);
	%set(gcf,'PaperPositionMode','auto');
	set(gcf,'PaperPositionMode','auto');
else % impl == 'matlab'
	set(gcf,'PaperSize', [13 20.0]);
	set(gcf,'PaperPosition', [0 0 13 20.0]);
	set(gcf,'PaperPositionMode','manual');
end
set(gcf,'PaperOrientation','landscape');

% Define the implementation strings. These appear in both the filenames of the
% files that contain the performance results as well as the variable names
% within those files.
blis_str = 'asm_blis';
open_str = 'openblas';
vend_str = 'vendor';
eige_str = 'eigen';

% Create filename "templates" for the files that contain the performance
% results.
filetemp      = '%s/output_%s_%s_%s.m';
filetemp_blis = sprintf( filetemp, '%s', '%s', '%s', blis_str );
filetemp_open = sprintf( filetemp, '%s', '%s', '%s', open_str );
filetemp_vend = sprintf( filetemp, '%s', '%s', '%s', vend_str );
filetemp_eige = sprintf( filetemp, '%s', '%s', '%s', eige_str );

% Create a variable name "template" for the variables contained in the
% files outlined above.
vartemp = 'data_%s_%s_%s( :, : )';

% Define the datatypes and operations we will be plotting.
dts = [ 's' 'd' 'c' 'z' ];
ops( 1, : ) = 'gemm';
ops( 2, : ) = 'hemm';
ops( 3, : ) = 'herk';
ops( 4, : ) = 'trmm';
ops( 5, : ) = 'trsm';

% Generate datatype-specific operation names from the set of operations
% and datatypes.
opnames = gen_opnames( ops, dts );
n_opnames = size( opnames, 1 );

% Iterate over the list of datatype-specific operation names.
for opi = 1:n_opnames
%for opi = 1:1

	% Grab the current datatype combination.
	opname = opnames( opi, : );

	str = sprintf( 'Plotting %d: %s', opi, opname ); disp(str);

	data_blis = read_data( filetemp_blis, dirpath, vartemp, thr_str, opname, blis_str );
	data_open = read_data( filetemp_open, dirpath, vartemp, thr_str, opname, open_str );
	data_vend = read_data( filetemp_vend, dirpath, vartemp, thr_str, opname, vend_str );
	data_eige = read_data( filetemp_eige, dirpath, vartemp, thr_str, opname, eige_str );
	
	% Plot one result in an m x n grid of plots, via the subplot()
	% function.
	plot_l3_perf( opname, ...
	              data_blis, ...
	              data_open, ...
	              data_eige, ...
	              data_vend, vend_leg_str, ...
	              nth, ...
	              4, 5, ...
	              cfreq, ...
	              dflopspercycle, ...
	              opi, ...
	              leg_pos_st, leg_pos_mt, ...
	              sp_margins );

end


% Construct the name of the file to which we will output the graph.
outfile = sprintf( 'l3_perf_%s_nt%d.pdf', arch_str, nth );

% Output the graph to pdf format.
if strcmp( impl, 'octave' )
	print( gcf, outfile );
else
	print( gcf, outfile, '-bestfit', '-dpdf' );
end

