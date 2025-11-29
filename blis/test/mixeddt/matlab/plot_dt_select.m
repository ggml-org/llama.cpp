function r_val = plot_dt_select( dom, is_mt )

if is_mt == 1
	thr_str = 'mt';
else
	thr_str = 'st';
end

if dom == 'r'

	dt_combos(  1, : ) = [ 'dsss' ];
	dt_combos(  2, : ) = [ 'sddd' ];
	dt_combos(  3, : ) = [ 'sdds' ];
	dt_combos(  4, : ) = [ 'dssd' ];
	dt_combos(  5, : ) = [ 'ddds' ];
	dt_combos(  6, : ) = [ 'sssd' ];

else

	dt_combos(  1, : ) = [ 'csss' ];
	dt_combos(  2, : ) = [ 'zddd' ];
	dt_combos(  3, : ) = [ 'ccss' ];
	dt_combos(  4, : ) = [ 'zzdd' ];
	dt_combos(  5, : ) = [ 'cscs' ];
	dt_combos(  6, : ) = [ 'zdzd' ];
end

n_combos = size(dt_combos,1);

filetemp_blis = '../output_%s_%sgemm_asm_blis.m';
filetemp_open = '../output_%s_%sgemm_openblas.m';

% Construct filenames for the "reference" (single real) data, then load
% the data files, and finally save the results to different variable names.
file_blis_sref = sprintf( filetemp_blis, thr_str, 'ssss' );
file_open_sref = sprintf( filetemp_open, thr_str, 'ssss' );
run( file_blis_sref )
run( file_open_sref )
data_gemm_asm_blis_sref( :, : ) = data_gemm_asm_blis( :, : );
data_gemm_openblas_sref( :, : ) = data_gemm_openblas( :, : );

% Construct filenames for the "reference" (double real) data, then load
% the data files, and finally save the results to different variable names.
file_blis_dref = sprintf( filetemp_blis, thr_str, 'dddd' );
file_open_dref = sprintf( filetemp_open, thr_str, 'dddd' );
run( file_blis_dref )
run( file_open_dref )
data_gemm_asm_blis_dref( :, : ) = data_gemm_asm_blis( :, : );
data_gemm_openblas_dref( :, : ) = data_gemm_openblas( :, : );

% Construct filenames for the "reference" (single complex) data, then load
% the data files, and finally save the results to different variable names.
file_blis_cref = sprintf( filetemp_blis, thr_str, 'cccs' );
file_open_cref = sprintf( filetemp_open, thr_str, 'cccs' );
run( file_blis_cref )
run( file_open_cref )
data_gemm_asm_blis_cref( :, : ) = data_gemm_asm_blis( :, : );
data_gemm_openblas_cref( :, : ) = data_gemm_openblas( :, : );

% Construct filenames for the "reference" (double complex) data, then load
% the data files, and finally save the results to different variable names.
file_blis_zref = sprintf( filetemp_blis, thr_str, 'zzzd' );
file_open_zref = sprintf( filetemp_open, thr_str, 'zzzd' );
run( file_blis_zref )
run( file_open_zref )
data_gemm_asm_blis_zref( :, : ) = data_gemm_asm_blis( :, : );
data_gemm_openblas_zref( :, : ) = data_gemm_openblas( :, : );

%fig = figure;
fig = figure('Position', [100, 100, 1024, 1300]);
orient( fig, 'portrait' );
%set(gcf,'Position',[0 0 2000 900]);
set(gcf,'PaperUnits', 'inches');
%set(gcf,'PaperSize', [16 12.4]);
%set(gcf,'PaperPosition', [0 0 16 12.4]);
set(gcf,'PaperSize', [9 11.0]);
set(gcf,'PaperPosition', [0 0 9 11.0]);
%set(gcf,'PaperPositionMode','auto');
set(gcf,'PaperPositionMode','manual');
set(gcf,'PaperOrientation','portrait');

for dti = 1:n_combos
%for dti = 1:1

	% Grab the current datatype combination.
	combo = dt_combos( dti, : );

	str = sprintf( 'Plotting %d: %s', dti, combo ); disp(str);

	if combo(4) == 's'
		data_gemm_asm_blis_ref( :, : ) = data_gemm_asm_blis_sref( :, : );
		data_gemm_openblas_ref( :, : ) = data_gemm_openblas_sref( :, : );
		refch = 's';
	else %if combo(4) == 'd'
		data_gemm_asm_blis_ref( :, : ) = data_gemm_asm_blis_dref( :, : );
		data_gemm_openblas_ref( :, : ) = data_gemm_openblas_dref( :, : );
		refch = 'd';
	end

	if ( combo(1) == 'c' || combo(1) == 'z' ) && ...
	   ( combo(2) == 'c' || combo(2) == 'z' ) && ...
	   ( combo(3) == 'c' || combo(3) == 'z' )
		if combo(4) == 's'
			data_gemm_asm_blis_ref( :, : ) = data_gemm_asm_blis_cref( :, : );
			data_gemm_openblas_ref( :, : ) = data_gemm_openblas_cref( :, : );
			refch = 'c';
		else %if combo(4) == 'd'
			data_gemm_asm_blis_ref( :, : ) = data_gemm_asm_blis_zref( :, : );
			data_gemm_openblas_ref( :, : ) = data_gemm_openblas_zref( :, : );
			refch = 'z';
		end
	end

	% Construct filenames for the data files from templates.
	file_blis = sprintf( filetemp_blis, thr_str, combo );
	file_open = sprintf( filetemp_open, thr_str, combo );

	% Load the data files.
	%str = sprintf( '  Loading %s', file_blis ); disp(str);
	run( file_blis )
	%str = sprintf( '  Loading %s', file_open ); disp(str);
	run( file_open )

	% Plot the result.
	plot_gemm_perf( combo, ...
	                data_gemm_asm_blis, ...
	                data_gemm_asm_blis_ref, ...
	                data_gemm_openblas, ...
	                data_gemm_openblas_ref, ...
	                is_mt, refch, 3, 2, dti );

end


%if 0
%set(gcf,'Position',[0 0 2000 900]);
%set(gcf,'PaperUnits', 'inches');
%set(gcf,'PaperSize', [48 22]);
%set(gcf,'PaperPosition', [0 0 48 22]);
%%set(gcf,'PaperPositionMode','auto');
%set(gcf,'PaperPositionMode','manual');
%set(gcf,'PaperOrientation','landscape');
%end

outfile = sprintf( 'output/gemm_select_%c', dom );

print(gcf, outfile,'-bestfit','-dpdf');
%print(gcf, 'gemm_md','-fillpage','-dpdf');
