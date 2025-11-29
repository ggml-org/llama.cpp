function r_val = plot_dom_case( mdcase, ...
                                cfreq, ...
                                dflopspercycle, ...
                                nth, ...
                                dirpath_w, ...
                                dirpath_wo, ...
                                dirpath_out, ...
                                arch_str )

% Create filename "templates" for the files that contain the performance
% results.
filetemp_intern = '%s/output_%s_%sgemm_intern.m';
filetemp_ad_hoc = '%s/output_%s_%sgemm_ad_hoc.m';

if nth == 1
	thr_str = 'st';
else
	thr_str = 'mt';
end

if 1
	dt_combos = gen_prec_combos( mdcase );
else
	dt_combos( 1, : ) = [ 'ssss' ];
	dt_combos( 2, : ) = [ 'sssd' ];
	dt_combos( 3, : ) = [ 'ssds' ];
	dt_combos( 4, : ) = [ 'sdss' ];
	dt_combos( 5, : ) = [ 'dsss' ];
	dt_combos( 6, : ) = [ 'ddds' ];
	dt_combos( 7, : ) = [ 'dddd' ];
end

n_combos = size(dt_combos,1);

% Construct filenames for the "reference" (single real) data, then load
% the data files, and finally save the results to different variable names.
file_blis_sref = sprintf( filetemp_intern, dirpath_w, thr_str, 'ssss' );
run( file_blis_sref )
data_gemm_intern_sref( :, : ) = data_gemm_intern( :, : );

% Construct filenames for the "reference" (double real) data, then load
% the data files, and finally save the results to different variable names.
file_blis_dref = sprintf( filetemp_intern, dirpath_w, thr_str, 'dddd' );
run( file_blis_dref )
data_gemm_intern_dref( :, : ) = data_gemm_intern( :, : );

% Construct filenames for the "reference" (single complex) data, then load
% the data files, and finally save the results to different variable names.
file_blis_cref = sprintf( filetemp_intern, dirpath_w, thr_str, 'cccs' );
run( file_blis_cref )
data_gemm_intern_cref( :, : ) = data_gemm_intern( :, : );

% Construct filenames for the "reference" (double complex) data, then load
% the data files, and finally save the results to different variable names.
file_blis_zref = sprintf( filetemp_intern, dirpath_w, thr_str, 'zzzd' );
run( file_blis_zref )
data_gemm_intern_zref( :, : ) = data_gemm_intern( :, : );

fig = figure;
orient( fig, 'portrait' );
%set(gcf,'Position',[0 0 2000 900]);
set(gcf,'PaperUnits', 'inches');
%set(gcf,'PaperSize', [16 12.4]);
%set(gcf,'PaperPosition', [0 0 16 12.4]);
set(gcf,'PaperSize', [14 11.0]);
set(gcf,'PaperPosition', [0 0 14 11.0]);
%set(gcf,'PaperPositionMode','auto');
set(gcf,'PaperPositionMode','manual');
set(gcf,'PaperOrientation','portrait');

fprintf( 'Plotting... ' );

for dti = 1:n_combos
%for dti = 1:1

	% Grab the current datatype combination.
	combo = dt_combos( dti, : );

	%str = sprintf( 'Plotting %d: %s', dti, combo ); disp(str);
	fprintf( '%d (%s) ', dti, combo );

	if combo(4) == 's'
		data_gemm_ref( :, : ) = data_gemm_intern_sref( :, : );
		refch = 's';
	else %if combo(4) == 'd'
		data_gemm_ref( :, : ) = data_gemm_intern_dref( :, : );
		refch = 'd';
	end

	if ( combo(1) == 'c' || combo(1) == 'z' ) && ...
	   ( combo(2) == 'c' || combo(2) == 'z' ) && ...
	   ( combo(3) == 'c' || combo(3) == 'z' )
		if combo(4) == 's'
			data_gemm_ref( :, : ) = data_gemm_intern_cref( :, : );
			refch = 'c';
		else %if combo(4) == 'd'
			data_gemm_ref( :, : ) = data_gemm_intern_zref( :, : );
			refch = 'z';
		end
	end

	% Construct filenames for the data files from templates.
	file_intern_w  = sprintf( filetemp_intern, dirpath_w,  thr_str, combo );
	file_intern_wo = sprintf( filetemp_intern, dirpath_wo, thr_str, combo );
	file_ad_hoc    = sprintf( filetemp_ad_hoc, dirpath_w,  thr_str, combo );

	% Load the data files.
	%str = sprintf( '  Loading %s', file_intern_w ); disp(str);
	run( file_intern_w )
	data_gemm_intern_w( :, : ) = data_gemm_intern( :, : );

	%str = sprintf( '  Loading %s', file_intern_wo ); disp(str);
	run( file_intern_wo )
	data_gemm_intern_wo( :, : ) = data_gemm_intern( :, : );

	%str = sprintf( '  Loading %s', file_ad_hoc ); disp(str);
	run( file_ad_hoc )

	% Plot the result.
	plot_gemm_perf( combo, ...
	                data_gemm_ref, ...
	                data_gemm_intern_w, ...
	                data_gemm_intern_wo, ...
	                data_gemm_ad_hoc, ...
	                refch, ...
	                nth, ...
	                4, 4, ...
	                cfreq, ...
	                dflopspercycle, ...
	                dti );

end

fprintf( '\n' );

%if 0
%set(gcf,'Position',[0 0 2000 900]);
%set(gcf,'PaperUnits', 'inches');
%set(gcf,'PaperSize', [48 22]);
%set(gcf,'PaperPosition', [0 0 48 22]);
%%set(gcf,'PaperPositionMode','auto');
%set(gcf,'PaperPositionMode','manual');
%set(gcf,'PaperOrientation','landscape');
%end

outfile = sprintf( '%s/gemm_%s_%s', dirpath_out, mdcase, arch_str );

print(gcf, outfile,'-bestfit','-dpdf');
%print(gcf, 'gemm_md','-fillpage','-dpdf');

