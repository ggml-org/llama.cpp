function data = read_data ...
     ( ...
       filetemp, ...
       dirpath, ...
       var_templ, ...
       thr_str, ...
       opname, ...
       impl_str  ...
     )

% Construct the full filepath for the data file from the template.
filepath = sprintf( filetemp, dirpath, thr_str, opname );

% Attempt to open the file.
fid = fopen( filepath );

if fid == -1
	% If the file was not opened successfully, it's probably because
	% the file is missing altogether. In these sitautions, we set the
	% first element of the data to -1, which will be a signal to the
	% plotting function to omit this curve from the graph.
	data(1,1) = -1;
else
	% If the file was opened successfully, we assume that it either
	% contains valid data, or it adheres to the "missing data" format
	% whereby the (1,1) element contains -1. In either case, we can
	% process it normally and we begin by closing the file since we
	% don't need the file descriptor.
	fclose( fid );

	% Load the data file.
	run( filepath )

	% Construct variable names for the variables in the data file.
	% Examples: data_st_dgemm_asm_blis
	%           data_1s_zherk_vendor
	var_name = sprintf( var_templ, thr_str, opname, impl_str );

	% Use eval() to instantiate the variable names constructed above,
	% copying each to a simplified name.
	data = eval( var_name );
end

% Return the 'data' variable.
