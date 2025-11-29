function r_val = gen_prec_combos( mdcase )

dt_chars = [ 's' 'd' 'c' 'z' ];
pr_chars = [ 's' 'd' ];
dm_chars = [ 'r' 'c' ];

dmc = mdcase( 1 );
dma = mdcase( 2 );
dmb = mdcase( 3 );

if 0
pr_combos(  1, : ) = 'ssss';
pr_combos(  2, : ) = 'ssds';
pr_combos(  3, : ) = 'sdss';
pr_combos(  4, : ) = 'sdds';
pr_combos(  5, : ) = 'dsss';
pr_combos(  6, : ) = 'dsds';
pr_combos(  7, : ) = 'ddss';
pr_combos(  8, : ) = 'ddds';
pr_combos(  9, : ) = 'dddd';
pr_combos( 10, : ) = 'ddsd';
pr_combos( 11, : ) = 'dsdd';
pr_combos( 12, : ) = 'dssd';
pr_combos( 13, : ) = 'sddd';
pr_combos( 14, : ) = 'sdsd';
pr_combos( 15, : ) = 'ssdd';
pr_combos( 16, : ) = 'sssd';
end

pr_combos(  1, : ) = 'ssss';
pr_combos(  2, : ) = 'ssds';
pr_combos(  3, : ) = 'dddd';
pr_combos(  4, : ) = 'ddsd';

pr_combos(  5, : ) = 'sdss';
pr_combos(  6, : ) = 'sdds';
pr_combos(  7, : ) = 'dsdd';
pr_combos(  8, : ) = 'dssd';

pr_combos(  9, : ) = 'dsss';
pr_combos( 10, : ) = 'dsds';
pr_combos( 11, : ) = 'sddd';
pr_combos( 12, : ) = 'sdsd';

pr_combos( 13, : ) = 'ddss';
pr_combos( 14, : ) = 'ddds';
pr_combos( 15, : ) = 'ssdd';
pr_combos( 16, : ) = 'sssd';

for i = 1:16

	pr_combo = pr_combos( i, : );

	%str = sprintf( '%s', pr_combo ); disp(str);

	prc = pr_combo( 1 );
	pra = pr_combo( 2 );
	prb = pr_combo( 3 );
	pr  = pr_combo( 4 );

	dtc = prec_dom_to_dt( prc, dmc );
	dta = prec_dom_to_dt( pra, dma );
	dtb = prec_dom_to_dt( prb, dmb );

	dt_combos( i, : ) = sprintf( '%c%c%c%c', dtc, dta, dtb, pr );

end


%if 0
%i = 1;
%pr = 's';
%for prc = pr_chars
%	for pra = pr_chars
%		for prb = pr_chars
%			dtc = prec_dom_to_dt( prc, dmc );
%			dta = prec_dom_to_dt( pra, dma );
%			dtb = prec_dom_to_dt( prb, dmb );
%			dt_combos( i, : ) = sprintf( '%c%c%c%c', dtc, dta, dtb, pr );
%			i = i + 1;
%		end
%	end
%end
%
%pr = 'd';
%for prc = flip( pr_chars )
%	for pra = flip( pr_chars )
%		for prb = flip( pr_chars )
%			dtc = prec_dom_to_dt( prc, dmc );
%			dta = prec_dom_to_dt( pra, dma );
%			dtb = prec_dom_to_dt( prb, dmb );
%			dt_combos( i, : ) = sprintf( '%c%c%c%c', dtc, dta, dtb, pr );
%			i = i + 1;
%		end
%	end
%end
%end

r_val = dt_combos;

end
