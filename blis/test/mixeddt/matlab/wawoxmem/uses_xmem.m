function r_val = uses_xmem( dt_str )

	a = dt_str(1);
	b = dt_str(4);

	a_prec = 'd';
	b_prec = 'd';

	if ( a == 's' || a == 'c' )
		a_prec = 's';
	end

	if ( b == 's' || b == 'c' )
		b_prec = 's';
	end

	dom_str = dt_to_dom( dt_str );

	r_val = 0;

	if ( a_prec ~= b_prec )

		r_val = 1;

	elseif ( strcmp( dom_str, 'crr' ) )

		r_val = 1;

	elseif ( strcmp( dom_str, 'crc' ) )

		r_val = 1;
	end

end
