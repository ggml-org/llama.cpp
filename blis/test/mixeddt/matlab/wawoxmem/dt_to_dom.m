function r_val = dt_to_dom( dt )

	dom = 'rrr';

	for ch = 1:3

		if dt(ch) == 'c' || dt(ch) == 'z'
			dom(ch) = 'c';
		end
	end

	r_val = dom;
end
