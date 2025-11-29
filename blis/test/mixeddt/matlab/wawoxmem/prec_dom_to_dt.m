function r_val = prec_dom_to_dt( pc, dc )

	if dc == 'r'
		if pc == 's'
			r_val = 's';
		else
			r_val = 'd';
		end
	else
		if pc == 's'
			r_val = 'c';
		else
			r_val = 'z';
		end
	end

end
