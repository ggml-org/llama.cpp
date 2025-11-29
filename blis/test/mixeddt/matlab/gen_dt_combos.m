function r_val = gen_dt_combos()

dt_chars = [ 's' 'd' 'c' 'z' ];
pr_chars = [ 's' 'd' ];

if 0
i = 1;
for dtc = dt_chars
	for dta = dt_chars
		for dtb = dt_chars
			for pr = pr_chars
				dt_combos( i, : ) = sprintf( '%c%c%c%c', dtc, dta, dtb, pr );
				i = i + 1;
			end
		end
	end
end
end

%n_combos = size(temp,1);

if 1
dt_combos(   1, : ) = 'ssss';
dt_combos(   2, : ) = 'ssds';
dt_combos(   3, : ) = 'sscs';
dt_combos(   4, : ) = 'sszs';
dt_combos(   5, : ) = 'sdss';
dt_combos(   6, : ) = 'sdds';
dt_combos(   7, : ) = 'sdcs';
dt_combos(   8, : ) = 'sdzs';
dt_combos(   9, : ) = 'sssd';
dt_combos(  10, : ) = 'ssdd';
dt_combos(  11, : ) = 'sscd';
dt_combos(  12, : ) = 'sszd';
dt_combos(  13, : ) = 'sdsd';
dt_combos(  14, : ) = 'sddd';
dt_combos(  15, : ) = 'sdcd';
dt_combos(  16, : ) = 'sdzd';

dt_combos(  17, : ) = 'scss';
dt_combos(  18, : ) = 'scds';
dt_combos(  19, : ) = 'sccs';
dt_combos(  20, : ) = 'sczs';
dt_combos(  21, : ) = 'szss';
dt_combos(  22, : ) = 'szds';
dt_combos(  23, : ) = 'szcs';
dt_combos(  24, : ) = 'szzs';
dt_combos(  25, : ) = 'scsd';
dt_combos(  26, : ) = 'scdd';
dt_combos(  27, : ) = 'sccd';
dt_combos(  28, : ) = 'sczd';
dt_combos(  29, : ) = 'szsd';
dt_combos(  30, : ) = 'szdd';
dt_combos(  31, : ) = 'szcd';
dt_combos(  32, : ) = 'szzd';

dt_combos(  33, : ) = 'dsss';
dt_combos(  34, : ) = 'dsds';
dt_combos(  35, : ) = 'dscs';
dt_combos(  36, : ) = 'dszs';
dt_combos(  37, : ) = 'ddss';
dt_combos(  38, : ) = 'ddds';
dt_combos(  39, : ) = 'ddcs';
dt_combos(  40, : ) = 'ddzs';
dt_combos(  41, : ) = 'dssd';
dt_combos(  42, : ) = 'dsdd';
dt_combos(  43, : ) = 'dscd';
dt_combos(  44, : ) = 'dszd';
dt_combos(  45, : ) = 'ddsd';
dt_combos(  46, : ) = 'dddd';
dt_combos(  47, : ) = 'ddcd';
dt_combos(  48, : ) = 'ddzd';

dt_combos(  49, : ) = 'dcss';
dt_combos(  50, : ) = 'dcds';
dt_combos(  51, : ) = 'dccs';
dt_combos(  52, : ) = 'dczs';
dt_combos(  53, : ) = 'dzss';
dt_combos(  54, : ) = 'dzds';
dt_combos(  55, : ) = 'dzcs';
dt_combos(  56, : ) = 'dzzs';
dt_combos(  57, : ) = 'dcsd';
dt_combos(  58, : ) = 'dcdd';
dt_combos(  59, : ) = 'dccd';
dt_combos(  60, : ) = 'dczd';
dt_combos(  61, : ) = 'dzsd';
dt_combos(  62, : ) = 'dzdd';
dt_combos(  63, : ) = 'dzcd';
dt_combos(  64, : ) = 'dzzd';

dt_combos(  65, : ) = 'csss';
dt_combos(  66, : ) = 'csds';
dt_combos(  67, : ) = 'cscs';
dt_combos(  68, : ) = 'cszs';
dt_combos(  69, : ) = 'cdss';
dt_combos(  70, : ) = 'cdds';
dt_combos(  71, : ) = 'cdcs';
dt_combos(  72, : ) = 'cdzs';
dt_combos(  73, : ) = 'cssd';
dt_combos(  74, : ) = 'csdd';
dt_combos(  75, : ) = 'cscd';
dt_combos(  76, : ) = 'cszd';
dt_combos(  77, : ) = 'cdsd';
dt_combos(  78, : ) = 'cddd';
dt_combos(  79, : ) = 'cdcd';
dt_combos(  80, : ) = 'cdzd';

dt_combos(  81, : ) = 'ccss';
dt_combos(  82, : ) = 'ccds';
dt_combos(  83, : ) = 'cccs';
dt_combos(  84, : ) = 'cczs';
dt_combos(  85, : ) = 'czss';
dt_combos(  86, : ) = 'czds';
dt_combos(  87, : ) = 'czcs';
dt_combos(  88, : ) = 'czzs';
dt_combos(  89, : ) = 'ccsd';
dt_combos(  90, : ) = 'ccdd';
dt_combos(  91, : ) = 'cccd';
dt_combos(  92, : ) = 'cczd';
dt_combos(  93, : ) = 'czsd';
dt_combos(  94, : ) = 'czdd';
dt_combos(  95, : ) = 'czcd';
dt_combos(  96, : ) = 'czzd';

dt_combos(  97, : ) = 'zsss';
dt_combos(  98, : ) = 'zsds';
dt_combos(  99, : ) = 'zscs';
dt_combos( 100, : ) = 'zszs';
dt_combos( 101, : ) = 'zdss';
dt_combos( 102, : ) = 'zdds';
dt_combos( 103, : ) = 'zdcs';
dt_combos( 104, : ) = 'zdzs';
dt_combos( 105, : ) = 'zssd';
dt_combos( 106, : ) = 'zsdd';
dt_combos( 107, : ) = 'zscd';
dt_combos( 108, : ) = 'zszd';
dt_combos( 109, : ) = 'zdsd';
dt_combos( 110, : ) = 'zddd';
dt_combos( 111, : ) = 'zdcd';
dt_combos( 112, : ) = 'zdzd';

dt_combos( 113, : ) = 'zcss';
dt_combos( 114, : ) = 'zcds';
dt_combos( 115, : ) = 'zccs';
dt_combos( 116, : ) = 'zczs';
dt_combos( 117, : ) = 'zzss';
dt_combos( 118, : ) = 'zzds';
dt_combos( 119, : ) = 'zzcs';
dt_combos( 120, : ) = 'zzzs';
dt_combos( 121, : ) = 'zcsd';
dt_combos( 122, : ) = 'zcdd';
dt_combos( 123, : ) = 'zccd';
dt_combos( 124, : ) = 'zczd';
dt_combos( 125, : ) = 'zzsd';
dt_combos( 126, : ) = 'zzdd';
dt_combos( 127, : ) = 'zzcd';
dt_combos( 128, : ) = 'zzzd';
end




r_val = dt_combos;

end
