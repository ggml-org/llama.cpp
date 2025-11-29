function r_val = gen_opnames( ops, dts )

nops = size( ops, 1 );
ndts = size( dts, 2 );

i = 1;

for id = 1:ndts

	dt = dts( id );

	for io = 1:nops

		op = ops( io, : );

		opnames( i, : ) = sprintf( '%c%s', dt, op );
		i = i + 1;
	end
end

r_val = opnames;

