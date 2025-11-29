function [ r_val1, r_val2 ] = gen_opsupnames( ops, stor, smalldims )

nops = size( ops, 1 );

smallm = smalldims( 1 );
smalln = smalldims( 2 );
smallk = smalldims( 3 );

i = 1;

for io = 1:nops

	op = ops( io, : );

	opsupnames( i+0, : ) = sprintf( '%s_%s_m%dnpkp', op, stor, smallm );
	opsupnames( i+1, : ) = sprintf( '%s_%s_mpn%dkp', op, stor, smalln );
	opsupnames( i+2, : ) = sprintf( '%s_%s_mpnpk%d', op, stor, smallk );
	opsupnames( i+3, : ) = sprintf( '%s_%s_mpn%dk%d', op, stor, smalln, smallk );
	opsupnames( i+4, : ) = sprintf( '%s_%s_m%dnpk%d', op, stor, smallm, smallk );
	opsupnames( i+5, : ) = sprintf( '%s_%s_m%dn%dkp', op, stor, smallm, smalln );
	opsupnames( i+6, : ) = sprintf( '%s_%s_mpnpkp', op, stor );

	opnames( i+0, : ) = sprintf( '%s', op );
	opnames( i+1, : ) = sprintf( '%s', op );
	opnames( i+2, : ) = sprintf( '%s', op );
	opnames( i+3, : ) = sprintf( '%s', op );
	opnames( i+4, : ) = sprintf( '%s', op );
	opnames( i+5, : ) = sprintf( '%s', op );
	opnames( i+6, : ) = sprintf( '%s', op );

	i = i + 7;
end

r_val1 = opsupnames;
r_val2 = opnames;

