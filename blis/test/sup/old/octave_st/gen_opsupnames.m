function [ r_val1, r_val2 ] = gen_opsupnames( ops, stor, smalldims, ldim, pack )

nops = size( ops, 1 );

smallm = smalldims( 1 );
smalln = smalldims( 2 );
smallk = smalldims( 3 );

i = 1;

for io = 1:nops

	op = ops( io, : );

	opsupnames( i+0, : ) = sprintf( '%s_%s_m%dnpkp_%s_%s',  op, stor, smallm,         ldim, pack );
	opsupnames( i+1, : ) = sprintf( '%s_%s_mpn%dkp_%s_%s',  op, stor, smalln,         ldim, pack );
	opsupnames( i+2, : ) = sprintf( '%s_%s_mpnpk%d_%s_%s',  op, stor, smallk,         ldim, pack );
	opsupnames( i+3, : ) = sprintf( '%s_%s_mpn%dk%d_%s_%s', op, stor, smalln, smallk, ldim, pack );
	opsupnames( i+4, : ) = sprintf( '%s_%s_m%dnpk%d_%s_%s', op, stor, smallm, smallk, ldim, pack );
	opsupnames( i+5, : ) = sprintf( '%s_%s_m%dn%dkp_%s_%s', op, stor, smallm, smalln, ldim, pack );
	opsupnames( i+6, : ) = sprintf( '%s_%s_mpnpkp_%s_%s',   op, stor,                 ldim, pack );

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

