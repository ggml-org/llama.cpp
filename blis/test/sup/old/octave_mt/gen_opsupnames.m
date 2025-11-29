function [ r_val1, r_val2 ] = gen_opsupnames( ops, stor, smalldims, ldim, pack )

nops = size( ops, 1 );

smallm = smalldims( 1 );
smalln = smalldims( 2 );
smallk = smalldims( 3 );

i = 1;

for io = 1:nops

	op = ops( io, : );

	% sprintf'ing directly into an array of strings, as in:
	%   
	%   opsupnames( i+0, : ) = sprintf( '%s_%s_m%dnpkp_%s_%s', ... );
	%
	% doesn't work when the string lengths as they would if any of the constant
	% dimensions is greater than 9.
	str0 = sprintf( '%s_%s_m%dnpkp_%s_%s',  op, stor, smallm,         ldim, pack );
	str1 = sprintf( '%s_%s_mpn%dkp_%s_%s',  op, stor, smalln,         ldim, pack );
	str2 = sprintf( '%s_%s_mpnpk%d_%s_%s',  op, stor, smallk,         ldim, pack );
	str3 = sprintf( '%s_%s_mpn%dk%d_%s_%s', op, stor, smalln, smallk, ldim, pack );
	str4 = sprintf( '%s_%s_m%dnpk%d_%s_%s', op, stor, smallm, smallk, ldim, pack );
	str5 = sprintf( '%s_%s_m%dn%dkp_%s_%s', op, stor, smallm, smalln, ldim, pack );
	str6 = sprintf( '%s_%s_mpnpkp_%s_%s',   op, stor,                 ldim, pack );

	opsupnames( i+0, : ) = sprintf( '%-31s', str0 );
	opsupnames( i+1, : ) = sprintf( '%-31s', str1 );
	opsupnames( i+2, : ) = sprintf( '%-31s', str2 );
	opsupnames( i+3, : ) = sprintf( '%-31s', str3 );
	opsupnames( i+4, : ) = sprintf( '%-31s', str4 );
	opsupnames( i+5, : ) = sprintf( '%-31s', str5 );
	opsupnames( i+6, : ) = sprintf( '%-31s', str6 );

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

