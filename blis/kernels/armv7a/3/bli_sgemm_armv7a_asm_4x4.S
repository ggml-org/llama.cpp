
#define REALNAME bli_sgemm_armv7a_ker_4x4

#define STACKSIZE 256

#define	K		r0
#define	PTR_ALPHA	r1
#define	OLD_A		r2
#define	OLD_B		r3
#define PTR_BETA	[fp, #0 ]
#define OLD_C		[fp, #4 ]
#define OLD_RSC		[fp, #8 ]
#define OLD_CSC		[fp, #12 ]
#define AUX		[fp, #16 ]

/******************************************************
* [fp, #-128] - [fp, #-64] is reserved
* for store and restore of floating point
* register
*******************************************************/

#define L	r2

#define	AO	r5
#define	BO	r6

#define	CO1	r7
#define	CO2	r8
#define	CO3	r9
#define	CO4	r12


#define A_PRE	96
#define B_PRE	96
#define C_PRE	0

/**************************************************************************************
* Macro definitions
**************************************************************************************/

.macro INIT4x4

	vsub.f32		s16 , s16 , s16
	vmov.f32		s17, s16
	vmov.f32		s18, s16
	vmov.f32		s19, s16
	vmov.f32		s20, s16
	vmov.f32		s21, s16
	vmov.f32		s22, s16
	vmov.f32		s23, s16
	vmov.f32		s24, s16
	vmov.f32		s25, s16
	vmov.f32		s26, s16
	vmov.f32		s27, s16
	vmov.f32		s28, s16
	vmov.f32		s29, s16
	vmov.f32		s30, s16
	vmov.f32		s31, s16

.endm

.macro KERNEL4x4_I

	pld	[ AO , #A_PRE ]
	fldmias AO!, { s0 - s1 }
	pld	[ BO , #B_PRE ]
	fldmias BO!, { s8 - s9 }

	fmuls	s16  , s0,  s8
	fldmias AO!, { s2 - s3 }
	fmuls	s17  , s1,  s8
	fmuls	s18  , s2,  s8
	fldmias BO!, { s10 - s11 }
	fmuls	s19  , s3,  s8

	fmuls	s20  , s0,  s9
	fldmias AO!, { s4 - s5 }
	fmuls	s21  , s1,  s9
	fmuls	s22  , s2,  s9
	fldmias AO!, { s6 - s7 }
	fmuls	s23  , s3,  s9

	fmuls	s24  , s0,  s10
	fldmias BO!, { s12 - s13 }
	fmuls	s25  , s1,  s10
	fmuls	s26  , s2,  s10
	fldmias BO!, { s14 - s15 }
	fmuls	s27  , s3,  s10

	fmuls	s28  , s0,  s11
	fmuls	s29  , s1,  s11
	fmuls	s30  , s2,  s11
	fmuls	s31  , s3,  s11

.endm


.macro KERNEL4x4_M2

	pld	[ AO , #A_PRE ]
	fmacs	s16  , s4,  s12
	fmacs	s17  , s5,  s12
	fldmias AO!, { s0 - s3 }
	fmacs	s18  , s6,  s12
	pld	[ BO , #B_PRE ]
	fmacs	s19  , s7,  s12

	fmacs	s20  , s4,  s13
	fldmias BO!, { s8 - s11 }
	fmacs	s21  , s5,  s13
	fmacs	s22  , s6,  s13
	//fldmias AO!, { s2 - s3 }
	fmacs	s23  , s7,  s13

	fmacs	s24  , s4,  s14
	//fldmias BO!, { s10 - s11 }
	fmacs	s25  , s5,  s14
	fmacs	s26  , s6,  s14
	fmacs	s27  , s7,  s14

	fmacs	s28  , s4,  s15
	fmacs	s29  , s5,  s15
	fmacs	s30  , s6,  s15
	fmacs	s31  , s7,  s15

.endm


.macro KERNEL4x4_M1

	fmacs	s16  , s0,  s8
	fldmias AO!, { s4 - s7 }
	fmacs	s17  , s1,  s8
	fmacs	s18  , s2,  s8
	fldmias BO!, { s12 - s15 }
	//fldmias AO!, { s6 - s7 }
	fmacs	s19  , s3,  s8

	fmacs	s20  , s0,  s9
	fmacs	s21  , s1,  s9
	fmacs	s22  , s2,  s9
	//fldmias BO!, { s14 - s15 }
	fmacs	s23  , s3,  s9

	fmacs	s24  , s0,  s10
	fmacs	s25  , s1,  s10
	fmacs	s26  , s2,  s10
	fmacs	s27  , s3,  s10

	fmacs	s28  , s0,  s11
	fmacs	s29  , s1,  s11
	fmacs	s30  , s2,  s11
	fmacs	s31  , s3,  s11

.endm



.macro KERNEL4x4_E

	fmacs	s16  , s4,  s12
	fmacs	s17  , s5,  s12
	fmacs	s18  , s6,  s12
	fmacs	s19  , s7,  s12

	fmacs	s20  , s4,  s13
	fmacs	s21  , s5,  s13
	fmacs	s22  , s6,  s13
	fmacs	s23  , s7,  s13

	fmacs	s24  , s4,  s14
	fmacs	s25  , s5,  s14
	fmacs	s26  , s6,  s14
	fmacs	s27  , s7,  s14

	fmacs	s28  , s4,  s15
	fmacs	s29  , s5,  s15
	fmacs	s30  , s6,  s15
	fmacs	s31  , s7,  s15

.endm




.macro KERNEL4x4_SUB

	flds	s8 , [ BO ]

	flds	s0 , [ AO ]
	flds	s1 , [ AO, #4 ]

	fmacs	s16  , s0,  s8
	flds	s2 , [ AO, #8 ]
	fmacs	s17  , s1,  s8
	flds	s3 , [ AO, #12 ]
	fmacs	s18  , s2,  s8
	flds	s9 , [ BO, #4 ]
	fmacs	s19  , s3,  s8

	flds	s10, [ BO, #8 ]
	fmacs	s20  , s0,  s9
	flds	s11, [ BO, #12 ]
	fmacs	s21  , s1,  s9
	fmacs	s22  , s2,  s9
	fmacs	s23  , s3,  s9

	fmacs	s24  , s0,  s10
	fmacs	s25  , s1,  s10
	fmacs	s26  , s2,  s10
	fmacs	s27  , s3,  s10

	fmacs	s28  , s0,  s11
	fmacs	s29  , s1,  s11
	add	AO , AO, #16
	fmacs	s30  , s2,  s11
	add	BO , BO, #16
	fmacs	s31  , s3,  s11

.endm


.macro SAVE4x4

	ldr	r3, OLD_RSC				// Row stride size
	lsl	r3, r3, #2				// multiply with size of float

	flds	s0, [ PTR_ALPHA	]			// load alpha
	ldr	r4, PTR_BETA
	flds	s1, [ r4 ]				// load beta

//-----------------------------------------------------------
	mov	r2, CO1					// save pointer
	mov	r4, CO2					// save pointer
	flds	s8, [ CO1 ]				// load value from C
	flds	s12, [ CO2 ]				// load value from C
	fmuls	s8, s8, s1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacs	s8, s0, s16				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	flds	s9, [ CO1 ]				// load value from C
	flds	s13, [ CO2 ]				// load value from C
	fmuls	s9, s9, s1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacs	s9, s0, s17				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	flds	s10, [ CO1 ]				// load value from C
	flds	s14, [ CO2 ]				// load value from C
	fmuls	s10, s10, s1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacs	s10, s0, s18				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	flds	s11, [ CO1 ]				// load value from C
	flds	s15, [ CO2 ]				// load value from C
	fmuls	s11, s11, s1				// multiply with beta
	mov	CO1, r2					// restore pointer
	fmacs	s11, s0, s19				// multiply sum with alpha and add to value of C	
	mov	CO2, r4					// restore pointer
	
	fsts	s8, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fsts	s9, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fsts	s10, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fsts	s11, [ CO1 ]				// store value in C

//-----------------------------------------------------------
	mov	r2, CO3					// save pointer
	flds	s8, [ CO3 ]				// load value from C
	fmuls	s12, s12, s1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacs	s12, s0, s20				// multiply sum with alpha and add to value of C	

	flds	s9, [ CO3 ]				// load value from C
	fmuls	s13, s13, s1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacs	s13, s0, s21				// multiply sum with alpha and add to value of C	

	flds	s10, [ CO3 ]				// load value from C
	fmuls	s14, s14, s1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacs	s14, s0, s22				// multiply sum with alpha and add to value of C	

	flds	s11, [ CO3 ]				// load value from C
	fmuls	s15, s15, s1				// multiply with beta
	mov	CO3, r2					// restore pointer
	fmacs	s15, s0, s23				// multiply sum with alpha and add to value of C	
	
	fsts	s12, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fsts	s13, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fsts	s14, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fsts	s15, [ CO2 ]				// store value in C

//-----------------------------------------------------------
	mov	r4, CO4					// save pointer
	flds	s12, [ CO4 ]				// load value from C
	fmuls	s8, s8, s1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacs	s8, s0, s24				// multiply sum with alpha and add to value of C	

	flds	s13, [ CO4 ]				// load value from C
	fmuls	s9, s9, s1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacs	s9, s0, s25				// multiply sum with alpha and add to value of C	

	flds	s14, [ CO4 ]				// load value from C
	fmuls	s10, s10, s1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacs	s10, s0, s26				// multiply sum with alpha and add to value of C	

	flds	s15, [ CO4 ]				// load value from C
	fmuls	s11, s11, s1				// multiply with beta
	mov	CO4, r4					// restore pointer
	fmacs	s11, s0, s27				// multiply sum with alpha and add to value of C	
	

//-----------------------------------------------------------
	fsts	s8, [ CO3 ]				// store value in C
	fmuls	s12, s12, s1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacs	s12, s0, s28				// multiply sum with alpha and add to value of C	

	fsts	s9, [ CO3 ]				// store value in C
	fmuls	s13, s13, s1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacs	s13, s0, s29				// multiply sum with alpha and add to value of C	

	fsts	s10, [ CO3 ]				// store value in C
	fmuls	s14, s14, s1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacs	s14, s0, s30				// multiply sum with alpha and add to value of C	

	fsts	s11, [ CO3 ]				// store value in C
	fmuls	s15, s15, s1				// multiply with beta
	fsts	s12, [ CO4 ]				// store value in C
	fmacs	s15, s0, s31				// multiply sum with alpha and add to value of C	
	
	add	CO4 , CO4, r3				// compute next pointer
	fsts	s13, [ CO4 ]				// store value in C
	add	CO4 , CO4, r3				// compute next pointer
	fsts	s14, [ CO4 ]				// store value in C
	add	CO4 , CO4, r3				// compute next pointer
	fsts	s15, [ CO4 ]				// store value in C

.endm


/**************************************************************************************
* End of macro definitions
**************************************************************************************/

        .arm             	
        .global REALNAME 	
        .func   REALNAME 	

REALNAME:

	push	{r4 - r9, fp}					// save register
	add	fp, sp, #28					// add number of saved register multiplied by size of int
	sub	sp, sp, #STACKSIZE				// reserve stack

	mov	AO, OLD_A					// pointer matrix A
	mov	BO, OLD_B					// pointer matrix B

	sub	r3, fp, #128
	vstm	r3, { s8 - s31 } 				// store floating point registers

	ldr	r2, OLD_C					// pointer matrix C
	ldr	r3, OLD_CSC					// Col stride size of C
	lsl	r3, r3, #2					// multiply with size of float

	mov	CO1, r2						// first line of C
	add	CO2, CO1, r3					// second line of C
	add	CO3, CO2, r3					// third line of C
	add	CO4, CO3, r3					// fourth line of C

	pld	[ CO1, #C_PRE ]					// prefetch the lines of C
	pld	[ CO2, #C_PRE ]					// prefetch the lines of C
	pld	[ CO3, #C_PRE ]					// prefetch the lines of C
	pld	[ CO3, #C_PRE ]					// prefetch the lines of C

sgemm_kernel_L4_M4_20:

	asrs	L , K, #3					// L = K / 8
	cmp	L , #2
	blt	sgemm_kernel_L4_M4_32

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	subs	L, L, #2
	ble	sgemm_kernel_L4_M4_22a
	.align 5

sgemm_kernel_L4_M4_22:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	subs	L, L, #1
	bgt	sgemm_kernel_L4_M4_22

sgemm_kernel_L4_M4_22a:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	b	 sgemm_kernel_L4_M4_44

sgemm_kernel_L4_M4_32:

	tst	L, #1
	ble	sgemm_kernel_L4_M4_40

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	b	 sgemm_kernel_L4_M4_44

sgemm_kernel_L4_M4_40:

	INIT4x4

sgemm_kernel_L4_M4_44:

	ands	L , K, #7					// L = K % 8
	ble	sgemm_kernel_L4_M4_100

sgemm_kernel_L4_M4_46:

	KERNEL4x4_SUB

	subs	L, L, #1
	bne	sgemm_kernel_L4_M4_46
	
sgemm_kernel_L4_M4_100:

	SAVE4x4

sgemm_kernel_L999:

	sub	r3, fp, #128
	vldm	r3, { s8 - s31 }				// restore floating point registers

	sub	sp, fp, #28
	pop	{r4 - r9, fp}
	bx	lr

