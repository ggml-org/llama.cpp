
#define REALNAME bli_cgemm_armv7a_ker_2x2

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


#define A_PRE	96
#define B_PRE	96
#define C_PRE	0

/**************************************************************************************
* Macro definitions
**************************************************************************************/

#define FMAC_BR	fnmacs
#define FMAC_BI	fmacs

#define NN 1

#if defined(NN) || defined(NT) || defined(TN) || defined(TT) 

	#define	FADD_R	fsubs
	#define	FADD_I	fadds

	#define	FMAC_R1	fnmacs
	#define	FMAC_R2	fnmacs
	#define	FMAC_I1	fmacs
	#define	FMAC_I2	fnmacs

#elif defined(CN) || defined(CT)

	#define	FADD_R	fadds
	#define	FADD_I	fsubs

	#define	FMAC_R1	fmacs
	#define	FMAC_R2	fmacs
	#define	FMAC_I1	fnmacs
	#define	FMAC_I2	fmacs

#elif defined(NC) || defined(TC)

	#define	FADD_R	fadds
	#define	FADD_I	fsubs

	#define	FMAC_R1	fmacs
	#define	FMAC_R2	fnmacs
	#define	FMAC_I1	fmacs
	#define	FMAC_I2	fmacs

#else

	#define	FADD_R  fsubs
	#define	FADD_I	fadds

	#define	FMAC_R1	fnmacs
	#define	FMAC_R2	fmacs
	#define	FMAC_I1	fnmacs
	#define	FMAC_I2	fnmacs

#endif



.macro INIT2x2

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

.macro KERNEL2x2_I
	pld	[ AO , #A_PRE ]
	pld	[ BO , #B_PRE ]
	flds	s0 , [ AO ]
	flds	s1 , [ AO, #4 ]
	flds	s8 , [ BO ]
	flds	s9 , [ BO, #4 ]

	fmuls	s16  , s0,  s8
	flds	s2 , [ AO, #8 ]
	fmuls	s24  , s1,  s9
	flds	s3 , [ AO, #12 ]
	fmuls	s17  , s0,  s9
	flds	s10, [ BO, #8 ]
	fmuls	s25  , s1,  s8

	flds	s11, [ BO, #12 ]
	fmuls	s18  , s2,  s8
	add	BO , BO, #16
	fmuls	s26  , s3,  s9
	add	AO , AO, #16
	fmuls	s19  , s2,  s9
	pld	[ BO , #B_PRE ]
	fmuls	s27  , s3,  s8

	pld	[ AO , #A_PRE ]
	fmuls	s20  , s0,  s10
	flds	s4 , [ AO, #0 ]
	fmuls	s28  , s1,  s11
	flds	s5 , [ AO, #4 ]
	fmuls	s21  , s0,  s11
	flds	s12, [ BO ]
	fmuls	s29  , s1,  s10

	flds	s13, [ BO, #4 ]
	fmuls	s22  , s2,  s10
	flds	s6 , [ AO, #8 ]
	fmuls	s30  , s3,  s11
	flds	s7 , [ AO, #12 ]
	fmuls	s23  , s2,  s11
	flds	s14, [ BO, #8 ]
	fmuls	s31  , s3,  s10
	flds	s15, [ BO, #12 ]

	add	BO , BO, #16
	add	AO , AO, #16
.endm



.macro KERNEL2x2_M1
	pld	[ AO , #A_PRE ]

	fmacs	s16  , s0,  s8
	pld	[ BO , #B_PRE ]
	fmacs	s24  , s1,  s9
	flds	s4 , [ AO, #0 ]
	fmacs	s17  , s0,  s9
	flds	s5 , [ AO, #4 ]
	fmacs	s25  , s1,  s8

	flds	s12, [ BO ]
	fmacs	s18  , s2,  s8
	flds	s13, [ BO, #4 ]
	fmacs	s26  , s3,  s9
	flds	s6 , [ AO, #8 ]
	fmacs	s19  , s2,  s9
	flds	s7 , [ AO, #12 ]
	fmacs	s27  , s3,  s8

	fmacs	s20  , s0,  s10
	flds	s14, [ BO, #8 ]
	fmacs	s28  , s1,  s11
	fmacs	s21  , s0,  s11
	flds	s15, [ BO, #12 ]
	fmacs	s29  , s1,  s10

	fmacs	s22  , s2,  s10
	add	BO , BO, #16
	fmacs	s30  , s3,  s11
	fmacs	s23  , s2,  s11
	add	AO , AO, #16
	fmacs	s31  , s3,  s10

.endm

.macro KERNEL2x2_M2

	fmacs	s16  , s4,  s12
	fmacs	s24  , s5,  s13
	flds	s0 , [ AO, #0 ]
	fmacs	s17  , s4,  s13
	flds	s1 , [ AO, #4 ]
	fmacs	s25  , s5,  s12

	fmacs	s18  , s6,  s12
	flds	s8 , [ BO ]
	fmacs	s26  , s7,  s13
	flds	s9 , [ BO, #4 ]
	fmacs	s19  , s6,  s13
	fmacs	s27  , s7,  s12

	flds	s2 , [ AO, #8 ]
	fmacs	s20  , s4,  s14
	flds	s3 , [ AO, #12 ]
	fmacs	s28  , s5,  s15
	fmacs	s21  , s4,  s15
	flds	s10, [ BO, #8 ]
	fmacs	s29  , s5,  s14

	flds	s11, [ BO, #12 ]
	fmacs	s22  , s6,  s14
	fmacs	s30  , s7,  s15
	add	BO , BO, #16
	fmacs	s23  , s6,  s15
	add	AO , AO, #16
	fmacs	s31  , s7,  s14

.endm


.macro KERNEL2x2_E

	fmacs	s16  , s4,  s12
	fmacs	s24  , s5,  s13
	fmacs	s17  , s4,  s13
	fmacs	s25  , s5,  s12

	fmacs	s18  , s6,  s12
	fmacs	s26  , s7,  s13
	fmacs	s19  , s6,  s13
	fmacs	s27  , s7,  s12

	fmacs	s20  , s4,  s14
	fmacs	s28  , s5,  s15
	fmacs	s21  , s4,  s15
	fmacs	s29  , s5,  s14

	fmacs	s22  , s6,  s14
	fmacs	s30  , s7,  s15
	fmacs	s23  , s6,  s15
	fmacs	s31  , s7,  s14

.endm

.macro KERNEL2x2_SUB

	flds	s0 , [ AO ]
	flds	s1 , [ AO, #4 ]
	flds	s8 , [ BO ]
	flds	s9 , [ BO, #4 ]

	fmacs	s16  , s0,  s8
	flds	s2 , [ AO, #8 ]
	fmacs	s24  , s1,  s9
	flds	s3 , [ AO, #12 ]
	fmacs	s17  , s0,  s9
	flds	s10, [ BO, #8 ]
	fmacs	s25  , s1,  s8

	flds	s11, [ BO, #12 ]
	fmacs	s18  , s2,  s8
	fmacs	s26  , s3,  s9
	fmacs	s19  , s2,  s9
	fmacs	s27  , s3,  s8

	fmacs	s20  , s0,  s10
	fmacs	s28  , s1,  s11
	fmacs	s21  , s0,  s11
	fmacs	s29  , s1,  s10

	fmacs	s22  , s2,  s10
	add	BO , BO, #16
	fmacs	s30  , s3,  s11
	fmacs	s23  , s2,  s11
	add	AO , AO, #16
	fmacs	s31  , s3,  s10

.endm




.macro SAVE2x2

        ldr     r3, OLD_RSC                             // Row stride size
        lsl     r3, r3, #3                              // multiply with size of complex float

        flds    s0, [ PTR_ALPHA ]                       // load real part of alpha
        flds    s1, [ PTR_ALPHA, #4 ]                   // load imag part of alpha
        ldr     r4, PTR_BETA
        flds    s2, [ r4 ]                              // load real part of beta
        flds    s3, [ r4, #4  ]                         // load imag part of beta

	// Add/Sub the real and the imag parts
	FADD_R	s16, s24 , s16
	FADD_I  s17, s25 , s17
	FADD_R	s18, s26 , s18
	FADD_I  s19, s27 , s19
	FADD_R	s20, s28 , s20
	FADD_I  s21, s29 , s21
	FADD_R	s22, s30 , s22
	FADD_I  s23, s31 , s23

	mov	r4, CO1					// save pointer
	fldmias CO1, { s4 - s5 }			// read real and imag part from C
	add	CO1, CO1, r3
	
	mov	r2, CO2					// save pointer
	fldmias CO2, { s8 - s9 }			// read real and imag part from C
	add	CO2, CO2, r3

	fmuls	s24, s4, s2				// multiply Beta-real with C-real
	fmuls	s25, s5, s2				// multiply Beta-real with C-imag
	fmuls	s28, s8, s2				// multiply Beta-real with C-real
	fmuls	s29, s9, s2				// multiply Beta-real with C-imag

	FMAC_BR	s24, s3, s5				// multiply beta-imag with C-imag and add
	FMAC_BI	s25, s3, s4				// multiply beta-imag with C-real and add
	FMAC_BR	s28, s3, s9				// multiply beta-imag with C-imag and add
	FMAC_BI	s29, s3, s8				// multiply beta-imag with C-real and add

	FMAC_R1 s24 , s0 , s16
	FMAC_I1 s25 , s0 , s17
	FMAC_R2 s24 , s1 , s17
	FMAC_I2	s25 , s1 , s16

	FMAC_R1 s28 , s0 , s20
	FMAC_I1 s29 , s0 , s21
	FMAC_R2 s28 , s1 , s21
	FMAC_I2	s29 , s1 , s20

	fldmias CO1, { s4 - s5 }			// read real and imag part from C
	fldmias CO2, { s8 - s9 }			// read real and imag part from C

	fmuls	s26, s4, s2				// multiply Beta-real with C-real
	fmuls	s27, s5, s2				// multiply Beta-real with C-imag
	fmuls	s30, s8, s2				// multiply Beta-real with C-real
	fmuls	s31, s9, s2				// multiply Beta-real with C-imag

	FMAC_BR	s26, s3, s5				// multiply beta-imag with C-imag and add
	FMAC_BI	s27, s3, s4				// multiply beta-imag with C-real and add
	FMAC_BR	s30, s3, s9				// multiply beta-imag with C-imag and add
	FMAC_BI	s31, s3, s8				// multiply beta-imag with C-real and add

	FMAC_R1 s26 , s0 , s18
	FMAC_I1 s27 , s0 , s19
	FMAC_R2 s26 , s1 , s19
	FMAC_I2	s27 , s1 , s18

	FMAC_R1 s30, s0 , s22
	FMAC_I1 s31, s0 , s23
	FMAC_R2 s30, s1 , s23
	FMAC_I2	s31, s1 , s22

	mov	CO1, r4					// restore pointer
	mov	CO2, r2					// restore pointer
	fstmias CO1, { s24 - s25 }
	fstmias CO2, { s28 - s29 }
	add	CO1, CO1, r3
	add	CO2, CO2, r3
	fstmias CO1, { s26 - s27 }
	fstmias CO2, { s30 - s31 }


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
	vstm	r3, { s8 - s31} 				// store floating point registers

	ldr	r2, OLD_C					// pointer matrix C
	ldr	r3, OLD_CSC					// Col stride size of C
	lsl	r3, r3, #3					// multiply with size of complex float

	mov	CO1, r2						// first line of C
	add	CO2, CO1, r3					// second line of C

	pld	[ CO1, #C_PRE ]					// prefetch the lines of C
	pld	[ CO2, #C_PRE ]					// prefetch the lines of C

cgemm_kernel_L2_M2_20:

	asrs	L , K, #3					// L = K / 8
	cmp	L , #2
	blt	cgemm_kernel_L2_M2_32

	KERNEL2x2_I
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	subs	L, L, #2
	ble	cgemm_kernel_L2_M2_22a
	.align 5

cgemm_kernel_L2_M2_22:

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	subs	L, L, #1
	bgt	cgemm_kernel_L2_M2_22

cgemm_kernel_L2_M2_22a:

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_E

	b	 cgemm_kernel_L2_M2_44

cgemm_kernel_L2_M2_32:

	tst	L, #1
	ble	cgemm_kernel_L2_M2_40

	KERNEL2x2_I
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_E

	b	 cgemm_kernel_L2_M2_44

cgemm_kernel_L2_M2_40:

	INIT2x2

cgemm_kernel_L2_M2_44:

	ands	L , K, #7					// L = K % 8
	ble	cgemm_kernel_L2_M2_100

cgemm_kernel_L2_M2_46:

	KERNEL2x2_SUB

	subs	L, L, #1
	bne	cgemm_kernel_L2_M2_46
	
cgemm_kernel_L2_M2_100:

	SAVE2x2

cgemm_kernel_L999:

	sub	r3, fp, #128
	vldm	r3, { s8 - s31}					// restore floating point registers

	sub	sp, fp, #28
	pop	{r4 - r9, fp}
	bx	lr

