
#define REALNAME bli_zgemm_armv7a_ker_2x2

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

#define FMAC_BR	fnmacd
#define FMAC_BI	fmacd

#define NN 1

#if defined(NN) || defined(NT) || defined(TN) || defined(TT) 

	#define	FADD_R	fsubd
	#define	FADD_I	faddd

	#define	FMAC_R1	fnmacd
	#define	FMAC_R2	fnmacd
	#define	FMAC_I1	fmacd
	#define	FMAC_I2	fnmacd

#elif defined(CN) || defined(CT)

	#define	FADD_R	faddd
	#define	FADD_I	fsubd

	#define	FMAC_R1	fmacd
	#define	FMAC_R2	fmacd
	#define	FMAC_I1	fnmacd
	#define	FMAC_I2	fmacd

#elif defined(NC) || defined(TC)

	#define	FADD_R	faddd
	#define	FADD_I	fsubd

	#define	FMAC_R1	fmacd
	#define	FMAC_R2	fnmacd
	#define	FMAC_I1	fmacd
	#define	FMAC_I2	fmacd

#else

	#define	FADD_R  fsubd
	#define	FADD_I	faddd

	#define	FMAC_R1	fnmacd
	#define	FMAC_R2	fmacd
	#define	FMAC_I1	fnmacd
	#define	FMAC_I2	fnmacd

#endif



.macro INIT2x2

	vsub.f64		d16 , d16 , d16
	vmov.f64		d17, d16
	vmov.f64		d18, d16
	vmov.f64		d19, d16
	vmov.f64		d20, d16
	vmov.f64		d21, d16
	vmov.f64		d22, d16
	vmov.f64		d23, d16
	vmov.f64		d24, d16
	vmov.f64		d25, d16
	vmov.f64		d26, d16
	vmov.f64		d27, d16
	vmov.f64		d28, d16
	vmov.f64		d29, d16
	vmov.f64		d30, d16
	vmov.f64		d31, d16

.endm

.macro KERNEL2x2_I
	pld	[ AO , #A_PRE ]
	pld	[ BO , #B_PRE ]
	fldd	d0 , [ AO ]
	fldd	d1 , [ AO, #8 ]
	fldd	d8 , [ BO ]
	fldd	d9 , [ BO, #8 ]

	fmuld	d16  , d0,  d8
	fldd	d2 , [ AO, #16 ]
	fmuld	d24  , d1,  d9
	fldd	d3 , [ AO, #24 ]
	fmuld	d17  , d0,  d9
	fldd	d10, [ BO, #16 ]
	fmuld	d25  , d1,  d8

	fldd	d11, [ BO, #24 ]
	fmuld	d18  , d2,  d8
	add	BO , BO, #32
	fmuld	d26  , d3,  d9
	add	AO , AO, #32
	fmuld	d19  , d2,  d9
	pld	[ BO , #B_PRE ]
	fmuld	d27  , d3,  d8

	pld	[ AO , #A_PRE ]
	fmuld	d20  , d0,  d10
	fldd	d4 , [ AO, #0 ]
	fmuld	d28  , d1,  d11
	fldd	d5 , [ AO, #8 ]
	fmuld	d21  , d0,  d11
	fldd	d12, [ BO ]
	fmuld	d29  , d1,  d10

	fldd	d13, [ BO, #8 ]
	fmuld	d22  , d2,  d10
	fldd	d6 , [ AO, #16 ]
	fmuld	d30  , d3,  d11
	fldd	d7 , [ AO, #24 ]
	fmuld	d23  , d2,  d11
	fldd	d14, [ BO, #16 ]
	fmuld	d31  , d3,  d10
	fldd	d15, [ BO, #24 ]

	add	BO , BO, #32
	add	AO , AO, #32
.endm



.macro KERNEL2x2_M1
	pld	[ AO , #A_PRE ]

	fmacd	d16  , d0,  d8
	pld	[ BO , #B_PRE ]
	fmacd	d24  , d1,  d9
	fldd	d4 , [ AO, #0 ]
	fmacd	d17  , d0,  d9
	fldd	d5 , [ AO, #8 ]
	fmacd	d25  , d1,  d8

	fldd	d12, [ BO ]
	fmacd	d18  , d2,  d8
	fldd	d13, [ BO, #8 ]
	fmacd	d26  , d3,  d9
	fldd	d6 , [ AO, #16 ]
	fmacd	d19  , d2,  d9
	fldd	d7 , [ AO, #24 ]
	fmacd	d27  , d3,  d8

	fmacd	d20  , d0,  d10
	fldd	d14, [ BO, #16 ]
	fmacd	d28  , d1,  d11
	fmacd	d21  , d0,  d11
	fldd	d15, [ BO, #24 ]
	fmacd	d29  , d1,  d10

	fmacd	d22  , d2,  d10
	add	BO , BO, #32
	fmacd	d30  , d3,  d11
	fmacd	d23  , d2,  d11
	add	AO , AO, #32
	fmacd	d31  , d3,  d10

.endm

.macro KERNEL2x2_M2
	pld	[ AO , #A_PRE ]

	fmacd	d16  , d4,  d12
	pld	[ BO , #B_PRE ]
	fmacd	d24  , d5,  d13
	fldd	d0 , [ AO, #0 ]
	fmacd	d17  , d4,  d13
	fldd	d1 , [ AO, #8 ]
	fmacd	d25  , d5,  d12

	fmacd	d18  , d6,  d12
	fldd	d8 , [ BO ]
	fmacd	d26  , d7,  d13
	fldd	d9 , [ BO, #8 ]
	fmacd	d19  , d6,  d13
	fmacd	d27  , d7,  d12

	fldd	d2 , [ AO, #16 ]
	fmacd	d20  , d4,  d14
	fldd	d3 , [ AO, #24 ]
	fmacd	d28  , d5,  d15
	fmacd	d21  , d4,  d15
	fldd	d10, [ BO, #16 ]
	fmacd	d29  , d5,  d14

	fldd	d11, [ BO, #24 ]
	fmacd	d22  , d6,  d14
	fmacd	d30  , d7,  d15
	add	BO , BO, #32
	fmacd	d23  , d6,  d15
	add	AO , AO, #32
	fmacd	d31  , d7,  d14

.endm


.macro KERNEL2x2_E

	fmacd	d16  , d4,  d12
	fmacd	d24  , d5,  d13
	fmacd	d17  , d4,  d13
	fmacd	d25  , d5,  d12

	fmacd	d18  , d6,  d12
	fmacd	d26  , d7,  d13
	fmacd	d19  , d6,  d13
	fmacd	d27  , d7,  d12

	fmacd	d20  , d4,  d14
	fmacd	d28  , d5,  d15
	fmacd	d21  , d4,  d15
	fmacd	d29  , d5,  d14

	fmacd	d22  , d6,  d14
	fmacd	d30  , d7,  d15
	fmacd	d23  , d6,  d15
	fmacd	d31  , d7,  d14

.endm

.macro KERNEL2x2_SUB

	pld	[ AO , #A_PRE ]
	pld	[ BO , #B_PRE ]
	fldd	d0 , [ AO ]
	fldd	d1 , [ AO, #8 ]
	fldd	d8 , [ BO ]
	fldd	d9 , [ BO, #8 ]

	fmacd	d16  , d0,  d8
	fldd	d2 , [ AO, #16 ]
	fmacd	d24  , d1,  d9
	fldd	d3 , [ AO, #24 ]
	fmacd	d17  , d0,  d9
	fldd	d10, [ BO, #16 ]
	fmacd	d25  , d1,  d8

	fldd	d11, [ BO, #24 ]
	fmacd	d18  , d2,  d8
	fmacd	d26  , d3,  d9
	fmacd	d19  , d2,  d9
	fmacd	d27  , d3,  d8

	fmacd	d20  , d0,  d10
	fmacd	d28  , d1,  d11
	fmacd	d21  , d0,  d11
	fmacd	d29  , d1,  d10

	fmacd	d22  , d2,  d10
	add	BO , BO, #32
	fmacd	d30  , d3,  d11
	fmacd	d23  , d2,  d11
	add	AO , AO, #32
	fmacd	d31  , d3,  d10

.endm




.macro SAVE2x2

        ldr     r3, OLD_RSC                             // Row stride size
        lsl     r3, r3, #4                              // multiply with size of complex double

        fldd    d0, [ PTR_ALPHA ]                       // load real part of alpha
        fldd    d1, [ PTR_ALPHA, #8 ]                   // load imag part of alpha
        ldr     r4, PTR_BETA
        fldd    d2, [ r4 ]                              // load real part of beta
        fldd    d3, [ r4, #8  ]                         // load imag part of beta

	// Add/Sub the real and the imag parts
	FADD_R	d16, d24 , d16
	FADD_I  d17, d25 , d17
	FADD_R	d18, d26 , d18
	FADD_I  d19, d27 , d19
	FADD_R	d20, d28 , d20
	FADD_I  d21, d29 , d21
	FADD_R	d22, d30 , d22
	FADD_I  d23, d31 , d23

	mov	r4, CO1					// save pointer
	fldmiad CO1, { d4 - d5 }			// read real and imag part from C
	add	CO1, CO1, r3
	
	mov	r2, CO2					// save pointer
	fldmiad CO2, { d8 - d9 }			// read real and imag part from C
	add	CO2, CO2, r3

	fmuld	d24, d4, d2				// multiply Beta-real with C-real
	fmuld	d25, d5, d2				// multiply Beta-real with C-imag
	fmuld	d28, d8, d2				// multiply Beta-real with C-real
	fmuld	d29, d9, d2				// multiply Beta-real with C-imag

	FMAC_BR	d24, d3, d5				// multiply beta-imag with C-imag and add
	FMAC_BI	d25, d3, d4				// multiply beta-imag with C-real and add
	FMAC_BR	d28, d3, d9				// multiply beta-imag with C-imag and add
	FMAC_BI	d29, d3, d8				// multiply beta-imag with C-real and add

	FMAC_R1 d24 , d0 , d16
	FMAC_I1 d25 , d0 , d17
	FMAC_R2 d24 , d1 , d17
	FMAC_I2	d25 , d1 , d16

	FMAC_R1 d28 , d0 , d20
	FMAC_I1 d29 , d0 , d21
	FMAC_R2 d28 , d1 , d21
	FMAC_I2	d29 , d1 , d20

	fldmiad CO1, { d4 - d5 }			// read real and imag part from C
	fldmiad CO2, { d8 - d9 }			// read real and imag part from C

	fmuld	d26, d4, d2				// multiply Beta-real with C-real
	fmuld	d27, d5, d2				// multiply Beta-real with C-imag
	fmuld	d30, d8, d2				// multiply Beta-real with C-real
	fmuld	d31, d9, d2				// multiply Beta-real with C-imag

	FMAC_BR	d26, d3, d5				// multiply beta-imag with C-imag and add
	FMAC_BI	d27, d3, d4				// multiply beta-imag with C-real and add
	FMAC_BR	d30, d3, d9				// multiply beta-imag with C-imag and add
	FMAC_BI	d31, d3, d8				// multiply beta-imag with C-real and add

	FMAC_R1 d26 , d0 , d18
	FMAC_I1 d27 , d0 , d19
	FMAC_R2 d26 , d1 , d19
	FMAC_I2	d27 , d1 , d18

	FMAC_R1 d30, d0 , d22
	FMAC_I1 d31, d0 , d23
	FMAC_R2 d30, d1 , d23
	FMAC_I2	d31, d1 , d22

	mov	CO1, r4					// restore pointer
	mov	CO2, r2					// restore pointer
	fstmiad CO1, { d24 - d25 }
	fstmiad CO2, { d28 - d29 }
	add	CO1, CO1, r3
	add	CO2, CO2, r3
	fstmiad CO1, { d26 - d27 }
	fstmiad CO2, { d30 - d31 }


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
	vstm	r3, { d8 - d15} 				// store floating point registers

	ldr	r2, OLD_C					// pointer matrix C
	ldr	r3, OLD_CSC					// Col stride size of C
	lsl	r3, r3, #4					// multiply with size of complex double

	mov	CO1, r2						// first line of C
	add	CO2, CO1, r3					// second line of C

	pld	[ CO1, #C_PRE ]					// prefetch the lines of C
	pld	[ CO2, #C_PRE ]					// prefetch the lines of C

zgemm_kernel_L2_M2_20:

	asrs	L , K, #3					// L = K / 8
	cmp	L , #2
	blt	zgemm_kernel_L2_M2_32

	KERNEL2x2_I
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	subs	L, L, #2
	ble	zgemm_kernel_L2_M2_22a
	.align 5

zgemm_kernel_L2_M2_22:

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	subs	L, L, #1
	bgt	zgemm_kernel_L2_M2_22

zgemm_kernel_L2_M2_22a:

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_E

	b	 zgemm_kernel_L2_M2_44

zgemm_kernel_L2_M2_32:

	tst	L, #1
	ble	zgemm_kernel_L2_M2_40

	KERNEL2x2_I
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_M2

	KERNEL2x2_M1
	KERNEL2x2_M2
	KERNEL2x2_M1
	KERNEL2x2_E

	b	 zgemm_kernel_L2_M2_44

zgemm_kernel_L2_M2_40:

	INIT2x2

zgemm_kernel_L2_M2_44:

	ands	L , K, #7					// L = K % 8
	ble	zgemm_kernel_L2_M2_100

zgemm_kernel_L2_M2_46:

	KERNEL2x2_SUB

	subs	L, L, #1
	bne	zgemm_kernel_L2_M2_46
	
zgemm_kernel_L2_M2_100:

	SAVE2x2

zgemm_kernel_L999:

	sub	r3, fp, #128
	vldm	r3, { d8 - d15}					// restore floating point registers

	sub	sp, fp, #28
	pop	{r4 - r9, fp}
	bx	lr

