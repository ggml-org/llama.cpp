
#define REALNAME bli_dgemm_armv7a_ker_4x4

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

.macro KERNEL4x4_I
	pld	[ BO , #B_PRE ]
	fldd	d8 , [ BO ]
	fldd	d0 , [ AO ]
	pld	[ AO , #A_PRE ]

	fldd	d1 , [ AO, #8 ]
	fmuld	d16  , d0,  d8
	fldd	d2 , [ AO, #16 ]
	fmuld	d17  , d1,  d8
	fldd	d3 , [ AO, #24 ]
	fmuld	d18  , d2,  d8
	fldd	d9 , [ BO, #8 ]
	fmuld	d19  , d3,  d8

	fldd	d10, [ BO, #16 ]
	fmuld	d20  , d0,  d9
	fldd	d11, [ BO, #24 ]
	fmuld	d21  , d1,  d9
	add	BO , BO, #32
	add	AO , AO, #32
	fmuld	d22  , d2,  d9

	pld	[ BO , #B_PRE ]
	fldd	d12, [ BO ]
	fmuld	d23  , d3,  d9

	pld	[ AO , #A_PRE ]
	fldd	d4 , [ AO, #0 ]
	fmuld	d24  , d0,  d10
	fldd	d5 , [ AO, #8 ]
	fmuld	d25  , d1,  d10
	fldd	d6 , [ AO, #16 ]
	fmuld	d26  , d2,  d10
	fldd	d7 , [ AO, #24 ]
	fmuld	d27  , d3,  d10

	fldd	d13, [ BO, #8 ]
	fmuld	d28  , d0,  d11
	fldd	d14, [ BO, #16 ]
	fmuld	d29  , d1,  d11
	fldd	d15, [ BO, #24 ]
	fmuld	d30  , d2,  d11
	fmuld	d31  , d3,  d11

.endm

.macro KERNEL4x4_M2

	fmacd	d16  , d4,  d12
	pld	[ AO , #A_PRE+32 ]
	fmacd	d17  , d5,  d12
	fldd	d0 , [ AO , #32 ]
	fmacd	d18  , d6,  d12
	pld	[ BO , #B_PRE+32 ]
	fmacd	d19  , d7,  d12

	fldd	d8 , [ BO , #32 ]
	fmacd	d20  , d4,  d13
	fldd	d1 , [ AO, #40 ]
	fmacd	d21  , d5,  d13
	fldd	d2 , [ AO, #48 ]
	fmacd	d22  , d6,  d13
	fldd	d3 , [ AO, #56 ]
	fmacd	d23  , d7,  d13

	fmacd	d24  , d4,  d14
	fmacd	d25  , d5,  d14
	fldd	d9 , [ BO, #40 ]
	fmacd	d26  , d6,  d14
	fldd	d10, [ BO, #48 ]
	fmacd	d27  , d7,  d14

	fldd	d11, [ BO, #56 ]
	fmacd	d28  , d4,  d15
	fmacd	d29  , d5,  d15
	add	AO , AO, #64
	fmacd	d30  , d6,  d15
	add	BO , BO, #64
	fmacd	d31  , d7,  d15

.endm

.macro KERNEL4x4_M1

	fmacd	d16  , d0,  d8
	pld	[ AO , #A_PRE ]
	fmacd	d17  , d1,  d8
	fldd	d4 , [ AO ]
	fmacd	d18  , d2,  d8
	pld	[ BO , #B_PRE ]
	fmacd	d19  , d3,  d8

	fldd	d12, [ BO ]
	fmacd	d20  , d0,  d9
	fldd	d5 , [ AO, #8 ]
	fmacd	d21  , d1,  d9
	fldd	d6 , [ AO, #16 ]
	fmacd	d22  , d2,  d9
	fldd	d7 , [ AO, #24 ]
	fmacd	d23  , d3,  d9

	fmacd	d24  , d0,  d10
	fmacd	d25  , d1,  d10
	fldd	d13, [ BO, #8 ]
	fmacd	d26  , d2,  d10
	fldd	d14, [ BO, #16 ]
	fmacd	d27  , d3,  d10

	fldd	d15, [ BO, #24 ]
	fmacd	d28  , d0,  d11
	fmacd	d29  , d1,  d11
	fmacd	d30  , d2,  d11
	fmacd	d31  , d3,  d11

.endm

.macro KERNEL4x4_E

	fmacd	d16  , d4,  d12
	fmacd	d17  , d5,  d12
	add	BO , BO, #32
	fmacd	d18  , d6,  d12
	add	AO , AO, #32
	fmacd	d19  , d7,  d12

	fmacd	d20  , d4,  d13
	fmacd	d21  , d5,  d13
	fmacd	d22  , d6,  d13
	fmacd	d23  , d7,  d13

	fmacd	d24  , d4,  d14
	fmacd	d25  , d5,  d14
	fmacd	d26  , d6,  d14
	fmacd	d27  , d7,  d14

	fmacd	d28  , d4,  d15
	fmacd	d29  , d5,  d15
	fmacd	d30  , d6,  d15
	fmacd	d31  , d7,  d15

.endm

.macro KERNEL4x4_SUB

	fldd	d8 , [ BO ]
	pld	[ BO , #B_PRE ]

	fldd	d0 , [ AO ]
	pld	[ AO , #A_PRE ]
	fldd	d1 , [ AO, #8 ]

	fmacd	d16  , d0,  d8
	fldd	d2 , [ AO, #16 ]
	fmacd	d17  , d1,  d8
	fldd	d3 , [ AO, #24 ]
	fmacd	d18  , d2,  d8
	fldd	d9 , [ BO, #8 ]
	fmacd	d19  , d3,  d8

	fldd	d10, [ BO, #16 ]
	fmacd	d20  , d0,  d9
	fldd	d11, [ BO, #24 ]
	fmacd	d21  , d1,  d9
	fmacd	d22  , d2,  d9
	fmacd	d23  , d3,  d9

	fmacd	d24  , d0,  d10
	fmacd	d25  , d1,  d10
	fmacd	d26  , d2,  d10
	fmacd	d27  , d3,  d10

	fmacd	d28  , d0,  d11
	fmacd	d29  , d1,  d11
	add	AO , AO, #32
	fmacd	d30  , d2,  d11
	add	BO , BO, #32
	fmacd	d31  , d3,  d11

.endm

.macro SAVE4x4

	ldr	r3, OLD_RSC				// Row stride size
	lsl	r3, r3, #3				// multiply with size of double

	fldd	d0, [ PTR_ALPHA	]			// load alpha
	ldr	r4, PTR_BETA
	fldd	d1, [ r4 ]				// load beta

//-----------------------------------------------------------
	mov	r2, CO1					// save pointer
	mov	r4, CO2					// save pointer
	fldd	d8, [ CO1 ]				// load value from C
	fldd	d12, [ CO2 ]				// load value from C
	fmuld	d8, d8, d1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacd	d8, d0, d16				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	fldd	d9, [ CO1 ]				// load value from C
	fldd	d13, [ CO2 ]				// load value from C
	fmuld	d9, d9, d1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacd	d9, d0, d17				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	fldd	d10, [ CO1 ]				// load value from C
	fldd	d14, [ CO2 ]				// load value from C
	fmuld	d10, d10, d1				// multiply with beta
	add	CO1, CO1, r3				// compute next pointer
	fmacd	d10, d0, d18				// multiply sum with alpha and add to value of C	
	add	CO2, CO2, r3				// compute next pointer

	fldd	d11, [ CO1 ]				// load value from C
	fldd	d15, [ CO2 ]				// load value from C
	fmuld	d11, d11, d1				// multiply with beta
	mov	CO1, r2					// restore pointer
	fmacd	d11, d0, d19				// multiply sum with alpha and add to value of C	
	mov	CO2, r4					// restore pointer
	
	fstd	d8, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fstd	d9, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fstd	d10, [ CO1 ]				// store value in C
	add	CO1 , CO1, r3				// compute next pointer
	fstd	d11, [ CO1 ]				// store value in C

//-----------------------------------------------------------
	mov	r2, CO3					// save pointer
	fldd	d8, [ CO3 ]				// load value from C
	fmuld	d12, d12, d1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacd	d12, d0, d20				// multiply sum with alpha and add to value of C	

	fldd	d9, [ CO3 ]				// load value from C
	fmuld	d13, d13, d1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacd	d13, d0, d21				// multiply sum with alpha and add to value of C	

	fldd	d10, [ CO3 ]				// load value from C
	fmuld	d14, d14, d1				// multiply with beta
	add	CO3, CO3, r3				// compute next pointer
	fmacd	d14, d0, d22				// multiply sum with alpha and add to value of C	

	fldd	d11, [ CO3 ]				// load value from C
	fmuld	d15, d15, d1				// multiply with beta
	mov	CO3, r2					// restore pointer
	fmacd	d15, d0, d23				// multiply sum with alpha and add to value of C	
	
	fstd	d12, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fstd	d13, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fstd	d14, [ CO2 ]				// store value in C
	add	CO2 , CO2, r3				// compute next pointer
	fstd	d15, [ CO2 ]				// store value in C

//-----------------------------------------------------------
	mov	r4, CO4					// save pointer
	fldd	d12, [ CO4 ]				// load value from C
	fmuld	d8, d8, d1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacd	d8, d0, d24				// multiply sum with alpha and add to value of C	

	fldd	d13, [ CO4 ]				// load value from C
	fmuld	d9, d9, d1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacd	d9, d0, d25				// multiply sum with alpha and add to value of C	

	fldd	d14, [ CO4 ]				// load value from C
	fmuld	d10, d10, d1				// multiply with beta
	add	CO4, CO4, r3				// compute next pointer
	fmacd	d10, d0, d26				// multiply sum with alpha and add to value of C	

	fldd	d15, [ CO4 ]				// load value from C
	fmuld	d11, d11, d1				// multiply with beta
	mov	CO4, r4					// restore pointer
	fmacd	d11, d0, d27				// multiply sum with alpha and add to value of C	
	

//-----------------------------------------------------------
	fstd	d8, [ CO3 ]				// store value in C
	fmuld	d12, d12, d1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacd	d12, d0, d28				// multiply sum with alpha and add to value of C	

	fstd	d9, [ CO3 ]				// store value in C
	fmuld	d13, d13, d1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacd	d13, d0, d29				// multiply sum with alpha and add to value of C	

	fstd	d10, [ CO3 ]				// store value in C
	fmuld	d14, d14, d1				// multiply with beta
	add	CO3 , CO3, r3				// compute next pointer
	fmacd	d14, d0, d30				// multiply sum with alpha and add to value of C	

	fstd	d11, [ CO3 ]				// store value in C
	fmuld	d15, d15, d1				// multiply with beta
	fstd	d12, [ CO4 ]				// store value in C
	fmacd	d15, d0, d31				// multiply sum with alpha and add to value of C	
	
	add	CO4 , CO4, r3				// compute next pointer
	fstd	d13, [ CO4 ]				// store value in C
	add	CO4 , CO4, r3				// compute next pointer
	fstd	d14, [ CO4 ]				// store value in C
	add	CO4 , CO4, r3				// compute next pointer
	fstd	d15, [ CO4 ]				// store value in C

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
	lsl	r3, r3, #3					// multiply with size of double

	mov	CO1, r2						// first line of C
	add	CO2, CO1, r3					// second line of C
	add	CO3, CO2, r3					// third line of C
	add	CO4, CO3, r3					// fourth line of C

	pld	[ CO1, #C_PRE ]					// prefetch the lines of C
	pld	[ CO2, #C_PRE ]					// prefetch the lines of C
	pld	[ CO3, #C_PRE ]					// prefetch the lines of C
	pld	[ CO3, #C_PRE ]					// prefetch the lines of C

dgemm_kernel_L4_M4_20:

	asrs	L , K, #3					// L = K / 8
	cmp	L , #2
	blt	dgemm_kernel_L4_M4_32

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	subs	L, L, #2
	ble	dgemm_kernel_L4_M4_22a
	.align 5

dgemm_kernel_L4_M4_22:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	subs	L, L, #1
	bgt	dgemm_kernel_L4_M4_22

dgemm_kernel_L4_M4_22a:

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	b	 dgemm_kernel_L4_M4_44

dgemm_kernel_L4_M4_32:

	tst	L, #1
	ble	dgemm_kernel_L4_M4_40

	KERNEL4x4_I
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_M2

	KERNEL4x4_M1
	KERNEL4x4_M2
	KERNEL4x4_M1
	KERNEL4x4_E

	b	 dgemm_kernel_L4_M4_44

dgemm_kernel_L4_M4_40:

	INIT4x4

dgemm_kernel_L4_M4_44:

	ands	L , K, #7					// L = K % 8
	ble	dgemm_kernel_L4_M4_100

dgemm_kernel_L4_M4_46:

	KERNEL4x4_SUB

	subs	L, L, #1
	bne	dgemm_kernel_L4_M4_46
	
dgemm_kernel_L4_M4_100:

	SAVE4x4

dgemm_kernel_L999:

	sub	r3, fp, #128
	vldm	r3, { d8 - d15}					// restore floating point registers

	sub	sp, fp, #28
	pop	{r4 - r9, fp}
	bx	lr

