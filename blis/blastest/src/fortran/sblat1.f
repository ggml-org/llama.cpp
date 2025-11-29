*> \brief \b SBLAT1
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM SBLAT1
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    Test program for the REAL Level 1 BLAS.
*>
*>    Based upon the original BLAS test routine together with:
*>    F06EAF Example Program Text
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee 
*> \author Univ. of California Berkeley 
*> \author Univ. of Colorado Denver 
*> \author NAG Ltd. 
*
*> \date April 2012
*
*> \ingroup single_blas_testing
*
*  =====================================================================
      PROGRAM SBLAT1
*
*  -- Reference BLAS test routine (version 3.4.1) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     April 2012
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER          NOUT
      PARAMETER        (NOUT=6)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, N
      LOGICAL          PASS
*     .. Local Scalars ..
      REAL             SFAC
      INTEGER          IC
*     .. External Subroutines ..
      EXTERNAL         CHECK0, CHECK1, CHECK2, CHECK3, HEADER
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA             SFAC/9.765625E-4/
*     .. Executable Statements ..
      WRITE (NOUT,99999)
      DO 20 IC = 1, 13
         ICASE = IC
         CALL HEADER
*
*        .. Initialize  PASS,  INCX,  and INCY for a new case. ..
*        .. the value 9999 for INCX or INCY will appear in the ..
*        .. detailed  output, if any, for cases  that do not involve ..
*        .. these parameters ..
*
         PASS = .TRUE.
         INCX = 9999
         INCY = 9999
         IF (ICASE.EQ.3 .OR. ICASE.EQ.11) THEN
            CALL CHECK0(SFAC)
         ELSE IF (ICASE.EQ.7 .OR. ICASE.EQ.8 .OR. ICASE.EQ.9 .OR.
     +            ICASE.EQ.10) THEN
            CALL CHECK1(SFAC)
         ELSE IF (ICASE.EQ.1 .OR. ICASE.EQ.2 .OR. ICASE.EQ.5 .OR.
     +            ICASE.EQ.6 .OR. ICASE.EQ.12 .OR. ICASE.EQ.13) THEN
            CALL CHECK2(SFAC)
         ELSE IF (ICASE.EQ.4) THEN
            CALL CHECK3(SFAC)
         END IF
*        -- Print
         IF (PASS) WRITE (NOUT,99998)
   20 CONTINUE
      STOP
*
99999 FORMAT (' Real BLAS Test Program Results',/1X)
99998 FORMAT ('                                    ----- PASS -----')
      END
      SUBROUTINE HEADER
*     .. Parameters ..
      INTEGER          NOUT
      PARAMETER        (NOUT=6)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, N
      LOGICAL          PASS
*     .. Local Arrays ..
      CHARACTER*6      L(13)
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA             L(1)/' SDOT '/
      DATA             L(2)/'SAXPY '/
      DATA             L(3)/'SROTG '/
      DATA             L(4)/' SROT '/
      DATA             L(5)/'SCOPY '/
      DATA             L(6)/'SSWAP '/
      DATA             L(7)/'SNRM2 '/
      DATA             L(8)/'SASUM '/
      DATA             L(9)/'SSCAL '/
      DATA             L(10)/'ISAMAX'/
      DATA             L(11)/'SROTMG'/
      DATA             L(12)/'SROTM '/
      DATA             L(13)/'SDSDOT'/
*     .. Executable Statements ..
      WRITE (NOUT,99999) ICASE, L(ICASE)
      RETURN
*
99999 FORMAT (/' Test of subprogram number',I3,12X,A6)
      END
      SUBROUTINE CHECK0(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      REAL              SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      REAL              D12, SA, SB, SC, SS
      INTEGER           I, K
*     .. Local Arrays ..
      REAL              DA1(8), DATRUE(8), DB1(8), DBTRUE(8), DC1(8),
     +                  DS1(8), DAB(4,9), DTEMP(9), DTRUE(9,9)
*     .. External Subroutines ..
      EXTERNAL          SROTG, SROTMG, STEST1
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA              DA1/0.3E0, 0.4E0, -0.3E0, -0.4E0, -0.3E0, 0.0E0,
     +                  0.0E0, 1.0E0/
      DATA              DB1/0.4E0, 0.3E0, 0.4E0, 0.3E0, -0.4E0, 0.0E0,
     +                  1.0E0, 0.0E0/
      DATA              DC1/0.6E0, 0.8E0, -0.6E0, 0.8E0, 0.6E0, 1.0E0,
     +                  0.0E0, 1.0E0/
      DATA              DS1/0.8E0, 0.6E0, 0.8E0, -0.6E0, 0.8E0, 0.0E0,
     +                  1.0E0, 0.0E0/
      DATA              DATRUE/0.5E0, 0.5E0, 0.5E0, -0.5E0, -0.5E0,
     +                  0.0E0, 1.0E0, 1.0E0/
      DATA              DBTRUE/0.0E0, 0.6E0, 0.0E0, -0.6E0, 0.0E0,
     +                  0.0E0, 1.0E0, 0.0E0/
*     INPUT FOR MODIFIED GIVENS
      DATA DAB/ .1E0,.3E0,1.2E0,.2E0,
     A          .7E0, .2E0, .6E0, 4.2E0,
     B          0.E0,0.E0,0.E0,0.E0,
     C          4.E0, -1.E0, 2.E0, 4.E0,
     D          6.E-10, 2.E-2, 1.E5, 10.E0,
     E          4.E10, 2.E-2, 1.E-5, 10.E0,
     F          2.E-10, 4.E-2, 1.E5, 10.E0,
     G          2.E10, 4.E-2, 1.E-5, 10.E0,
     H          4.E0, -2.E0, 8.E0, 4.E0    /
*    TRUE RESULTS FOR MODIFIED GIVENS
      DATA DTRUE/0.E0,0.E0, 1.3E0, .2E0, 0.E0,0.E0,0.E0, .5E0, 0.E0,
     A           0.E0,0.E0, 4.5E0, 4.2E0, 1.E0, .5E0, 0.E0,0.E0,0.E0,
     B           0.E0,0.E0,0.E0,0.E0, -2.E0, 0.E0,0.E0,0.E0,0.E0,
     C           0.E0,0.E0,0.E0, 4.E0, -1.E0, 0.E0,0.E0,0.E0,0.E0,
     D           0.E0, 15.E-3, 0.E0, 10.E0, -1.E0, 0.E0, -1.E-4,
     E           0.E0, 1.E0,
     F           0.E0,0.E0, 6144.E-5, 10.E0, -1.E0, 4096.E0, -1.E6,
     G           0.E0, 1.E0,
     H           0.E0,0.E0,15.E0,10.E0,-1.E0, 5.E-5, 0.E0,1.E0,0.E0,
     I           0.E0,0.E0, 15.E0, 10.E0, -1. E0, 5.E5, -4096.E0,
     J           1.E0, 4096.E-6,
     K           0.E0,0.E0, 7.E0, 4.E0, 0.E0,0.E0, -.5E0, -.25E0, 0.E0/
*                   4096 = 2 ** 12
      DATA D12  /4096.E0/
      DTRUE(1,1) = 12.E0 / 130.E0
      DTRUE(2,1) = 36.E0 / 130.E0
      DTRUE(7,1) = -1.E0 / 6.E0
      DTRUE(1,2) = 14.E0 / 75.E0
      DTRUE(2,2) = 49.E0 / 75.E0
      DTRUE(9,2) = 1.E0 / 7.E0
      DTRUE(1,5) = 45.E-11 * (D12 * D12)
      DTRUE(3,5) = 4.E5 / (3.E0 * D12)
      DTRUE(6,5) = 1.E0 / D12
      DTRUE(8,5) = 1.E4 / (3.E0 * D12)
      DTRUE(1,6) = 4.E10 / (1.5E0 * D12 * D12)
      DTRUE(2,6) = 2.E-2 / 1.5E0
      DTRUE(8,6) = 5.E-7 * D12
      DTRUE(1,7) = 4.E0 / 150.E0
      DTRUE(2,7) = (2.E-10 / 1.5E0) * (D12 * D12)
      DTRUE(7,7) = -DTRUE(6,5)
      DTRUE(9,7) = 1.E4 / D12
      DTRUE(1,8) = DTRUE(1,7)
      DTRUE(2,8) = 2.E10 / (1.5E0 * D12 * D12)
      DTRUE(1,9) = 32.E0 / 7.E0
      DTRUE(2,9) = -16.E0 / 7.E0
*     .. Executable Statements ..
*
*     Compute true values which cannot be prestored
*     in decimal notation
*
      DBTRUE(1) = 1.0E0/0.6E0
      DBTRUE(3) = -1.0E0/0.6E0
      DBTRUE(5) = 1.0E0/0.6E0
*
      DO 20 K = 1, 8
*        .. Set N=K for identification in output if any ..
         N = K
         IF (ICASE.EQ.3) THEN
*           .. SROTG ..
            IF (K.GT.8) GO TO 40
            SA = DA1(K)
            SB = DB1(K)
            CALL SROTG(SA,SB,SC,SS)
            CALL STEST1(SA,DATRUE(K),DATRUE(K),SFAC)
            CALL STEST1(SB,DBTRUE(K),DBTRUE(K),SFAC)
            CALL STEST1(SC,DC1(K),DC1(K),SFAC)
            CALL STEST1(SS,DS1(K),DS1(K),SFAC)
         ELSEIF (ICASE.EQ.11) THEN
*           .. SROTMG ..
            DO I=1,4
               DTEMP(I)= DAB(I,K)
               DTEMP(I+4) = 0.0
            END DO
            DTEMP(9) = 0.0
            CALL SROTMG(DTEMP(1),DTEMP(2),DTEMP(3),DTEMP(4),DTEMP(5))
            CALL STEST(9,DTEMP,DTRUE(1,K),DTRUE(1,K),SFAC)
         ELSE
            WRITE (NOUT,*) ' Shouldn''t be here in CHECK0'
            STOP
         END IF
   20 CONTINUE
   40 RETURN
      END
      SUBROUTINE CHECK1(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      REAL              SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      INTEGER           I, LEN, NP1
*     .. Local Arrays ..
      REAL              DTRUE1(5), DTRUE3(5), DTRUE5(8,5,2), DV(8,5,2),
     +                  SA(10), STEMP(1), STRUE(8), SX(8)
      INTEGER           ITRUE2(5)
*     .. External Functions ..
      REAL              SASUM, SNRM2
      INTEGER           ISAMAX
      EXTERNAL          SASUM, SNRM2, ISAMAX
*     .. External Subroutines ..
      EXTERNAL          ITEST1, SSCAL, STEST, STEST1
*     .. Intrinsic Functions ..
      INTRINSIC         MAX
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA              SA/0.3E0, -1.0E0, 0.0E0, 1.0E0, 0.3E0, 0.3E0,
     +                  0.3E0, 0.3E0, 0.3E0, 0.3E0/
      DATA              DV/0.1E0, 2.0E0, 2.0E0, 2.0E0, 2.0E0, 2.0E0,
     +                  2.0E0, 2.0E0, 0.3E0, 3.0E0, 3.0E0, 3.0E0, 3.0E0,
     +                  3.0E0, 3.0E0, 3.0E0, 0.3E0, -0.4E0, 4.0E0,
     +                  4.0E0, 4.0E0, 4.0E0, 4.0E0, 4.0E0, 0.2E0,
     +                  -0.6E0, 0.3E0, 5.0E0, 5.0E0, 5.0E0, 5.0E0,
     +                  5.0E0, 0.1E0, -0.3E0, 0.5E0, -0.1E0, 6.0E0,
     +                  6.0E0, 6.0E0, 6.0E0, 0.1E0, 8.0E0, 8.0E0, 8.0E0,
     +                  8.0E0, 8.0E0, 8.0E0, 8.0E0, 0.3E0, 9.0E0, 9.0E0,
     +                  9.0E0, 9.0E0, 9.0E0, 9.0E0, 9.0E0, 0.3E0, 2.0E0,
     +                  -0.4E0, 2.0E0, 2.0E0, 2.0E0, 2.0E0, 2.0E0,
     +                  0.2E0, 3.0E0, -0.6E0, 5.0E0, 0.3E0, 2.0E0,
     +                  2.0E0, 2.0E0, 0.1E0, 4.0E0, -0.3E0, 6.0E0,
     +                  -0.5E0, 7.0E0, -0.1E0, 3.0E0/
      DATA              DTRUE1/0.0E0, 0.3E0, 0.5E0, 0.7E0, 0.6E0/
      DATA              DTRUE3/0.0E0, 0.3E0, 0.7E0, 1.1E0, 1.0E0/
      DATA              DTRUE5/0.10E0, 2.0E0, 2.0E0, 2.0E0, 2.0E0,
     +                  2.0E0, 2.0E0, 2.0E0, -0.3E0, 3.0E0, 3.0E0,
     +                  3.0E0, 3.0E0, 3.0E0, 3.0E0, 3.0E0, 0.0E0, 0.0E0,
     +                  4.0E0, 4.0E0, 4.0E0, 4.0E0, 4.0E0, 4.0E0,
     +                  0.20E0, -0.60E0, 0.30E0, 5.0E0, 5.0E0, 5.0E0,
     +                  5.0E0, 5.0E0, 0.03E0, -0.09E0, 0.15E0, -0.03E0,
     +                  6.0E0, 6.0E0, 6.0E0, 6.0E0, 0.10E0, 8.0E0,
     +                  8.0E0, 8.0E0, 8.0E0, 8.0E0, 8.0E0, 8.0E0,
     +                  0.09E0, 9.0E0, 9.0E0, 9.0E0, 9.0E0, 9.0E0,
     +                  9.0E0, 9.0E0, 0.09E0, 2.0E0, -0.12E0, 2.0E0,
     +                  2.0E0, 2.0E0, 2.0E0, 2.0E0, 0.06E0, 3.0E0,
     +                  -0.18E0, 5.0E0, 0.09E0, 2.0E0, 2.0E0, 2.0E0,
     +                  0.03E0, 4.0E0, -0.09E0, 6.0E0, -0.15E0, 7.0E0,
     +                  -0.03E0, 3.0E0/
      DATA              ITRUE2/0, 1, 2, 2, 3/
*     .. Executable Statements ..
      DO 80 INCX = 1, 2
         DO 60 NP1 = 1, 5
            N = NP1 - 1
            LEN = 2*MAX(N,1)
*           .. Set vector arguments ..
            DO 20 I = 1, LEN
               SX(I) = DV(I,NP1,INCX)
   20       CONTINUE
*
            IF (ICASE.EQ.7) THEN
*              .. SNRM2 ..
               STEMP(1) = DTRUE1(NP1)
               CALL STEST1(SNRM2(N,SX,INCX),STEMP(1),STEMP,SFAC)
            ELSE IF (ICASE.EQ.8) THEN
*              .. SASUM ..
               STEMP(1) = DTRUE3(NP1)
               CALL STEST1(SASUM(N,SX,INCX),STEMP(1),STEMP,SFAC)
            ELSE IF (ICASE.EQ.9) THEN
*              .. SSCAL ..
               CALL SSCAL(N,SA((INCX-1)*5+NP1),SX,INCX)
               DO 40 I = 1, LEN
                  STRUE(I) = DTRUE5(I,NP1,INCX)
   40          CONTINUE
               CALL STEST(LEN,SX,STRUE,STRUE,SFAC)
            ELSE IF (ICASE.EQ.10) THEN
*              .. ISAMAX ..
               CALL ITEST1(ISAMAX(N,SX,INCX),ITRUE2(NP1))
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK1'
               STOP
            END IF
   60    CONTINUE
   80 CONTINUE
      RETURN
      END
      SUBROUTINE CHECK2(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      REAL              SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      REAL              SA
      INTEGER           I, J, KI, KN, KNI, KPAR, KSIZE, LENX, LENY,
     $                  MX, MY 
*     .. Local Arrays ..
      REAL              DT10X(7,4,4), DT10Y(7,4,4), DT7(4,4),
     $                  DT8(7,4,4), DX1(7),
     $                  DY1(7), SSIZE1(4), SSIZE2(14,2), SSIZE3(4),
     $                  SSIZE(7), STX(7), STY(7), SX(7), SY(7),
     $                  DPAR(5,4), DT19X(7,4,16),DT19XA(7,4,4),
     $                  DT19XB(7,4,4), DT19XC(7,4,4),DT19XD(7,4,4),
     $                  DT19Y(7,4,16), DT19YA(7,4,4),DT19YB(7,4,4),
     $                  DT19YC(7,4,4), DT19YD(7,4,4), DTEMP(5),
     $                  ST7B(4,4)
      INTEGER           INCXS(4), INCYS(4), LENS(4,2), NS(4)
*     .. External Functions ..
      REAL              SDOT, SDSDOT
      EXTERNAL          SDOT, SDSDOT
*     .. External Subroutines ..
      EXTERNAL          SAXPY, SCOPY, SROTM, SSWAP, STEST, STEST1
*     .. Intrinsic Functions ..
      INTRINSIC         ABS, MIN
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      EQUIVALENCE (DT19X(1,1,1),DT19XA(1,1,1)),(DT19X(1,1,5),
     A   DT19XB(1,1,1)),(DT19X(1,1,9),DT19XC(1,1,1)),
     B   (DT19X(1,1,13),DT19XD(1,1,1))
      EQUIVALENCE (DT19Y(1,1,1),DT19YA(1,1,1)),(DT19Y(1,1,5),
     A   DT19YB(1,1,1)),(DT19Y(1,1,9),DT19YC(1,1,1)),
     B   (DT19Y(1,1,13),DT19YD(1,1,1))

      DATA              SA/0.3E0/
      DATA              INCXS/1, 2, -2, -1/
      DATA              INCYS/1, -2, 1, -2/
      DATA              LENS/1, 1, 2, 4, 1, 1, 3, 7/
      DATA              NS/0, 1, 2, 4/
      DATA              DX1/0.6E0, 0.1E0, -0.5E0, 0.8E0, 0.9E0, -0.3E0,
     +                  -0.4E0/
      DATA              DY1/0.5E0, -0.9E0, 0.3E0, 0.7E0, -0.6E0, 0.2E0,
     +                  0.8E0/
      DATA              DT7/0.0E0, 0.30E0, 0.21E0, 0.62E0, 0.0E0,
     +                  0.30E0, -0.07E0, 0.85E0, 0.0E0, 0.30E0, -0.79E0,
     +                  -0.74E0, 0.0E0, 0.30E0, 0.33E0, 1.27E0/
      DATA              ST7B/ .1, .4, .31, .72,     .1, .4, .03, .95,
     +                  .1, .4, -.69, -.64,   .1, .4, .43, 1.37/
      DATA              DT8/0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.68E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.68E0, -0.87E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.68E0, -0.87E0, 0.15E0,
     +                  0.94E0, 0.0E0, 0.0E0, 0.0E0, 0.5E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.68E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.35E0, -0.9E0, 0.48E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.38E0, -0.9E0, 0.57E0, 0.7E0, -0.75E0,
     +                  0.2E0, 0.98E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.68E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.35E0, -0.72E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.38E0,
     +                  -0.63E0, 0.15E0, 0.88E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.68E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.68E0, -0.9E0, 0.33E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.68E0, -0.9E0, 0.33E0, 0.7E0,
     +                  -0.75E0, 0.2E0, 1.04E0/
      DATA              DT10X/0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.5E0, -0.9E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.5E0, -0.9E0, 0.3E0, 0.7E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.6E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.3E0, 0.1E0, 0.5E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.8E0, 0.1E0, -0.6E0,
     +                  0.8E0, 0.3E0, -0.3E0, 0.5E0, 0.6E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.5E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, -0.9E0,
     +                  0.1E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.7E0,
     +                  0.1E0, 0.3E0, 0.8E0, -0.9E0, -0.3E0, 0.5E0,
     +                  0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.5E0, 0.3E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.5E0, 0.3E0, -0.6E0, 0.8E0, 0.0E0, 0.0E0,
     +                  0.0E0/
      DATA              DT10Y/0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.6E0, 0.1E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.6E0, 0.1E0, -0.5E0, 0.8E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, -0.5E0, -0.9E0, 0.6E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, -0.4E0, -0.9E0, 0.9E0,
     +                  0.7E0, -0.5E0, 0.2E0, 0.6E0, 0.5E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.6E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, -0.5E0,
     +                  0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  -0.4E0, 0.9E0, -0.5E0, 0.6E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.6E0, -0.9E0, 0.1E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.6E0, -0.9E0, 0.1E0, 0.7E0,
     +                  -0.5E0, 0.2E0, 0.8E0/
      DATA              SSIZE1/0.0E0, 0.3E0, 1.6E0, 3.2E0/
      DATA              SSIZE2/0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0,
     +                  1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0,
     +                  1.17E0, 1.17E0, 1.17E0/
      DATA              SSIZE3/ .1, .4, 1.7, 3.3 /
*
*                         FOR DROTM
*
      DATA DPAR/-2.E0,  0.E0,0.E0,0.E0,0.E0,
     A          -1.E0,  2.E0, -3.E0, -4.E0,  5.E0,
     B           0.E0,  0.E0,  2.E0, -3.E0,  0.E0,
     C           1.E0,  5.E0,  2.E0,  0.E0, -4.E0/
*                        TRUE X RESULTS F0R ROTATIONS DROTM
      DATA DT19XA/.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E           -.8E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           -.9E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G           3.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .6E0,   .1E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     I           -.8E0,  3.8E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     J           -.9E0,  2.8E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     K           3.5E0,  -.4E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     L            .6E0,   .1E0,  -.5E0,   .8E0,          0.E0,0.E0,0.E0,
     M           -.8E0,  3.8E0, -2.2E0, -1.2E0,          0.E0,0.E0,0.E0,
     N           -.9E0,  2.8E0, -1.4E0, -1.3E0,          0.E0,0.E0,0.E0,
     O           3.5E0,  -.4E0, -2.2E0,  4.7E0,          0.E0,0.E0,0.E0/
*
      DATA DT19XB/.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E           -.8E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           -.9E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G           3.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .6E0,   .1E0,  -.5E0,             0.E0,0.E0,0.E0,0.E0,
     I           0.E0,    .1E0, -3.0E0,             0.E0,0.E0,0.E0,0.E0,
     J           -.3E0,   .1E0, -2.0E0,             0.E0,0.E0,0.E0,0.E0,
     K           3.3E0,   .1E0, -2.0E0,             0.E0,0.E0,0.E0,0.E0,
     L            .6E0,   .1E0,  -.5E0,   .8E0,   .9E0,  -.3E0,  -.4E0,
     M          -2.0E0,   .1E0,  1.4E0,   .8E0,   .6E0,  -.3E0, -2.8E0,
     N          -1.8E0,   .1E0,  1.3E0,   .8E0,  0.E0,   -.3E0, -1.9E0,
     O           3.8E0,   .1E0, -3.1E0,   .8E0,  4.8E0,  -.3E0, -1.5E0 /
*
      DATA DT19XC/.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E           -.8E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           -.9E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G           3.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .6E0,   .1E0,  -.5E0,             0.E0,0.E0,0.E0,0.E0,
     I           4.8E0,   .1E0, -3.0E0,             0.E0,0.E0,0.E0,0.E0,
     J           3.3E0,   .1E0, -2.0E0,             0.E0,0.E0,0.E0,0.E0,
     K           2.1E0,   .1E0, -2.0E0,             0.E0,0.E0,0.E0,0.E0,
     L            .6E0,   .1E0,  -.5E0,   .8E0,   .9E0,  -.3E0,  -.4E0,
     M          -1.6E0,   .1E0, -2.2E0,   .8E0,  5.4E0,  -.3E0, -2.8E0,
     N          -1.5E0,   .1E0, -1.4E0,   .8E0,  3.6E0,  -.3E0, -1.9E0,
     O           3.7E0,   .1E0, -2.2E0,   .8E0,  3.6E0,  -.3E0, -1.5E0 /
*
      DATA DT19XD/.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E           -.8E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           -.9E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G           3.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .6E0,   .1E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     I           -.8E0, -1.0E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     J           -.9E0,  -.8E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     K           3.5E0,   .8E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     L            .6E0,   .1E0,  -.5E0,   .8E0,          0.E0,0.E0,0.E0,
     M           -.8E0, -1.0E0,  1.4E0, -1.6E0,          0.E0,0.E0,0.E0,
     N           -.9E0,  -.8E0,  1.3E0, -1.6E0,          0.E0,0.E0,0.E0,
     O           3.5E0,   .8E0, -3.1E0,  4.8E0,          0.E0,0.E0,0.E0/
*                        TRUE Y RESULTS FOR ROTATIONS DROTM
      DATA DT19YA/.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E            .7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           1.7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G          -2.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .5E0,  -.9E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     I            .7E0, -4.8E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     J           1.7E0,  -.7E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     K          -2.6E0,  3.5E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     L            .5E0,  -.9E0,   .3E0,   .7E0,          0.E0,0.E0,0.E0,
     M            .7E0, -4.8E0,  3.0E0,  1.1E0,          0.E0,0.E0,0.E0,
     N           1.7E0,  -.7E0,  -.7E0,  2.3E0,          0.E0,0.E0,0.E0,
     O          -2.6E0,  3.5E0,  -.7E0, -3.6E0,          0.E0,0.E0,0.E0/
*
      DATA DT19YB/.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E            .7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           1.7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G          -2.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .5E0,  -.9E0,   .3E0,             0.E0,0.E0,0.E0,0.E0,
     I           4.0E0,  -.9E0,  -.3E0,             0.E0,0.E0,0.E0,0.E0,
     J           -.5E0,  -.9E0,  1.5E0,             0.E0,0.E0,0.E0,0.E0,
     K          -1.5E0,  -.9E0, -1.8E0,             0.E0,0.E0,0.E0,0.E0,
     L            .5E0,  -.9E0,   .3E0,   .7E0,  -.6E0,   .2E0,   .8E0,
     M           3.7E0,  -.9E0, -1.2E0,   .7E0, -1.5E0,   .2E0,  2.2E0,
     N           -.3E0,  -.9E0,  2.1E0,   .7E0, -1.6E0,   .2E0,  2.0E0,
     O          -1.6E0,  -.9E0, -2.1E0,   .7E0,  2.9E0,   .2E0, -3.8E0 /
*
      DATA DT19YC/.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E            .7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           1.7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G          -2.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .5E0,  -.9E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     I           4.0E0, -6.3E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     J           -.5E0,   .3E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     K          -1.5E0,  3.0E0,             0.E0,0.E0,0.E0,0.E0,0.E0,
     L            .5E0,  -.9E0,   .3E0,   .7E0,          0.E0,0.E0,0.E0,
     M           3.7E0, -7.2E0,  3.0E0,  1.7E0,          0.E0,0.E0,0.E0,
     N           -.3E0,   .9E0,  -.7E0,  1.9E0,          0.E0,0.E0,0.E0,
     O          -1.6E0,  2.7E0,  -.7E0, -3.4E0,          0.E0,0.E0,0.E0/
*
      DATA DT19YD/.5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     A            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     B            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     C            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     D            .5E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     E            .7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     F           1.7E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     G          -2.6E0,                  0.E0,0.E0,0.E0,0.E0,0.E0,0.E0,
     H            .5E0,  -.9E0,   .3E0,             0.E0,0.E0,0.E0,0.E0,
     I            .7E0,  -.9E0,  1.2E0,             0.E0,0.E0,0.E0,0.E0,
     J           1.7E0,  -.9E0,   .5E0,             0.E0,0.E0,0.E0,0.E0,
     K          -2.6E0,  -.9E0, -1.3E0,             0.E0,0.E0,0.E0,0.E0,
     L            .5E0,  -.9E0,   .3E0,   .7E0,  -.6E0,   .2E0,   .8E0,
     M            .7E0,  -.9E0,  1.2E0,   .7E0, -1.5E0,   .2E0,  1.6E0,
     N           1.7E0,  -.9E0,   .5E0,   .7E0, -1.6E0,   .2E0,  2.4E0,
     O          -2.6E0,  -.9E0, -1.3E0,   .7E0,  2.9E0,   .2E0, -4.0E0 /
*
*     .. Executable Statements ..
*
      DO 120 KI = 1, 4
         INCX = INCXS(KI)
         INCY = INCYS(KI)
         MX = ABS(INCX)
         MY = ABS(INCY)
*
         DO 100 KN = 1, 4
            N = NS(KN)
            KSIZE = MIN(2,KN)
            LENX = LENS(KN,MX)
            LENY = LENS(KN,MY)
*           .. Initialize all argument arrays ..
            DO 20 I = 1, 7
               SX(I) = DX1(I)
               SY(I) = DY1(I)
   20       CONTINUE
*
            IF (ICASE.EQ.1) THEN
*              .. SDOT ..
               CALL STEST1(SDOT(N,SX,INCX,SY,INCY),DT7(KN,KI),SSIZE1(KN)
     +                     ,SFAC)
            ELSE IF (ICASE.EQ.2) THEN
*              .. SAXPY ..
               CALL SAXPY(N,SA,SX,INCX,SY,INCY)
               DO 40 J = 1, LENY
                  STY(J) = DT8(J,KN,KI)
   40          CONTINUE
               CALL STEST(LENY,SY,STY,SSIZE2(1,KSIZE),SFAC)
            ELSE IF (ICASE.EQ.5) THEN
*              .. SCOPY ..
               DO 60 I = 1, 7
                  STY(I) = DT10Y(I,KN,KI)
   60          CONTINUE
               CALL SCOPY(N,SX,INCX,SY,INCY)
               CALL STEST(LENY,SY,STY,SSIZE2(1,1),1.0E0)
            ELSE IF (ICASE.EQ.6) THEN
*              .. SSWAP ..
               CALL SSWAP(N,SX,INCX,SY,INCY)
               DO 80 I = 1, 7
                  STX(I) = DT10X(I,KN,KI)
                  STY(I) = DT10Y(I,KN,KI)
   80          CONTINUE
               CALL STEST(LENX,SX,STX,SSIZE2(1,1),1.0E0)
               CALL STEST(LENY,SY,STY,SSIZE2(1,1),1.0E0)
            ELSEIF (ICASE.EQ.12) THEN
*              .. SROTM ..
               KNI=KN+4*(KI-1)
               DO KPAR=1,4
                  DO I=1,7
                     SX(I) = DX1(I)
                     SY(I) = DY1(I)
                     STX(I)= DT19X(I,KPAR,KNI)
                     STY(I)= DT19Y(I,KPAR,KNI)
                  END DO
*
                  DO I=1,5
                     DTEMP(I) = DPAR(I,KPAR)
                  END DO
*
                  DO  I=1,LENX
                     SSIZE(I)=STX(I)
                  END DO
*                   SEE REMARK ABOVE ABOUT DT11X(1,2,7)
*                       AND DT11X(5,3,8).
                  IF ((KPAR .EQ. 2) .AND. (KNI .EQ. 7))
     $               SSIZE(1) = 2.4E0
                  IF ((KPAR .EQ. 3) .AND. (KNI .EQ. 8))
     $               SSIZE(5) = 1.8E0
*
                  CALL   SROTM(N,SX,INCX,SY,INCY,DTEMP)
                  CALL   STEST(LENX,SX,STX,SSIZE,SFAC)
                  CALL   STEST(LENY,SY,STY,STY,SFAC)
               END DO
            ELSEIF (ICASE.EQ.13) THEN
*              .. SDSROT ..
               CALL STEST1 (SDSDOT(N,.1,SX,INCX,SY,INCY),
     $                 ST7B(KN,KI),SSIZE3(KN),SFAC)
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK2'
               STOP
            END IF
  100    CONTINUE
  120 CONTINUE
      RETURN
      END
      SUBROUTINE CHECK3(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      REAL              SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      REAL              SC, SS
      INTEGER           I, K, KI, KN, KSIZE, LENX, LENY, MX, MY
*     .. Local Arrays ..
      REAL              COPYX(5), COPYY(5), DT9X(7,4,4), DT9Y(7,4,4),
     +                  DX1(7), DY1(7), MWPC(11), MWPS(11), MWPSTX(5),
     +                  MWPSTY(5), MWPTX(11,5), MWPTY(11,5), MWPX(5),
     +                  MWPY(5), SSIZE2(14,2), STX(7), STY(7), SX(7),
     +                  SY(7)
      INTEGER           INCXS(4), INCYS(4), LENS(4,2), MWPINX(11),
     +                  MWPINY(11), MWPN(11), NS(4)
*     .. External Subroutines ..
      EXTERNAL          SROT, STEST
*     .. Intrinsic Functions ..
      INTRINSIC         ABS, MIN
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Data statements ..
      DATA              INCXS/1, 2, -2, -1/
      DATA              INCYS/1, -2, 1, -2/
      DATA              LENS/1, 1, 2, 4, 1, 1, 3, 7/
      DATA              NS/0, 1, 2, 4/
      DATA              DX1/0.6E0, 0.1E0, -0.5E0, 0.8E0, 0.9E0, -0.3E0,
     +                  -0.4E0/
      DATA              DY1/0.5E0, -0.9E0, 0.3E0, 0.7E0, -0.6E0, 0.2E0,
     +                  0.8E0/
      DATA              SC, SS/0.8E0, 0.6E0/
      DATA              DT9X/0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.78E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.78E0, -0.46E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.78E0, -0.46E0, -0.22E0,
     +                  1.06E0, 0.0E0, 0.0E0, 0.0E0, 0.6E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.78E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.66E0, 0.1E0, -0.1E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.96E0, 0.1E0, -0.76E0, 0.8E0, 0.90E0,
     +                  -0.3E0, -0.02E0, 0.6E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.78E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, -0.06E0, 0.1E0,
     +                  -0.1E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.90E0,
     +                  0.1E0, -0.22E0, 0.8E0, 0.18E0, -0.3E0, -0.02E0,
     +                  0.6E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.78E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.78E0, 0.26E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.78E0, 0.26E0, -0.76E0, 1.12E0,
     +                  0.0E0, 0.0E0, 0.0E0/
      DATA              DT9Y/0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.04E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.04E0, -0.78E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.04E0, -0.78E0, 0.54E0,
     +                  0.08E0, 0.0E0, 0.0E0, 0.0E0, 0.5E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.04E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.7E0,
     +                  -0.9E0, -0.12E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.64E0, -0.9E0, -0.30E0, 0.7E0, -0.18E0, 0.2E0,
     +                  0.28E0, 0.5E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.04E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.7E0, -1.08E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.64E0, -1.26E0,
     +                  0.54E0, 0.20E0, 0.0E0, 0.0E0, 0.0E0, 0.5E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.04E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.04E0, -0.9E0, 0.18E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.04E0, -0.9E0, 0.18E0, 0.7E0,
     +                  -0.18E0, 0.2E0, 0.16E0/
      DATA              SSIZE2/0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0, 0.0E0,
     +                  0.0E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0,
     +                  1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0, 1.17E0,
     +                  1.17E0, 1.17E0, 1.17E0/
*     .. Executable Statements ..
*
      DO 60 KI = 1, 4
         INCX = INCXS(KI)
         INCY = INCYS(KI)
         MX = ABS(INCX)
         MY = ABS(INCY)
*
         DO 40 KN = 1, 4
            N = NS(KN)
            KSIZE = MIN(2,KN)
            LENX = LENS(KN,MX)
            LENY = LENS(KN,MY)
*
            IF (ICASE.EQ.4) THEN
*              .. SROT ..
               DO 20 I = 1, 7
                  SX(I) = DX1(I)
                  SY(I) = DY1(I)
                  STX(I) = DT9X(I,KN,KI)
                  STY(I) = DT9Y(I,KN,KI)
   20          CONTINUE
               CALL SROT(N,SX,INCX,SY,INCY,SC,SS)
               CALL STEST(LENX,SX,STX,SSIZE2(1,KSIZE),SFAC)
               CALL STEST(LENY,SY,STY,SSIZE2(1,KSIZE),SFAC)
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK3'
               STOP
            END IF
   40    CONTINUE
   60 CONTINUE
*
      MWPC(1) = 1
      DO 80 I = 2, 11
         MWPC(I) = 0
   80 CONTINUE
      MWPS(1) = 0
      DO 100 I = 2, 6
         MWPS(I) = 1
  100 CONTINUE
      DO 120 I = 7, 11
         MWPS(I) = -1
  120 CONTINUE
      MWPINX(1) = 1
      MWPINX(2) = 1
      MWPINX(3) = 1
      MWPINX(4) = -1
      MWPINX(5) = 1
      MWPINX(6) = -1
      MWPINX(7) = 1
      MWPINX(8) = 1
      MWPINX(9) = -1
      MWPINX(10) = 1
      MWPINX(11) = -1
      MWPINY(1) = 1
      MWPINY(2) = 1
      MWPINY(3) = -1
      MWPINY(4) = -1
      MWPINY(5) = 2
      MWPINY(6) = 1
      MWPINY(7) = 1
      MWPINY(8) = -1
      MWPINY(9) = -1
      MWPINY(10) = 2
      MWPINY(11) = 1
      DO 140 I = 1, 11
         MWPN(I) = 5
  140 CONTINUE
      MWPN(5) = 3
      MWPN(10) = 3
      DO 160 I = 1, 5
         MWPX(I) = I
         MWPY(I) = I
         MWPTX(1,I) = I
         MWPTY(1,I) = I
         MWPTX(2,I) = I
         MWPTY(2,I) = -I
         MWPTX(3,I) = 6 - I
         MWPTY(3,I) = I - 6
         MWPTX(4,I) = I
         MWPTY(4,I) = -I
         MWPTX(6,I) = 6 - I
         MWPTY(6,I) = I - 6
         MWPTX(7,I) = -I
         MWPTY(7,I) = I
         MWPTX(8,I) = I - 6
         MWPTY(8,I) = 6 - I
         MWPTX(9,I) = -I
         MWPTY(9,I) = I
         MWPTX(11,I) = I - 6
         MWPTY(11,I) = 6 - I
  160 CONTINUE
      MWPTX(5,1) = 1
      MWPTX(5,2) = 3
      MWPTX(5,3) = 5
      MWPTX(5,4) = 4
      MWPTX(5,5) = 5
      MWPTY(5,1) = -1
      MWPTY(5,2) = 2
      MWPTY(5,3) = -2
      MWPTY(5,4) = 4
      MWPTY(5,5) = -3
      MWPTX(10,1) = -1
      MWPTX(10,2) = -3
      MWPTX(10,3) = -5
      MWPTX(10,4) = 4
      MWPTX(10,5) = 5
      MWPTY(10,1) = 1
      MWPTY(10,2) = 2
      MWPTY(10,3) = 2
      MWPTY(10,4) = 4
      MWPTY(10,5) = 3
      DO 200 I = 1, 11
         INCX = MWPINX(I)
         INCY = MWPINY(I)
         DO 180 K = 1, 5
            COPYX(K) = MWPX(K)
            COPYY(K) = MWPY(K)
            MWPSTX(K) = MWPTX(I,K)
            MWPSTY(K) = MWPTY(I,K)
  180    CONTINUE
         CALL SROT(MWPN(I),COPYX,INCX,COPYY,INCY,MWPC(I),MWPS(I))
         CALL STEST(5,COPYX,MWPSTX,MWPSTX,SFAC)
         CALL STEST(5,COPYY,MWPSTY,MWPSTY,SFAC)
  200 CONTINUE
      RETURN
      END
      SUBROUTINE STEST(LEN,SCOMP,STRUE,SSIZE,SFAC)
*     ********************************* STEST **************************
*
*     THIS SUBR COMPARES ARRAYS  SCOMP() AND STRUE() OF LENGTH LEN TO
*     SEE IF THE TERM BY TERM DIFFERENCES, MULTIPLIED BY SFAC, ARE
*     NEGLIGIBLE.
*
*     C. L. LAWSON, JPL, 1974 DEC 10
*
*     .. Parameters ..
      INTEGER          NOUT
      REAL             ZERO
      PARAMETER        (NOUT=6, ZERO=0.0E0)
*     .. Scalar Arguments ..
      REAL             SFAC
      INTEGER          LEN
*     .. Array Arguments ..
      REAL             SCOMP(LEN), SSIZE(LEN), STRUE(LEN)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, N
      LOGICAL          PASS
*     .. Local Scalars ..
      REAL             SD
      INTEGER          I
*     .. External Functions ..
      REAL             SDIFF
      EXTERNAL         SDIFF
*     .. Intrinsic Functions ..
      INTRINSIC        ABS
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Executable Statements ..
*
      DO 40 I = 1, LEN
         SD = SCOMP(I) - STRUE(I)
         IF (ABS(SFAC*SD) .LE. ABS(SSIZE(I))*EPSILON(ZERO))
     +       GO TO 40
*
*                             HERE    SCOMP(I) IS NOT CLOSE TO STRUE(I).
*
         IF ( .NOT. PASS) GO TO 20
*                             PRINT FAIL MESSAGE AND HEADER.
         PASS = .FALSE.
         WRITE (NOUT,99999)
         WRITE (NOUT,99998)
   20    WRITE (NOUT,99997) ICASE, N, INCX, INCY, I, SCOMP(I),
     +     STRUE(I), SD, SSIZE(I)
   40 CONTINUE
      RETURN
*
99999 FORMAT ('                                       FAIL')
99998 FORMAT (/' CASE  N INCX INCY  I                            ',
     +       ' COMP(I)                             TRUE(I)  DIFFERENCE',
     +       '     SIZE(I)',/1X)
99997 FORMAT (1X,I4,I3,2I5,I3,2E36.8,2E12.4)
      END
      SUBROUTINE STEST1(SCOMP1,STRUE1,SSIZE,SFAC)
*     ************************* STEST1 *****************************
*
*     THIS IS AN INTERFACE SUBROUTINE TO ACCOMODATE THE FORTRAN
*     REQUIREMENT THAT WHEN A DUMMY ARGUMENT IS AN ARRAY, THE
*     ACTUAL ARGUMENT MUST ALSO BE AN ARRAY OR AN ARRAY ELEMENT.
*
*     C.L. LAWSON, JPL, 1978 DEC 6
*
*     .. Scalar Arguments ..
      REAL              SCOMP1, SFAC, STRUE1
*     .. Array Arguments ..
      REAL              SSIZE(*)
*     .. Local Arrays ..
      REAL              SCOMP(1), STRUE(1)
*     .. External Subroutines ..
      EXTERNAL          STEST
*     .. Executable Statements ..
*
      SCOMP(1) = SCOMP1
      STRUE(1) = STRUE1
      CALL STEST(1,SCOMP,STRUE,SSIZE,SFAC)
*
      RETURN
      END
      REAL             FUNCTION SDIFF(SA,SB)
*     ********************************* SDIFF **************************
*     COMPUTES DIFFERENCE OF TWO NUMBERS.  C. L. LAWSON, JPL 1974 FEB 15
*
*     .. Scalar Arguments ..
      REAL                            SA, SB
*     .. Executable Statements ..
      SDIFF = SA - SB
      RETURN
      END
      SUBROUTINE ITEST1(ICOMP,ITRUE)
*     ********************************* ITEST1 *************************
*
*     THIS SUBROUTINE COMPARES THE VARIABLES ICOMP AND ITRUE FOR
*     EQUALITY.
*     C. L. LAWSON, JPL, 1974 DEC 10
*
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      INTEGER           ICOMP, ITRUE
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, N
      LOGICAL           PASS
*     .. Local Scalars ..
      INTEGER           ID
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, PASS
*     .. Executable Statements ..
*
      IF (ICOMP.EQ.ITRUE) GO TO 40
*
*                            HERE ICOMP IS NOT EQUAL TO ITRUE.
*
      IF ( .NOT. PASS) GO TO 20
*                             PRINT FAIL MESSAGE AND HEADER.
      PASS = .FALSE.
      WRITE (NOUT,99999)
      WRITE (NOUT,99998)
   20 ID = ICOMP - ITRUE
      WRITE (NOUT,99997) ICASE, N, INCX, INCY, ICOMP, ITRUE, ID
   40 CONTINUE
      RETURN
*
99999 FORMAT ('                                       FAIL')
99998 FORMAT (/' CASE  N INCX INCY                               ',
     +       ' COMP                                TRUE     DIFFERENCE',
     +       /1X)
99997 FORMAT (1X,I4,I3,2I5,2I36,I12)
      END
