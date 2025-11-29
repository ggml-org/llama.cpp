*> \brief \b ZBLAT1
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM ZBLAT1
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    Test program for the COMPLEX*16 Level 1 BLAS.
*>
*>    Based upon the original BLAS test routine together with:
*>    F06GAF Example Program Text
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
*> \ingroup complex16_blas_testing
*
*  =====================================================================
      PROGRAM ZBLAT1
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
      INTEGER          ICASE, INCX, INCY, MODE, N
      LOGICAL          PASS
*     .. Local Scalars ..
      DOUBLE PRECISION SFAC
      INTEGER          IC
*     .. External Subroutines ..
      EXTERNAL         CHECK1, CHECK2, HEADER
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
*     .. Data statements ..
      DATA             SFAC/9.765625D-4/
*     .. Executable Statements ..
      WRITE (NOUT,99999)
      DO 20 IC = 1, 10
         ICASE = IC
         CALL HEADER
*
*        Initialize PASS, INCX, INCY, and MODE for a new case.
*        The value 9999 for INCX, INCY or MODE will appear in the
*        detailed  output, if any, for cases that do not involve
*        these parameters.
*
         PASS = .TRUE.
         INCX = 9999
         INCY = 9999
         MODE = 9999
         IF (ICASE.LE.5) THEN
            CALL CHECK2(SFAC)
         ELSE IF (ICASE.GE.6) THEN
            CALL CHECK1(SFAC)
         END IF
*        -- Print
         IF (PASS) WRITE (NOUT,99998)
   20 CONTINUE
      STOP
*
99999 FORMAT (' Complex BLAS Test Program Results',/1X)
99998 FORMAT ('                                    ----- PASS -----')
      END
      SUBROUTINE HEADER
*     .. Parameters ..
      INTEGER          NOUT
      PARAMETER        (NOUT=6)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, MODE, N
      LOGICAL          PASS
*     .. Local Arrays ..
      CHARACTER*6      L(10)
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
*     .. Data statements ..
      DATA             L(1)/'ZDOTC '/
      DATA             L(2)/'ZDOTU '/
      DATA             L(3)/'ZAXPY '/
      DATA             L(4)/'ZCOPY '/
      DATA             L(5)/'ZSWAP '/
      DATA             L(6)/'DZNRM2'/
      DATA             L(7)/'DZASUM'/
      DATA             L(8)/'ZSCAL '/
      DATA             L(9)/'ZDSCAL'/
      DATA             L(10)/'IZAMAX'/
*     .. Executable Statements ..
      WRITE (NOUT,99999) ICASE, L(ICASE)
      RETURN
*
99999 FORMAT (/' Test of subprogram number',I3,12X,A6)
      END
      SUBROUTINE CHECK1(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      DOUBLE PRECISION  SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, MODE, N
      LOGICAL           PASS
*     .. Local Scalars ..
      COMPLEX*16        CA
      DOUBLE PRECISION  SA
      INTEGER           I, J, LEN, NP1
*     .. Local Arrays ..
      COMPLEX*16        CTRUE5(8,5,2), CTRUE6(8,5,2), CV(8,5,2), CX(8),
     +                  MWPCS(5), MWPCT(5)
      DOUBLE PRECISION  STRUE2(5), STRUE4(5)
      INTEGER           ITRUE3(5)
*     .. External Functions ..
      DOUBLE PRECISION  DZASUM, DZNRM2
      INTEGER           IZAMAX
      EXTERNAL          DZASUM, DZNRM2, IZAMAX
*     .. External Subroutines ..
      EXTERNAL          ZSCAL, ZDSCAL, CTEST, ITEST1, STEST1
*     .. Intrinsic Functions ..
      INTRINSIC         MAX
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
*     .. Data statements ..
      DATA              SA, CA/0.3D0, (0.4D0,-0.7D0)/
      DATA              ((CV(I,J,1),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (0.3D0,-0.4D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (0.1D0,-0.3D0), (0.5D0,-0.1D0), (5.0D0,6.0D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (5.0D0,6.0D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (0.1D0,0.1D0),
     +                  (-0.6D0,0.1D0), (0.1D0,-0.3D0), (7.0D0,8.0D0),
     +                  (7.0D0,8.0D0), (7.0D0,8.0D0), (7.0D0,8.0D0),
     +                  (7.0D0,8.0D0), (0.3D0,0.1D0), (0.5D0,0.0D0),
     +                  (0.0D0,0.5D0), (0.0D0,0.2D0), (2.0D0,3.0D0),
     +                  (2.0D0,3.0D0), (2.0D0,3.0D0), (2.0D0,3.0D0)/
      DATA              ((CV(I,J,2),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (0.3D0,-0.4D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (0.1D0,-0.3D0), (8.0D0,9.0D0), (0.5D0,-0.1D0),
     +                  (2.0D0,5.0D0), (2.0D0,5.0D0), (2.0D0,5.0D0),
     +                  (2.0D0,5.0D0), (2.0D0,5.0D0), (0.1D0,0.1D0),
     +                  (3.0D0,6.0D0), (-0.6D0,0.1D0), (4.0D0,7.0D0),
     +                  (0.1D0,-0.3D0), (7.0D0,2.0D0), (7.0D0,2.0D0),
     +                  (7.0D0,2.0D0), (0.3D0,0.1D0), (5.0D0,8.0D0),
     +                  (0.5D0,0.0D0), (6.0D0,9.0D0), (0.0D0,0.5D0),
     +                  (8.0D0,3.0D0), (0.0D0,0.2D0), (9.0D0,4.0D0)/
      DATA              STRUE2/0.0D0, 0.5D0, 0.6D0, 0.7D0, 0.8D0/
      DATA              STRUE4/0.0D0, 0.7D0, 1.0D0, 1.3D0, 1.6D0/
      DATA              ((CTRUE5(I,J,1),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (-0.16D0,-0.37D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (-0.17D0,-0.19D0), (0.13D0,-0.39D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (5.0D0,6.0D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (5.0D0,6.0D0),
     +                  (0.11D0,-0.03D0), (-0.17D0,0.46D0),
     +                  (-0.17D0,-0.19D0), (7.0D0,8.0D0), (7.0D0,8.0D0),
     +                  (7.0D0,8.0D0), (7.0D0,8.0D0), (7.0D0,8.0D0),
     +                  (0.19D0,-0.17D0), (0.20D0,-0.35D0),
     +                  (0.35D0,0.20D0), (0.14D0,0.08D0),
     +                  (2.0D0,3.0D0), (2.0D0,3.0D0), (2.0D0,3.0D0),
     +                  (2.0D0,3.0D0)/
      DATA              ((CTRUE5(I,J,2),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (-0.16D0,-0.37D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (-0.17D0,-0.19D0), (8.0D0,9.0D0),
     +                  (0.13D0,-0.39D0), (2.0D0,5.0D0), (2.0D0,5.0D0),
     +                  (2.0D0,5.0D0), (2.0D0,5.0D0), (2.0D0,5.0D0),
     +                  (0.11D0,-0.03D0), (3.0D0,6.0D0),
     +                  (-0.17D0,0.46D0), (4.0D0,7.0D0),
     +                  (-0.17D0,-0.19D0), (7.0D0,2.0D0), (7.0D0,2.0D0),
     +                  (7.0D0,2.0D0), (0.19D0,-0.17D0), (5.0D0,8.0D0),
     +                  (0.20D0,-0.35D0), (6.0D0,9.0D0),
     +                  (0.35D0,0.20D0), (8.0D0,3.0D0),
     +                  (0.14D0,0.08D0), (9.0D0,4.0D0)/
      DATA              ((CTRUE6(I,J,1),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (1.0D0,2.0D0), (1.0D0,2.0D0),
     +                  (1.0D0,2.0D0), (0.09D0,-0.12D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (3.0D0,4.0D0), (3.0D0,4.0D0), (3.0D0,4.0D0),
     +                  (0.03D0,-0.09D0), (0.15D0,-0.03D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (5.0D0,6.0D0),
     +                  (5.0D0,6.0D0), (5.0D0,6.0D0), (5.0D0,6.0D0),
     +                  (0.03D0,0.03D0), (-0.18D0,0.03D0),
     +                  (0.03D0,-0.09D0), (7.0D0,8.0D0), (7.0D0,8.0D0),
     +                  (7.0D0,8.0D0), (7.0D0,8.0D0), (7.0D0,8.0D0),
     +                  (0.09D0,0.03D0), (0.15D0,0.00D0),
     +                  (0.00D0,0.15D0), (0.00D0,0.06D0), (2.0D0,3.0D0),
     +                  (2.0D0,3.0D0), (2.0D0,3.0D0), (2.0D0,3.0D0)/
      DATA              ((CTRUE6(I,J,2),I=1,8),J=1,5)/(0.1D0,0.1D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (4.0D0,5.0D0), (4.0D0,5.0D0),
     +                  (4.0D0,5.0D0), (0.09D0,-0.12D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (6.0D0,7.0D0), (6.0D0,7.0D0), (6.0D0,7.0D0),
     +                  (0.03D0,-0.09D0), (8.0D0,9.0D0),
     +                  (0.15D0,-0.03D0), (2.0D0,5.0D0), (2.0D0,5.0D0),
     +                  (2.0D0,5.0D0), (2.0D0,5.0D0), (2.0D0,5.0D0),
     +                  (0.03D0,0.03D0), (3.0D0,6.0D0),
     +                  (-0.18D0,0.03D0), (4.0D0,7.0D0),
     +                  (0.03D0,-0.09D0), (7.0D0,2.0D0), (7.0D0,2.0D0),
     +                  (7.0D0,2.0D0), (0.09D0,0.03D0), (5.0D0,8.0D0),
     +                  (0.15D0,0.00D0), (6.0D0,9.0D0), (0.00D0,0.15D0),
     +                  (8.0D0,3.0D0), (0.00D0,0.06D0), (9.0D0,4.0D0)/
      DATA              ITRUE3/0, 1, 2, 2, 2/
*     .. Executable Statements ..
      DO 60 INCX = 1, 2
         DO 40 NP1 = 1, 5
            N = NP1 - 1
            LEN = 2*MAX(N,1)
*           .. Set vector arguments ..
            DO 20 I = 1, LEN
               CX(I) = CV(I,NP1,INCX)
   20       CONTINUE
            IF (ICASE.EQ.6) THEN
*              .. DZNRM2 ..
               CALL STEST1(DZNRM2(N,CX,INCX),STRUE2(NP1),STRUE2(NP1),
     +                     SFAC)
            ELSE IF (ICASE.EQ.7) THEN
*              .. DZASUM ..
               CALL STEST1(DZASUM(N,CX,INCX),STRUE4(NP1),STRUE4(NP1),
     +                     SFAC)
            ELSE IF (ICASE.EQ.8) THEN
*              .. ZSCAL ..
               CALL ZSCAL(N,CA,CX,INCX)
               CALL CTEST(LEN,CX,CTRUE5(1,NP1,INCX),CTRUE5(1,NP1,INCX),
     +                    SFAC)
            ELSE IF (ICASE.EQ.9) THEN
*              .. ZDSCAL ..
               CALL ZDSCAL(N,SA,CX,INCX)
               CALL CTEST(LEN,CX,CTRUE6(1,NP1,INCX),CTRUE6(1,NP1,INCX),
     +                    SFAC)
            ELSE IF (ICASE.EQ.10) THEN
*              .. IZAMAX ..
               CALL ITEST1(IZAMAX(N,CX,INCX),ITRUE3(NP1))
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK1'
               STOP
            END IF
*
   40    CONTINUE
   60 CONTINUE
*
      INCX = 1
      IF (ICASE.EQ.8) THEN
*        ZSCAL
*        Add a test for alpha equal to zero.
         CA = (0.0D0,0.0D0)
         DO 80 I = 1, 5
            MWPCT(I) = (0.0D0,0.0D0)
            MWPCS(I) = (1.0D0,1.0D0)
   80    CONTINUE
         CALL ZSCAL(5,CA,CX,INCX)
         CALL CTEST(5,CX,MWPCT,MWPCS,SFAC)
      ELSE IF (ICASE.EQ.9) THEN
*        ZDSCAL
*        Add a test for alpha equal to zero.
         SA = 0.0D0
         DO 100 I = 1, 5
            MWPCT(I) = (0.0D0,0.0D0)
            MWPCS(I) = (1.0D0,1.0D0)
  100    CONTINUE
         CALL ZDSCAL(5,SA,CX,INCX)
         CALL CTEST(5,CX,MWPCT,MWPCS,SFAC)
*        Add a test for alpha equal to one.
         SA = 1.0D0
         DO 120 I = 1, 5
            MWPCT(I) = CX(I)
            MWPCS(I) = CX(I)
  120    CONTINUE
         CALL ZDSCAL(5,SA,CX,INCX)
         CALL CTEST(5,CX,MWPCT,MWPCS,SFAC)
*        Add a test for alpha equal to minus one.
         SA = -1.0D0
         DO 140 I = 1, 5
            MWPCT(I) = -CX(I)
            MWPCS(I) = -CX(I)
  140    CONTINUE
         CALL ZDSCAL(5,SA,CX,INCX)
         CALL CTEST(5,CX,MWPCT,MWPCS,SFAC)
      END IF
      RETURN
      END
      SUBROUTINE CHECK2(SFAC)
*     .. Parameters ..
      INTEGER           NOUT
      PARAMETER         (NOUT=6)
*     .. Scalar Arguments ..
      DOUBLE PRECISION  SFAC
*     .. Scalars in Common ..
      INTEGER           ICASE, INCX, INCY, MODE, N
      LOGICAL           PASS
*     .. Local Scalars ..
      COMPLEX*16        CA
      INTEGER           I, J, KI, KN, KSIZE, LENX, LENY, MX, MY
*     .. Local Arrays ..
      COMPLEX*16        CDOT(1), CSIZE1(4), CSIZE2(7,2), CSIZE3(14),
     +                  CT10X(7,4,4), CT10Y(7,4,4), CT6(4,4), CT7(4,4),
     +                  CT8(7,4,4), CX(7), CX1(7), CY(7), CY1(7)
      INTEGER           INCXS(4), INCYS(4), LENS(4,2), NS(4)
*     .. External Functions ..
      COMPLEX*16        ZDOTC, ZDOTU
      EXTERNAL          ZDOTC, ZDOTU
*     .. External Subroutines ..
      EXTERNAL          ZAXPY, ZCOPY, ZSWAP, CTEST
*     .. Intrinsic Functions ..
      INTRINSIC         ABS, MIN
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
*     .. Data statements ..
      DATA              CA/(0.4D0,-0.7D0)/
      DATA              INCXS/1, 2, -2, -1/
      DATA              INCYS/1, -2, 1, -2/
      DATA              LENS/1, 1, 2, 4, 1, 1, 3, 7/
      DATA              NS/0, 1, 2, 4/
      DATA              CX1/(0.7D0,-0.8D0), (-0.4D0,-0.7D0),
     +                  (-0.1D0,-0.9D0), (0.2D0,-0.8D0),
     +                  (-0.9D0,-0.4D0), (0.1D0,0.4D0), (-0.6D0,0.6D0)/
      DATA              CY1/(0.6D0,-0.6D0), (-0.9D0,0.5D0),
     +                  (0.7D0,-0.6D0), (0.1D0,-0.5D0), (-0.1D0,-0.2D0),
     +                  (-0.5D0,-0.3D0), (0.8D0,-0.7D0)/
      DATA              ((CT8(I,J,1),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.32D0,-1.41D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.32D0,-1.41D0),
     +                  (-1.55D0,0.5D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.32D0,-1.41D0), (-1.55D0,0.5D0),
     +                  (0.03D0,-0.89D0), (-0.38D0,-0.96D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0)/
      DATA              ((CT8(I,J,2),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.32D0,-1.41D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (-0.07D0,-0.89D0),
     +                  (-0.9D0,0.5D0), (0.42D0,-1.41D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.78D0,0.06D0), (-0.9D0,0.5D0),
     +                  (0.06D0,-0.13D0), (0.1D0,-0.5D0),
     +                  (-0.77D0,-0.49D0), (-0.5D0,-0.3D0),
     +                  (0.52D0,-1.51D0)/
      DATA              ((CT8(I,J,3),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.32D0,-1.41D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (-0.07D0,-0.89D0),
     +                  (-1.18D0,-0.31D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.78D0,0.06D0), (-1.54D0,0.97D0),
     +                  (0.03D0,-0.89D0), (-0.18D0,-1.31D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0)/
      DATA              ((CT8(I,J,4),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.32D0,-1.41D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.32D0,-1.41D0), (-0.9D0,0.5D0),
     +                  (0.05D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.32D0,-1.41D0),
     +                  (-0.9D0,0.5D0), (0.05D0,-0.6D0), (0.1D0,-0.5D0),
     +                  (-0.77D0,-0.49D0), (-0.5D0,-0.3D0),
     +                  (0.32D0,-1.16D0)/
      DATA              CT7/(0.0D0,0.0D0), (-0.06D0,-0.90D0),
     +                  (0.65D0,-0.47D0), (-0.34D0,-1.22D0),
     +                  (0.0D0,0.0D0), (-0.06D0,-0.90D0),
     +                  (-0.59D0,-1.46D0), (-1.04D0,-0.04D0),
     +                  (0.0D0,0.0D0), (-0.06D0,-0.90D0),
     +                  (-0.83D0,0.59D0), (0.07D0,-0.37D0),
     +                  (0.0D0,0.0D0), (-0.06D0,-0.90D0),
     +                  (-0.76D0,-1.15D0), (-1.33D0,-1.82D0)/
      DATA              CT6/(0.0D0,0.0D0), (0.90D0,0.06D0),
     +                  (0.91D0,-0.77D0), (1.80D0,-0.10D0),
     +                  (0.0D0,0.0D0), (0.90D0,0.06D0), (1.45D0,0.74D0),
     +                  (0.20D0,0.90D0), (0.0D0,0.0D0), (0.90D0,0.06D0),
     +                  (-0.55D0,0.23D0), (0.83D0,-0.39D0),
     +                  (0.0D0,0.0D0), (0.90D0,0.06D0), (1.04D0,0.79D0),
     +                  (1.95D0,1.22D0)/
      DATA              ((CT10X(I,J,1),I=1,7),J=1,4)/(0.7D0,-0.8D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.6D0,-0.6D0), (-0.9D0,0.5D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.6D0,-0.6D0),
     +                  (-0.9D0,0.5D0), (0.7D0,-0.6D0), (0.1D0,-0.5D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0)/
      DATA              ((CT10X(I,J,2),I=1,7),J=1,4)/(0.7D0,-0.8D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.7D0,-0.6D0), (-0.4D0,-0.7D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.8D0,-0.7D0),
     +                  (-0.4D0,-0.7D0), (-0.1D0,-0.2D0),
     +                  (0.2D0,-0.8D0), (0.7D0,-0.6D0), (0.1D0,0.4D0),
     +                  (0.6D0,-0.6D0)/
      DATA              ((CT10X(I,J,3),I=1,7),J=1,4)/(0.7D0,-0.8D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (-0.9D0,0.5D0), (-0.4D0,-0.7D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.1D0,-0.5D0),
     +                  (-0.4D0,-0.7D0), (0.7D0,-0.6D0), (0.2D0,-0.8D0),
     +                  (-0.9D0,0.5D0), (0.1D0,0.4D0), (0.6D0,-0.6D0)/
      DATA              ((CT10X(I,J,4),I=1,7),J=1,4)/(0.7D0,-0.8D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.6D0,-0.6D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.6D0,-0.6D0), (0.7D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.6D0,-0.6D0),
     +                  (0.7D0,-0.6D0), (-0.1D0,-0.2D0), (0.8D0,-0.7D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0)/
      DATA              ((CT10Y(I,J,1),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.7D0,-0.8D0), (-0.4D0,-0.7D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.7D0,-0.8D0),
     +                  (-0.4D0,-0.7D0), (-0.1D0,-0.9D0),
     +                  (0.2D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0)/
      DATA              ((CT10Y(I,J,2),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (-0.1D0,-0.9D0), (-0.9D0,0.5D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (-0.6D0,0.6D0),
     +                  (-0.9D0,0.5D0), (-0.9D0,-0.4D0), (0.1D0,-0.5D0),
     +                  (-0.1D0,-0.9D0), (-0.5D0,-0.3D0),
     +                  (0.7D0,-0.8D0)/
      DATA              ((CT10Y(I,J,3),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (-0.1D0,-0.9D0), (0.7D0,-0.8D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (-0.6D0,0.6D0),
     +                  (-0.9D0,-0.4D0), (-0.1D0,-0.9D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0)/
      DATA              ((CT10Y(I,J,4),I=1,7),J=1,4)/(0.6D0,-0.6D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.7D0,-0.8D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.7D0,-0.8D0), (-0.9D0,0.5D0),
     +                  (-0.4D0,-0.7D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.7D0,-0.8D0),
     +                  (-0.9D0,0.5D0), (-0.4D0,-0.7D0), (0.1D0,-0.5D0),
     +                  (-0.1D0,-0.9D0), (-0.5D0,-0.3D0),
     +                  (0.2D0,-0.8D0)/
      DATA              CSIZE1/(0.0D0,0.0D0), (0.9D0,0.9D0),
     +                  (1.63D0,1.73D0), (2.90D0,2.78D0)/
      DATA              CSIZE3/(0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (1.17D0,1.17D0),
     +                  (1.17D0,1.17D0), (1.17D0,1.17D0),
     +                  (1.17D0,1.17D0), (1.17D0,1.17D0),
     +                  (1.17D0,1.17D0), (1.17D0,1.17D0)/
      DATA              CSIZE2/(0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (0.0D0,0.0D0),
     +                  (0.0D0,0.0D0), (0.0D0,0.0D0), (1.54D0,1.54D0),
     +                  (1.54D0,1.54D0), (1.54D0,1.54D0),
     +                  (1.54D0,1.54D0), (1.54D0,1.54D0),
     +                  (1.54D0,1.54D0), (1.54D0,1.54D0)/
*     .. Executable Statements ..
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
*           .. initialize all argument arrays ..
            DO 20 I = 1, 7
               CX(I) = CX1(I)
               CY(I) = CY1(I)
   20       CONTINUE
            IF (ICASE.EQ.1) THEN
*              .. ZDOTC ..
               CDOT(1) = ZDOTC(N,CX,INCX,CY,INCY)
               CALL CTEST(1,CDOT,CT6(KN,KI),CSIZE1(KN),SFAC)
            ELSE IF (ICASE.EQ.2) THEN
*              .. ZDOTU ..
               CDOT(1) = ZDOTU(N,CX,INCX,CY,INCY)
               CALL CTEST(1,CDOT,CT7(KN,KI),CSIZE1(KN),SFAC)
            ELSE IF (ICASE.EQ.3) THEN
*              .. ZAXPY ..
               CALL ZAXPY(N,CA,CX,INCX,CY,INCY)
               CALL CTEST(LENY,CY,CT8(1,KN,KI),CSIZE2(1,KSIZE),SFAC)
            ELSE IF (ICASE.EQ.4) THEN
*              .. ZCOPY ..
               CALL ZCOPY(N,CX,INCX,CY,INCY)
               CALL CTEST(LENY,CY,CT10Y(1,KN,KI),CSIZE3,1.0D0)
            ELSE IF (ICASE.EQ.5) THEN
*              .. ZSWAP ..
               CALL ZSWAP(N,CX,INCX,CY,INCY)
               CALL CTEST(LENX,CX,CT10X(1,KN,KI),CSIZE3,1.0D0)
               CALL CTEST(LENY,CY,CT10Y(1,KN,KI),CSIZE3,1.0D0)
            ELSE
               WRITE (NOUT,*) ' Shouldn''t be here in CHECK2'
               STOP
            END IF
*
   40    CONTINUE
   60 CONTINUE
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
      DOUBLE PRECISION ZERO
      PARAMETER        (NOUT=6, ZERO=0.0D0)
*     .. Scalar Arguments ..
      DOUBLE PRECISION SFAC
      INTEGER          LEN
*     .. Array Arguments ..
      DOUBLE PRECISION SCOMP(LEN), SSIZE(LEN), STRUE(LEN)
*     .. Scalars in Common ..
      INTEGER          ICASE, INCX, INCY, MODE, N
      LOGICAL          PASS
*     .. Local Scalars ..
      DOUBLE PRECISION SD
      INTEGER          I
*     .. External Functions ..
      DOUBLE PRECISION SDIFF
      EXTERNAL         SDIFF
*     .. Intrinsic Functions ..
      INTRINSIC        ABS
*     .. Common blocks ..
      COMMON           /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
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
   20    WRITE (NOUT,99997) ICASE, N, INCX, INCY, MODE, I, SCOMP(I),
     +     STRUE(I), SD, SSIZE(I)
   40 CONTINUE
      RETURN
*
99999 FORMAT ('                                       FAIL')
99998 FORMAT (/' CASE  N INCX INCY MODE  I                            ',
     +       ' COMP(I)                             TRUE(I)  DIFFERENCE',
     +       '     SIZE(I)',/1X)
99997 FORMAT (1X,I4,I3,3I5,I3,2D36.8,2D12.4)
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
      DOUBLE PRECISION  SCOMP1, SFAC, STRUE1
*     .. Array Arguments ..
      DOUBLE PRECISION  SSIZE(*)
*     .. Local Arrays ..
      DOUBLE PRECISION  SCOMP(1), STRUE(1)
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
      DOUBLE PRECISION FUNCTION SDIFF(SA,SB)
*     ********************************* SDIFF **************************
*     COMPUTES DIFFERENCE OF TWO NUMBERS.  C. L. LAWSON, JPL 1974 FEB 15
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION                SA, SB
*     .. Executable Statements ..
      SDIFF = SA - SB
      RETURN
      END
      SUBROUTINE CTEST(LEN,CCOMP,CTRUE,CSIZE,SFAC)
*     **************************** CTEST *****************************
*
*     C.L. LAWSON, JPL, 1978 DEC 6
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION SFAC
      INTEGER          LEN
*     .. Array Arguments ..
      COMPLEX*16       CCOMP(LEN), CSIZE(LEN), CTRUE(LEN)
*     .. Local Scalars ..
      INTEGER          I
*     .. Local Arrays ..
      DOUBLE PRECISION SCOMP(20), SSIZE(20), STRUE(20)
*     .. External Subroutines ..
      EXTERNAL         STEST
*     .. Intrinsic Functions ..
      INTRINSIC        DIMAG, DBLE
*     .. Executable Statements ..
      DO 20 I = 1, LEN
         SCOMP(2*I-1) = DBLE(CCOMP(I))
         SCOMP(2*I) = DIMAG(CCOMP(I))
         STRUE(2*I-1) = DBLE(CTRUE(I))
         STRUE(2*I) = DIMAG(CTRUE(I))
         SSIZE(2*I-1) = DBLE(CSIZE(I))
         SSIZE(2*I) = DIMAG(CSIZE(I))
   20 CONTINUE
*
      CALL STEST(2*LEN,SCOMP,STRUE,SSIZE,SFAC)
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
      INTEGER           ICASE, INCX, INCY, MODE, N
      LOGICAL           PASS
*     .. Local Scalars ..
      INTEGER           ID
*     .. Common blocks ..
      COMMON            /COMBLA/ICASE, N, INCX, INCY, MODE, PASS
*     .. Executable Statements ..
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
      WRITE (NOUT,99997) ICASE, N, INCX, INCY, MODE, ICOMP, ITRUE, ID
   40 CONTINUE
      RETURN
*
99999 FORMAT ('                                       FAIL')
99998 FORMAT (/' CASE  N INCX INCY MODE                               ',
     +       ' COMP                                TRUE     DIFFERENCE',
     +       /1X)
99997 FORMAT (1X,I4,I3,3I5,2I36,I12)
      END
