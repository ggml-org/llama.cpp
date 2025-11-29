*> \brief \b DBLAT2
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at 
*            http://www.netlib.org/lapack/explore-html/ 
*
*  Definition:
*  ===========
*
*       PROGRAM DBLAT2
* 
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> Test program for the DOUBLE PRECISION Level 2 Blas.
*>
*> The program must be driven by a short data file. The first 18 records
*> of the file are read using list-directed input, the last 16 records
*> are read using the format ( A6, L2 ). An annotated example of a data
*> file can be obtained by deleting the first 3 characters from the
*> following 34 lines:
*> 'dblat2.out'      NAME OF SUMMARY OUTPUT FILE
*> 6                 UNIT NUMBER OF SUMMARY FILE
*> 'DBLAT2.SNAP'     NAME OF SNAPSHOT OUTPUT FILE
*> -1                UNIT NUMBER OF SNAPSHOT FILE (NOT USED IF .LT. 0)
*> F        LOGICAL FLAG, T TO REWIND SNAPSHOT FILE AFTER EACH RECORD.
*> F        LOGICAL FLAG, T TO STOP ON FAILURES.
*> T        LOGICAL FLAG, T TO TEST ERROR EXITS.
*> 16.0     THRESHOLD VALUE OF TEST RATIO
*> 6                 NUMBER OF VALUES OF N
*> 0 1 2 3 5 9       VALUES OF N
*> 4                 NUMBER OF VALUES OF K
*> 0 1 2 4           VALUES OF K
*> 4                 NUMBER OF VALUES OF INCX AND INCY
*> 1 2 -1 -2         VALUES OF INCX AND INCY
*> 3                 NUMBER OF VALUES OF ALPHA
*> 0.0 1.0 0.7       VALUES OF ALPHA
*> 3                 NUMBER OF VALUES OF BETA
*> 0.0 1.0 0.9       VALUES OF BETAC
*> DGEMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DGBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DSYMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DSBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DSPMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTRMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTBMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTPMV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTRSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTBSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DTPSV  T PUT F FOR NO TEST. SAME COLUMNS.
*> DGER   T PUT F FOR NO TEST. SAME COLUMNS.
*> DSYR   T PUT F FOR NO TEST. SAME COLUMNS.
*> DSPR   T PUT F FOR NO TEST. SAME COLUMNS.
*> DSYR2  T PUT F FOR NO TEST. SAME COLUMNS.
*> DSPR2  T PUT F FOR NO TEST. SAME COLUMNS.
*>
*> Further Details
*> ===============
*>
*>    See:
*>
*>       Dongarra J. J., Du Croz J. J., Hammarling S.  and Hanson R. J..
*>       An  extended  set of Fortran  Basic Linear Algebra Subprograms.
*>
*>       Technical  Memoranda  Nos. 41 (revision 3) and 81,  Mathematics
*>       and  Computer Science  Division,  Argonne  National Laboratory,
*>       9700 South Cass Avenue, Argonne, Illinois 60439, US.
*>
*>       Or
*>
*>       NAG  Technical Reports TR3/87 and TR4/87,  Numerical Algorithms
*>       Group  Ltd.,  NAG  Central  Office,  256  Banbury  Road, Oxford
*>       OX2 7DE, UK,  and  Numerical Algorithms Group Inc.,  1101  31st
*>       Street,  Suite 100,  Downers Grove,  Illinois 60515-1263,  USA.
*>
*>
*> -- Written on 10-August-1987.
*>    Richard Hanson, Sandia National Labs.
*>    Jeremy Du Croz, NAG Central Office.
*>
*>    10-9-00:  Change STATUS='NEW' to 'UNKNOWN' so that the testers
*>              can be run multiple times without deleting generated
*>              output files (susan)
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
*> \ingroup double_blas_testing
*
*  =====================================================================
      PROGRAM DBLAT2
*
*  -- Reference BLAS test routine (version 3.4.1) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     April 2012
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            NIN
      PARAMETER          ( NIN = 5 )
      INTEGER            NSUBS
      PARAMETER          ( NSUBS = 16 )
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
      INTEGER            NMAX, INCMAX
      PARAMETER          ( NMAX = 65, INCMAX = 2 )
      INTEGER            NINMAX, NIDMAX, NKBMAX, NALMAX, NBEMAX
      PARAMETER          ( NINMAX = 7, NIDMAX = 9, NKBMAX = 7,
     $                   NALMAX = 7, NBEMAX = 7 )
*     .. Local Scalars ..
      DOUBLE PRECISION   EPS, ERR, THRESH
      INTEGER            I, ISNUM, J, N, NALF, NBET, NIDIM, NINC, NKB,
     $                   NOUT, NTRA
      LOGICAL            FATAL, LTESTT, REWI, SAME, SFATAL, TRACE,
     $                   TSTERR
      CHARACTER*1        TRANS
      CHARACTER*6        SNAMET
      CHARACTER*32       SNAPS, SUMMRY
*     .. Local Arrays ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ),
     $                   ALF( NALMAX ), AS( NMAX*NMAX ), BET( NBEMAX ),
     $                   G( NMAX ), X( NMAX ), XS( NMAX*INCMAX ),
     $                   XX( NMAX*INCMAX ), Y( NMAX ),
     $                   YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX ), Z( 2*NMAX )
      INTEGER            IDIM( NIDMAX ), INC( NINMAX ), KB( NKBMAX )
      LOGICAL            LTEST( NSUBS )
      CHARACTER*6        SNAMES( NSUBS )
*     .. External Functions ..
      DOUBLE PRECISION   DDIFF
      LOGICAL            LDE
      EXTERNAL           DDIFF, LDE
*     .. External Subroutines ..
      EXTERNAL           DCHK1, DCHK2, DCHK3, DCHK4, DCHK5, DCHK6,
     $                   DCHKE, DMVCH
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
      CHARACTER*6        SRNAMT
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
      COMMON             /SRNAMC/SRNAMT
*     .. Data statements ..
      DATA               SNAMES/'DGEMV ', 'DGBMV ', 'DSYMV ', 'DSBMV ',
     $                   'DSPMV ', 'DTRMV ', 'DTBMV ', 'DTPMV ',
     $                   'DTRSV ', 'DTBSV ', 'DTPSV ', 'DGER  ',
     $                   'DSYR  ', 'DSPR  ', 'DSYR2 ', 'DSPR2 '/
*     .. Executable Statements ..
*
*     Read name and unit number for summary output file and open file.
*
      READ( NIN, FMT = * )SUMMRY
      READ( NIN, FMT = * )NOUT
      OPEN( NOUT, FILE = SUMMRY, STATUS = 'UNKNOWN' )
      NOUTC = NOUT
*
*     Read name and unit number for snapshot output file and open file.
*
      READ( NIN, FMT = * )SNAPS
      READ( NIN, FMT = * )NTRA
      TRACE = NTRA.GE.0
      IF( TRACE )THEN
         OPEN( NTRA, FILE = SNAPS, STATUS = 'UNKNOWN' )
      END IF
*     Read the flag that directs rewinding of the snapshot file.
      READ( NIN, FMT = * )REWI
      REWI = REWI.AND.TRACE
*     Read the flag that directs stopping on any failure.
      READ( NIN, FMT = * )SFATAL
*     Read the flag that indicates whether error exits are to be tested.
      READ( NIN, FMT = * )TSTERR
*     Read the threshold value of the test ratio
      READ( NIN, FMT = * )THRESH
*
*     Read and check the parameter values for the tests.
*
*     Values of N
      READ( NIN, FMT = * )NIDIM
      IF( NIDIM.LT.1.OR.NIDIM.GT.NIDMAX )THEN
         WRITE( NOUT, FMT = 9997 )'N', NIDMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( IDIM( I ), I = 1, NIDIM )
      DO 10 I = 1, NIDIM
         IF( IDIM( I ).LT.0.OR.IDIM( I ).GT.NMAX )THEN
            WRITE( NOUT, FMT = 9996 )NMAX
            GO TO 230
         END IF
   10 CONTINUE
*     Values of K
      READ( NIN, FMT = * )NKB
      IF( NKB.LT.1.OR.NKB.GT.NKBMAX )THEN
         WRITE( NOUT, FMT = 9997 )'K', NKBMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( KB( I ), I = 1, NKB )
      DO 20 I = 1, NKB
         IF( KB( I ).LT.0 )THEN
            WRITE( NOUT, FMT = 9995 )
            GO TO 230
         END IF
   20 CONTINUE
*     Values of INCX and INCY
      READ( NIN, FMT = * )NINC
      IF( NINC.LT.1.OR.NINC.GT.NINMAX )THEN
         WRITE( NOUT, FMT = 9997 )'INCX AND INCY', NINMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( INC( I ), I = 1, NINC )
      DO 30 I = 1, NINC
         IF( INC( I ).EQ.0.OR.ABS( INC( I ) ).GT.INCMAX )THEN
            WRITE( NOUT, FMT = 9994 )INCMAX
            GO TO 230
         END IF
   30 CONTINUE
*     Values of ALPHA
      READ( NIN, FMT = * )NALF
      IF( NALF.LT.1.OR.NALF.GT.NALMAX )THEN
         WRITE( NOUT, FMT = 9997 )'ALPHA', NALMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( ALF( I ), I = 1, NALF )
*     Values of BETA
      READ( NIN, FMT = * )NBET
      IF( NBET.LT.1.OR.NBET.GT.NBEMAX )THEN
         WRITE( NOUT, FMT = 9997 )'BETA', NBEMAX
         GO TO 230
      END IF
      READ( NIN, FMT = * )( BET( I ), I = 1, NBET )
*
*     Report values of parameters.
*
      WRITE( NOUT, FMT = 9993 )
      WRITE( NOUT, FMT = 9992 )( IDIM( I ), I = 1, NIDIM )
      WRITE( NOUT, FMT = 9991 )( KB( I ), I = 1, NKB )
      WRITE( NOUT, FMT = 9990 )( INC( I ), I = 1, NINC )
      WRITE( NOUT, FMT = 9989 )( ALF( I ), I = 1, NALF )
      WRITE( NOUT, FMT = 9988 )( BET( I ), I = 1, NBET )
      IF( .NOT.TSTERR )THEN
         WRITE( NOUT, FMT = * )
         WRITE( NOUT, FMT = 9980 )
      END IF
      WRITE( NOUT, FMT = * )
      WRITE( NOUT, FMT = 9999 )THRESH
      WRITE( NOUT, FMT = * )
*
*     Read names of subroutines and flags which indicate
*     whether they are to be tested.
*
      DO 40 I = 1, NSUBS
         LTEST( I ) = .FALSE.
   40 CONTINUE
   50 READ( NIN, FMT = 9984, END = 80 )SNAMET, LTESTT
      DO 60 I = 1, NSUBS
         IF( SNAMET.EQ.SNAMES( I ) )
     $      GO TO 70
   60 CONTINUE
      WRITE( NOUT, FMT = 9986 )SNAMET
      STOP
   70 LTEST( I ) = LTESTT
      GO TO 50
*
   80 CONTINUE
      CLOSE ( NIN )
*
*     Compute EPS (the machine precision).
*
      EPS = EPSILON(ZERO)
      WRITE( NOUT, FMT = 9998 )EPS
*
*     Check the reliability of DMVCH using exact data.
*
      N = MIN( 32, NMAX )
      DO 120 J = 1, N
         DO 110 I = 1, N
            A( I, J ) = MAX( I - J + 1, 0 )
  110    CONTINUE
         X( J ) = J
         Y( J ) = ZERO
  120 CONTINUE
      DO 130 J = 1, N
         YY( J ) = J*( ( J + 1 )*J )/2 - ( ( J + 1 )*J*( J - 1 ) )/3
  130 CONTINUE
*     YY holds the exact result. On exit from DMVCH YT holds
*     the result computed by DMVCH.
      TRANS = 'N'
      CALL DMVCH( TRANS, N, N, ONE, A, NMAX, X, 1, ZERO, Y, 1, YT, G,
     $            YY, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LDE( YY, YT, N )
      IF( .NOT.SAME.OR.ERR.NE.ZERO )THEN
         WRITE( NOUT, FMT = 9985 )TRANS, SAME, ERR
         STOP
      END IF
      TRANS = 'T'
      CALL DMVCH( TRANS, N, N, ONE, A, NMAX, X, -1, ZERO, Y, -1, YT, G,
     $            YY, EPS, ERR, FATAL, NOUT, .TRUE. )
      SAME = LDE( YY, YT, N )
      IF( .NOT.SAME.OR.ERR.NE.ZERO )THEN
         WRITE( NOUT, FMT = 9985 )TRANS, SAME, ERR
         STOP
      END IF
*
*     Test each subroutine in turn.
*
      DO 210 ISNUM = 1, NSUBS
         WRITE( NOUT, FMT = * )
         IF( .NOT.LTEST( ISNUM ) )THEN
*           Subprogram is not to be tested.
            WRITE( NOUT, FMT = 9983 )SNAMES( ISNUM )
         ELSE
            SRNAMT = SNAMES( ISNUM )
*           Test error exits.
            IF( TSTERR )THEN
               CALL DCHKE( ISNUM, SNAMES( ISNUM ), NOUT )
               WRITE( NOUT, FMT = * )
            END IF
*           Test computations.
            INFOT = 0
            OK = .TRUE.
            FATAL = .FALSE.
            GO TO ( 140, 140, 150, 150, 150, 160, 160,
     $              160, 160, 160, 160, 170, 180, 180,
     $              190, 190 )ISNUM
*           Test DGEMV, 01, and DGBMV, 02.
  140       CALL DCHK1( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NKB, KB, NALF, ALF,
     $                  NBET, BET, NINC, INC, NMAX, INCMAX, A, AA, AS,
     $                  X, XX, XS, Y, YY, YS, YT, G )
            GO TO 200
*           Test DSYMV, 03, DSBMV, 04, and DSPMV, 05.
  150       CALL DCHK2( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NKB, KB, NALF, ALF,
     $                  NBET, BET, NINC, INC, NMAX, INCMAX, A, AA, AS,
     $                  X, XX, XS, Y, YY, YS, YT, G )
            GO TO 200
*           Test DTRMV, 06, DTBMV, 07, DTPMV, 08,
*           DTRSV, 09, DTBSV, 10, and DTPSV, 11.
  160       CALL DCHK3( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NKB, KB, NINC, INC,
     $                  NMAX, INCMAX, A, AA, AS, Y, YY, YS, YT, G, Z )
            GO TO 200
*           Test DGER, 12.
  170       CALL DCHK4( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC,
     $                  NMAX, INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS,
     $                  YT, G, Z )
            GO TO 200
*           Test DSYR, 13, and DSPR, 14.
  180       CALL DCHK5( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC,
     $                  NMAX, INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS,
     $                  YT, G, Z )
            GO TO 200
*           Test DSYR2, 15, and DSPR2, 16.
  190       CALL DCHK6( SNAMES( ISNUM ), EPS, THRESH, NOUT, NTRA, TRACE,
     $                  REWI, FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC,
     $                  NMAX, INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS,
     $                  YT, G, Z )
*
  200       IF( FATAL.AND.SFATAL )
     $         GO TO 220
         END IF
  210 CONTINUE
      WRITE( NOUT, FMT = 9982 )
      GO TO 240
*
  220 CONTINUE
      WRITE( NOUT, FMT = 9981 )
      GO TO 240
*
  230 CONTINUE
      WRITE( NOUT, FMT = 9987 )
*
  240 CONTINUE
      IF( TRACE )
     $   CLOSE ( NTRA )
      CLOSE ( NOUT )
      STOP
*
 9999 FORMAT( ' ROUTINES PASS COMPUTATIONAL TESTS IF TEST RATIO IS LES',
     $      'S THAN', F8.2 )
 9998 FORMAT( ' RELATIVE MACHINE PRECISION IS TAKEN TO BE', 1P, D9.1 )
 9997 FORMAT( ' NUMBER OF VALUES OF ', A, ' IS LESS THAN 1 OR GREATER ',
     $      'THAN ', I2 )
 9996 FORMAT( ' VALUE OF N IS LESS THAN 0 OR GREATER THAN ', I2 )
 9995 FORMAT( ' VALUE OF K IS LESS THAN 0' )
 9994 FORMAT( ' ABSOLUTE VALUE OF INCX OR INCY IS 0 OR GREATER THAN ',
     $      I2 )
 9993 FORMAT( ' TESTS OF THE DOUBLE PRECISION LEVEL 2 BLAS', //' THE F',
     $      'OLLOWING PARAMETER VALUES WILL BE USED:' )
 9992 FORMAT( '   FOR N              ', 9I6 )
 9991 FORMAT( '   FOR K              ', 7I6 )
 9990 FORMAT( '   FOR INCX AND INCY  ', 7I6 )
 9989 FORMAT( '   FOR ALPHA          ', 7F6.1 )
 9988 FORMAT( '   FOR BETA           ', 7F6.1 )
 9987 FORMAT( ' AMEND DATA FILE OR INCREASE ARRAY SIZES IN PROGRAM',
     $      /' ******* TESTS ABANDONED *******' )
 9986 FORMAT( ' SUBPROGRAM NAME ', A6, ' NOT RECOGNIZED', /' ******* T',
     $      'ESTS ABANDONED *******' )
 9985 FORMAT( ' ERROR IN DMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALU',
     $      'ATED WRONGLY.', /' DMVCH WAS CALLED WITH TRANS = ', A1,
     $      ' AND RETURNED SAME = ', L1, ' AND ERR = ', F12.3, '.', /
     $   ' THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.'
     $      , /' ******* TESTS ABANDONED *******' )
 9984 FORMAT( A6, L2 )
 9983 FORMAT( 1X, A6, ' WAS NOT TESTED' )
 9982 FORMAT( /' END OF TESTS' )
 9981 FORMAT( /' ******* FATAL ERROR - TESTS ABANDONED *******' )
 9980 FORMAT( ' ERROR-EXITS WILL NOT BE TESTED' )
*
*     End of DBLAT2.
*
      END
      SUBROUTINE DCHK1( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NKB, KB, NALF, ALF, NBET,
     $                  BET, NINC, INC, NMAX, INCMAX, A, AA, AS, X, XX,
     $                  XS, Y, YY, YS, YT, G )
*
*  Tests DGEMV and DGBMV.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NALF, NBET, NIDIM, NINC, NKB, NMAX,
     $                   NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ), ALF( NALF ),
     $                   AS( NMAX*NMAX ), BET( NBET ), G( NMAX ),
     $                   X( NMAX ), XS( NMAX*INCMAX ),
     $                   XX( NMAX*INCMAX ), Y( NMAX ),
     $                   YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX )
      INTEGER            IDIM( NIDIM ), INC( NINC ), KB( NKB )
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, ALS, BETA, BLS, ERR, ERRMAX, TRANSL
      INTEGER            I, IA, IB, IC, IKU, IM, IN, INCX, INCXS, INCY,
     $                   INCYS, IX, IY, KL, KLS, KU, KUS, LAA, LDA,
     $                   LDAS, LX, LY, M, ML, MS, N, NARGS, NC, ND, NK,
     $                   NL, NS
      LOGICAL            BANDED, FULL, NULL, RESET, SAME, TRAN
      CHARACTER*1        TRANS, TRANSS
      CHARACTER*3        ICH
*     .. Local Arrays ..
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DGBMV, DGEMV, DMAKE, DMVCH
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Data statements ..
      DATA               ICH/'NTC'/
*     .. Executable Statements ..
      FULL = SNAME( 3: 3 ).EQ.'E'
      BANDED = SNAME( 3: 3 ).EQ.'B'
*     Define the number of arguments.
      IF( FULL )THEN
         NARGS = 11
      ELSE IF( BANDED )THEN
         NARGS = 13
      END IF
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*
      DO 120 IN = 1, NIDIM
         N = IDIM( IN )
         ND = N/2 + 1
*
         DO 110 IM = 1, 2
            IF( IM.EQ.1 )
     $         M = MAX( N - ND, 0 )
            IF( IM.EQ.2 )
     $         M = MIN( N + ND, NMAX )
*
            IF( BANDED )THEN
               NK = NKB
            ELSE
               NK = 1
            END IF
            DO 100 IKU = 1, NK
               IF( BANDED )THEN
                  KU = KB( IKU )
                  KL = MAX( KU - 1, 0 )
               ELSE
                  KU = N - 1
                  KL = M - 1
               END IF
*              Set LDA to 1 more than minimum value if room.
               IF( BANDED )THEN
                  LDA = KL + KU + 1
               ELSE
                  LDA = M
               END IF
               IF( LDA.LT.NMAX )
     $            LDA = LDA + 1
*              Skip tests if not enough room.
               IF( LDA.GT.NMAX )
     $            GO TO 100
               LAA = LDA*N
               NULL = N.LE.0.OR.M.LE.0
*
*              Generate the matrix A.
*
               TRANSL = ZERO
               CALL DMAKE( SNAME( 2: 3 ), ' ', ' ', M, N, A, NMAX, AA,
     $                     LDA, KL, KU, RESET, TRANSL )
*
               DO 90 IC = 1, 3
                  TRANS = ICH( IC: IC )
                  TRAN = TRANS.EQ.'T'.OR.TRANS.EQ.'C'
*
                  IF( TRAN )THEN
                     ML = N
                     NL = M
                  ELSE
                     ML = M
                     NL = N
                  END IF
*
                  DO 80 IX = 1, NINC
                     INCX = INC( IX )
                     LX = ABS( INCX )*NL
*
*                    Generate the vector X.
*
                     TRANSL = HALF
                     CALL DMAKE( 'GE', ' ', ' ', 1, NL, X, 1, XX,
     $                           ABS( INCX ), 0, NL - 1, RESET, TRANSL )
                     IF( NL.GT.1 )THEN
                        X( NL/2 ) = ZERO
                        XX( 1 + ABS( INCX )*( NL/2 - 1 ) ) = ZERO
                     END IF
*
                     DO 70 IY = 1, NINC
                        INCY = INC( IY )
                        LY = ABS( INCY )*ML
*
                        DO 60 IA = 1, NALF
                           ALPHA = ALF( IA )
*
                           DO 50 IB = 1, NBET
                              BETA = BET( IB )
*
*                             Generate the vector Y.
*
                              TRANSL = ZERO
                              CALL DMAKE( 'GE', ' ', ' ', 1, ML, Y, 1,
     $                                    YY, ABS( INCY ), 0, ML - 1,
     $                                    RESET, TRANSL )
*
                              NC = NC + 1
*
*                             Save every datum before calling the
*                             subroutine.
*
                              TRANSS = TRANS
                              MS = M
                              NS = N
                              KLS = KL
                              KUS = KU
                              ALS = ALPHA
                              DO 10 I = 1, LAA
                                 AS( I ) = AA( I )
   10                         CONTINUE
                              LDAS = LDA
                              DO 20 I = 1, LX
                                 XS( I ) = XX( I )
   20                         CONTINUE
                              INCXS = INCX
                              BLS = BETA
                              DO 30 I = 1, LY
                                 YS( I ) = YY( I )
   30                         CONTINUE
                              INCYS = INCY
*
*                             Call the subroutine.
*
                              IF( FULL )THEN
                                 IF( TRACE )
     $                              WRITE( NTRA, FMT = 9994 )NC, SNAME,
     $                              TRANS, M, N, ALPHA, LDA, INCX, BETA,
     $                              INCY
                                 IF( REWI )
     $                              REWIND NTRA
                                 CALL DGEMV( TRANS, M, N, ALPHA, AA,
     $                                       LDA, XX, INCX, BETA, YY,
     $                                       INCY )
                              ELSE IF( BANDED )THEN
                                 IF( TRACE )
     $                              WRITE( NTRA, FMT = 9995 )NC, SNAME,
     $                              TRANS, M, N, KL, KU, ALPHA, LDA,
     $                              INCX, BETA, INCY
                                 IF( REWI )
     $                              REWIND NTRA
                                 CALL DGBMV( TRANS, M, N, KL, KU, ALPHA,
     $                                       AA, LDA, XX, INCX, BETA,
     $                                       YY, INCY )
                              END IF
*
*                             Check if error-exit was taken incorrectly.
*
                              IF( .NOT.OK )THEN
                                 WRITE( NOUT, FMT = 9993 )
                                 FATAL = .TRUE.
                                 GO TO 130
                              END IF
*
*                             See what data changed inside subroutines.
*
                              ISAME( 1 ) = TRANS.EQ.TRANSS
                              ISAME( 2 ) = MS.EQ.M
                              ISAME( 3 ) = NS.EQ.N
                              IF( FULL )THEN
                                 ISAME( 4 ) = ALS.EQ.ALPHA
                                 ISAME( 5 ) = LDE( AS, AA, LAA )
                                 ISAME( 6 ) = LDAS.EQ.LDA
                                 ISAME( 7 ) = LDE( XS, XX, LX )
                                 ISAME( 8 ) = INCXS.EQ.INCX
                                 ISAME( 9 ) = BLS.EQ.BETA
                                 IF( NULL )THEN
                                    ISAME( 10 ) = LDE( YS, YY, LY )
                                 ELSE
                                    ISAME( 10 ) = LDERES( 'GE', ' ', 1,
     $                                            ML, YS, YY,
     $                                            ABS( INCY ) )
                                 END IF
                                 ISAME( 11 ) = INCYS.EQ.INCY
                              ELSE IF( BANDED )THEN
                                 ISAME( 4 ) = KLS.EQ.KL
                                 ISAME( 5 ) = KUS.EQ.KU
                                 ISAME( 6 ) = ALS.EQ.ALPHA
                                 ISAME( 7 ) = LDE( AS, AA, LAA )
                                 ISAME( 8 ) = LDAS.EQ.LDA
                                 ISAME( 9 ) = LDE( XS, XX, LX )
                                 ISAME( 10 ) = INCXS.EQ.INCX
                                 ISAME( 11 ) = BLS.EQ.BETA
                                 IF( NULL )THEN
                                    ISAME( 12 ) = LDE( YS, YY, LY )
                                 ELSE
                                    ISAME( 12 ) = LDERES( 'GE', ' ', 1,
     $                                            ML, YS, YY,
     $                                            ABS( INCY ) )
                                 END IF
                                 ISAME( 13 ) = INCYS.EQ.INCY
                              END IF
*
*                             If data was incorrectly changed, report
*                             and return.
*
                              SAME = .TRUE.
                              DO 40 I = 1, NARGS
                                 SAME = SAME.AND.ISAME( I )
                                 IF( .NOT.ISAME( I ) )
     $                              WRITE( NOUT, FMT = 9998 )I
   40                         CONTINUE
                              IF( .NOT.SAME )THEN
                                 FATAL = .TRUE.
                                 GO TO 130
                              END IF
*
                              IF( .NOT.NULL )THEN
*
*                                Check the result.
*
                                 CALL DMVCH( TRANS, M, N, ALPHA, A,
     $                                       NMAX, X, INCX, BETA, Y,
     $                                       INCY, YT, G, YY, EPS, ERR,
     $                                       FATAL, NOUT, .TRUE. )
                                 ERRMAX = MAX( ERRMAX, ERR )
*                                If got really bad answer, report and
*                                return.
                                 IF( FATAL )
     $                              GO TO 130
                              ELSE
*                                Avoid repeating tests with M.le.0 or
*                                N.le.0.
                                 GO TO 110
                              END IF
*
   50                      CONTINUE
*
   60                   CONTINUE
*
   70                CONTINUE
*
   80             CONTINUE
*
   90          CONTINUE
*
  100       CONTINUE
*
  110    CONTINUE
*
  120 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 140
*
  130 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      IF( FULL )THEN
         WRITE( NOUT, FMT = 9994 )NC, SNAME, TRANS, M, N, ALPHA, LDA,
     $      INCX, BETA, INCY
      ELSE IF( BANDED )THEN
         WRITE( NOUT, FMT = 9995 )NC, SNAME, TRANS, M, N, KL, KU,
     $      ALPHA, LDA, INCX, BETA, INCY
      END IF
*
  140 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', 4( I3, ',' ), F4.1,
     $      ', A,', I3, ', X,', I2, ',', F4.1, ', Y,', I2, ') .' )
 9994 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', 2( I3, ',' ), F4.1,
     $      ', A,', I3, ', X,', I2, ',', F4.1, ', Y,', I2,
     $      ')         .' )
 9993 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK1.
*
      END
      SUBROUTINE DCHK2( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NKB, KB, NALF, ALF, NBET,
     $                  BET, NINC, INC, NMAX, INCMAX, A, AA, AS, X, XX,
     $                  XS, Y, YY, YS, YT, G )
*
*  Tests DSYMV, DSBMV and DSPMV.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NALF, NBET, NIDIM, NINC, NKB, NMAX,
     $                   NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ), ALF( NALF ),
     $                   AS( NMAX*NMAX ), BET( NBET ), G( NMAX ),
     $                   X( NMAX ), XS( NMAX*INCMAX ),
     $                   XX( NMAX*INCMAX ), Y( NMAX ),
     $                   YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX )
      INTEGER            IDIM( NIDIM ), INC( NINC ), KB( NKB )
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, ALS, BETA, BLS, ERR, ERRMAX, TRANSL
      INTEGER            I, IA, IB, IC, IK, IN, INCX, INCXS, INCY,
     $                   INCYS, IX, IY, K, KS, LAA, LDA, LDAS, LX, LY,
     $                   N, NARGS, NC, NK, NS
      LOGICAL            BANDED, FULL, NULL, PACKED, RESET, SAME
      CHARACTER*1        UPLO, UPLOS
      CHARACTER*2        ICH
*     .. Local Arrays ..
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DMAKE, DMVCH, DSBMV, DSPMV, DSYMV
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Data statements ..
      DATA               ICH/'UL'/
*     .. Executable Statements ..
      FULL = SNAME( 3: 3 ).EQ.'Y'
      BANDED = SNAME( 3: 3 ).EQ.'B'
      PACKED = SNAME( 3: 3 ).EQ.'P'
*     Define the number of arguments.
      IF( FULL )THEN
         NARGS = 10
      ELSE IF( BANDED )THEN
         NARGS = 11
      ELSE IF( PACKED )THEN
         NARGS = 9
      END IF
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*
      DO 110 IN = 1, NIDIM
         N = IDIM( IN )
*
         IF( BANDED )THEN
            NK = NKB
         ELSE
            NK = 1
         END IF
         DO 100 IK = 1, NK
            IF( BANDED )THEN
               K = KB( IK )
            ELSE
               K = N - 1
            END IF
*           Set LDA to 1 more than minimum value if room.
            IF( BANDED )THEN
               LDA = K + 1
            ELSE
               LDA = N
            END IF
            IF( LDA.LT.NMAX )
     $         LDA = LDA + 1
*           Skip tests if not enough room.
            IF( LDA.GT.NMAX )
     $         GO TO 100
            IF( PACKED )THEN
               LAA = ( N*( N + 1 ) )/2
            ELSE
               LAA = LDA*N
            END IF
            NULL = N.LE.0
*
            DO 90 IC = 1, 2
               UPLO = ICH( IC: IC )
*
*              Generate the matrix A.
*
               TRANSL = ZERO
               CALL DMAKE( SNAME( 2: 3 ), UPLO, ' ', N, N, A, NMAX, AA,
     $                     LDA, K, K, RESET, TRANSL )
*
               DO 80 IX = 1, NINC
                  INCX = INC( IX )
                  LX = ABS( INCX )*N
*
*                 Generate the vector X.
*
                  TRANSL = HALF
                  CALL DMAKE( 'GE', ' ', ' ', 1, N, X, 1, XX,
     $                        ABS( INCX ), 0, N - 1, RESET, TRANSL )
                  IF( N.GT.1 )THEN
                     X( N/2 ) = ZERO
                     XX( 1 + ABS( INCX )*( N/2 - 1 ) ) = ZERO
                  END IF
*
                  DO 70 IY = 1, NINC
                     INCY = INC( IY )
                     LY = ABS( INCY )*N
*
                     DO 60 IA = 1, NALF
                        ALPHA = ALF( IA )
*
                        DO 50 IB = 1, NBET
                           BETA = BET( IB )
*
*                          Generate the vector Y.
*
                           TRANSL = ZERO
                           CALL DMAKE( 'GE', ' ', ' ', 1, N, Y, 1, YY,
     $                                 ABS( INCY ), 0, N - 1, RESET,
     $                                 TRANSL )
*
                           NC = NC + 1
*
*                          Save every datum before calling the
*                          subroutine.
*
                           UPLOS = UPLO
                           NS = N
                           KS = K
                           ALS = ALPHA
                           DO 10 I = 1, LAA
                              AS( I ) = AA( I )
   10                      CONTINUE
                           LDAS = LDA
                           DO 20 I = 1, LX
                              XS( I ) = XX( I )
   20                      CONTINUE
                           INCXS = INCX
                           BLS = BETA
                           DO 30 I = 1, LY
                              YS( I ) = YY( I )
   30                      CONTINUE
                           INCYS = INCY
*
*                          Call the subroutine.
*
                           IF( FULL )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9993 )NC, SNAME,
     $                           UPLO, N, ALPHA, LDA, INCX, BETA, INCY
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DSYMV( UPLO, N, ALPHA, AA, LDA, XX,
     $                                    INCX, BETA, YY, INCY )
                           ELSE IF( BANDED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9994 )NC, SNAME,
     $                           UPLO, N, K, ALPHA, LDA, INCX, BETA,
     $                           INCY
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DSBMV( UPLO, N, K, ALPHA, AA, LDA,
     $                                    XX, INCX, BETA, YY, INCY )
                           ELSE IF( PACKED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9995 )NC, SNAME,
     $                           UPLO, N, ALPHA, INCX, BETA, INCY
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DSPMV( UPLO, N, ALPHA, AA, XX, INCX,
     $                                    BETA, YY, INCY )
                           END IF
*
*                          Check if error-exit was taken incorrectly.
*
                           IF( .NOT.OK )THEN
                              WRITE( NOUT, FMT = 9992 )
                              FATAL = .TRUE.
                              GO TO 120
                           END IF
*
*                          See what data changed inside subroutines.
*
                           ISAME( 1 ) = UPLO.EQ.UPLOS
                           ISAME( 2 ) = NS.EQ.N
                           IF( FULL )THEN
                              ISAME( 3 ) = ALS.EQ.ALPHA
                              ISAME( 4 ) = LDE( AS, AA, LAA )
                              ISAME( 5 ) = LDAS.EQ.LDA
                              ISAME( 6 ) = LDE( XS, XX, LX )
                              ISAME( 7 ) = INCXS.EQ.INCX
                              ISAME( 8 ) = BLS.EQ.BETA
                              IF( NULL )THEN
                                 ISAME( 9 ) = LDE( YS, YY, LY )
                              ELSE
                                 ISAME( 9 ) = LDERES( 'GE', ' ', 1, N,
     $                                        YS, YY, ABS( INCY ) )
                              END IF
                              ISAME( 10 ) = INCYS.EQ.INCY
                           ELSE IF( BANDED )THEN
                              ISAME( 3 ) = KS.EQ.K
                              ISAME( 4 ) = ALS.EQ.ALPHA
                              ISAME( 5 ) = LDE( AS, AA, LAA )
                              ISAME( 6 ) = LDAS.EQ.LDA
                              ISAME( 7 ) = LDE( XS, XX, LX )
                              ISAME( 8 ) = INCXS.EQ.INCX
                              ISAME( 9 ) = BLS.EQ.BETA
                              IF( NULL )THEN
                                 ISAME( 10 ) = LDE( YS, YY, LY )
                              ELSE
                                 ISAME( 10 ) = LDERES( 'GE', ' ', 1, N,
     $                                         YS, YY, ABS( INCY ) )
                              END IF
                              ISAME( 11 ) = INCYS.EQ.INCY
                           ELSE IF( PACKED )THEN
                              ISAME( 3 ) = ALS.EQ.ALPHA
                              ISAME( 4 ) = LDE( AS, AA, LAA )
                              ISAME( 5 ) = LDE( XS, XX, LX )
                              ISAME( 6 ) = INCXS.EQ.INCX
                              ISAME( 7 ) = BLS.EQ.BETA
                              IF( NULL )THEN
                                 ISAME( 8 ) = LDE( YS, YY, LY )
                              ELSE
                                 ISAME( 8 ) = LDERES( 'GE', ' ', 1, N,
     $                                        YS, YY, ABS( INCY ) )
                              END IF
                              ISAME( 9 ) = INCYS.EQ.INCY
                           END IF
*
*                          If data was incorrectly changed, report and
*                          return.
*
                           SAME = .TRUE.
                           DO 40 I = 1, NARGS
                              SAME = SAME.AND.ISAME( I )
                              IF( .NOT.ISAME( I ) )
     $                           WRITE( NOUT, FMT = 9998 )I
   40                      CONTINUE
                           IF( .NOT.SAME )THEN
                              FATAL = .TRUE.
                              GO TO 120
                           END IF
*
                           IF( .NOT.NULL )THEN
*
*                             Check the result.
*
                              CALL DMVCH( 'N', N, N, ALPHA, A, NMAX, X,
     $                                    INCX, BETA, Y, INCY, YT, G,
     $                                    YY, EPS, ERR, FATAL, NOUT,
     $                                    .TRUE. )
                              ERRMAX = MAX( ERRMAX, ERR )
*                             If got really bad answer, report and
*                             return.
                              IF( FATAL )
     $                           GO TO 120
                           ELSE
*                             Avoid repeating tests with N.le.0
                              GO TO 110
                           END IF
*
   50                   CONTINUE
*
   60                CONTINUE
*
   70             CONTINUE
*
   80          CONTINUE
*
   90       CONTINUE
*
  100    CONTINUE
*
  110 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 130
*
  120 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      IF( FULL )THEN
         WRITE( NOUT, FMT = 9993 )NC, SNAME, UPLO, N, ALPHA, LDA, INCX,
     $      BETA, INCY
      ELSE IF( BANDED )THEN
         WRITE( NOUT, FMT = 9994 )NC, SNAME, UPLO, N, K, ALPHA, LDA,
     $      INCX, BETA, INCY
      ELSE IF( PACKED )THEN
         WRITE( NOUT, FMT = 9995 )NC, SNAME, UPLO, N, ALPHA, INCX,
     $      BETA, INCY
      END IF
*
  130 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', AP',
     $      ', X,', I2, ',', F4.1, ', Y,', I2, ')                .' )
 9994 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', 2( I3, ',' ), F4.1,
     $      ', A,', I3, ', X,', I2, ',', F4.1, ', Y,', I2,
     $      ')         .' )
 9993 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', A,',
     $      I3, ', X,', I2, ',', F4.1, ', Y,', I2, ')             .' )
 9992 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK2.
*
      END
      SUBROUTINE DCHK3( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NKB, KB, NINC, INC, NMAX,
     $                  INCMAX, A, AA, AS, X, XX, XS, XT, G, Z )
*
*  Tests DTRMV, DTBMV, DTPMV, DTRSV, DTBSV and DTPSV.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF, ONE
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0, ONE = 1.0D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NIDIM, NINC, NKB, NMAX, NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ),
     $                   AS( NMAX*NMAX ), G( NMAX ), X( NMAX ),
     $                   XS( NMAX*INCMAX ), XT( NMAX ),
     $                   XX( NMAX*INCMAX ), Z( NMAX )
      INTEGER            IDIM( NIDIM ), INC( NINC ), KB( NKB )
*     .. Local Scalars ..
      DOUBLE PRECISION   ERR, ERRMAX, TRANSL
      INTEGER            I, ICD, ICT, ICU, IK, IN, INCX, INCXS, IX, K,
     $                   KS, LAA, LDA, LDAS, LX, N, NARGS, NC, NK, NS
      LOGICAL            BANDED, FULL, NULL, PACKED, RESET, SAME
      CHARACTER*1        DIAG, DIAGS, TRANS, TRANSS, UPLO, UPLOS
      CHARACTER*2        ICHD, ICHU
      CHARACTER*3        ICHT
*     .. Local Arrays ..
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DMAKE, DMVCH, DTBMV, DTBSV, DTPMV, DTPSV,
     $                   DTRMV, DTRSV
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Data statements ..
      DATA               ICHU/'UL'/, ICHT/'NTC'/, ICHD/'UN'/
*     .. Executable Statements ..
      FULL = SNAME( 3: 3 ).EQ.'R'
      BANDED = SNAME( 3: 3 ).EQ.'B'
      PACKED = SNAME( 3: 3 ).EQ.'P'
*     Define the number of arguments.
      IF( FULL )THEN
         NARGS = 8
      ELSE IF( BANDED )THEN
         NARGS = 9
      ELSE IF( PACKED )THEN
         NARGS = 7
      END IF
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*     Set up zero vector for DMVCH.
      DO 10 I = 1, NMAX
         Z( I ) = ZERO
   10 CONTINUE
*
      DO 110 IN = 1, NIDIM
         N = IDIM( IN )
*
         IF( BANDED )THEN
            NK = NKB
         ELSE
            NK = 1
         END IF
         DO 100 IK = 1, NK
            IF( BANDED )THEN
               K = KB( IK )
            ELSE
               K = N - 1
            END IF
*           Set LDA to 1 more than minimum value if room.
            IF( BANDED )THEN
               LDA = K + 1
            ELSE
               LDA = N
            END IF
            IF( LDA.LT.NMAX )
     $         LDA = LDA + 1
*           Skip tests if not enough room.
            IF( LDA.GT.NMAX )
     $         GO TO 100
            IF( PACKED )THEN
               LAA = ( N*( N + 1 ) )/2
            ELSE
               LAA = LDA*N
            END IF
            NULL = N.LE.0
*
            DO 90 ICU = 1, 2
               UPLO = ICHU( ICU: ICU )
*
               DO 80 ICT = 1, 3
                  TRANS = ICHT( ICT: ICT )
*
                  DO 70 ICD = 1, 2
                     DIAG = ICHD( ICD: ICD )
*
*                    Generate the matrix A.
*
                     TRANSL = ZERO
                     CALL DMAKE( SNAME( 2: 3 ), UPLO, DIAG, N, N, A,
     $                           NMAX, AA, LDA, K, K, RESET, TRANSL )
*
                     DO 60 IX = 1, NINC
                        INCX = INC( IX )
                        LX = ABS( INCX )*N
*
*                       Generate the vector X.
*
                        TRANSL = HALF
                        CALL DMAKE( 'GE', ' ', ' ', 1, N, X, 1, XX,
     $                              ABS( INCX ), 0, N - 1, RESET,
     $                              TRANSL )
                        IF( N.GT.1 )THEN
                           X( N/2 ) = ZERO
                           XX( 1 + ABS( INCX )*( N/2 - 1 ) ) = ZERO
                        END IF
*
                        NC = NC + 1
*
*                       Save every datum before calling the subroutine.
*
                        UPLOS = UPLO
                        TRANSS = TRANS
                        DIAGS = DIAG
                        NS = N
                        KS = K
                        DO 20 I = 1, LAA
                           AS( I ) = AA( I )
   20                   CONTINUE
                        LDAS = LDA
                        DO 30 I = 1, LX
                           XS( I ) = XX( I )
   30                   CONTINUE
                        INCXS = INCX
*
*                       Call the subroutine.
*
                        IF( SNAME( 4: 5 ).EQ.'MV' )THEN
                           IF( FULL )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9993 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, LDA, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTRMV( UPLO, TRANS, DIAG, N, AA, LDA,
     $                                    XX, INCX )
                           ELSE IF( BANDED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9994 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, K, LDA, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTBMV( UPLO, TRANS, DIAG, N, K, AA,
     $                                    LDA, XX, INCX )
                           ELSE IF( PACKED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9995 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTPMV( UPLO, TRANS, DIAG, N, AA, XX,
     $                                    INCX )
                           END IF
                        ELSE IF( SNAME( 4: 5 ).EQ.'SV' )THEN
                           IF( FULL )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9993 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, LDA, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTRSV( UPLO, TRANS, DIAG, N, AA, LDA,
     $                                    XX, INCX )
                           ELSE IF( BANDED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9994 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, K, LDA, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTBSV( UPLO, TRANS, DIAG, N, K, AA,
     $                                    LDA, XX, INCX )
                           ELSE IF( PACKED )THEN
                              IF( TRACE )
     $                           WRITE( NTRA, FMT = 9995 )NC, SNAME,
     $                           UPLO, TRANS, DIAG, N, INCX
                              IF( REWI )
     $                           REWIND NTRA
                              CALL DTPSV( UPLO, TRANS, DIAG, N, AA, XX,
     $                                    INCX )
                           END IF
                        END IF
*
*                       Check if error-exit was taken incorrectly.
*
                        IF( .NOT.OK )THEN
                           WRITE( NOUT, FMT = 9992 )
                           FATAL = .TRUE.
                           GO TO 120
                        END IF
*
*                       See what data changed inside subroutines.
*
                        ISAME( 1 ) = UPLO.EQ.UPLOS
                        ISAME( 2 ) = TRANS.EQ.TRANSS
                        ISAME( 3 ) = DIAG.EQ.DIAGS
                        ISAME( 4 ) = NS.EQ.N
                        IF( FULL )THEN
                           ISAME( 5 ) = LDE( AS, AA, LAA )
                           ISAME( 6 ) = LDAS.EQ.LDA
                           IF( NULL )THEN
                              ISAME( 7 ) = LDE( XS, XX, LX )
                           ELSE
                              ISAME( 7 ) = LDERES( 'GE', ' ', 1, N, XS,
     $                                     XX, ABS( INCX ) )
                           END IF
                           ISAME( 8 ) = INCXS.EQ.INCX
                        ELSE IF( BANDED )THEN
                           ISAME( 5 ) = KS.EQ.K
                           ISAME( 6 ) = LDE( AS, AA, LAA )
                           ISAME( 7 ) = LDAS.EQ.LDA
                           IF( NULL )THEN
                              ISAME( 8 ) = LDE( XS, XX, LX )
                           ELSE
                              ISAME( 8 ) = LDERES( 'GE', ' ', 1, N, XS,
     $                                     XX, ABS( INCX ) )
                           END IF
                           ISAME( 9 ) = INCXS.EQ.INCX
                        ELSE IF( PACKED )THEN
                           ISAME( 5 ) = LDE( AS, AA, LAA )
                           IF( NULL )THEN
                              ISAME( 6 ) = LDE( XS, XX, LX )
                           ELSE
                              ISAME( 6 ) = LDERES( 'GE', ' ', 1, N, XS,
     $                                     XX, ABS( INCX ) )
                           END IF
                           ISAME( 7 ) = INCXS.EQ.INCX
                        END IF
*
*                       If data was incorrectly changed, report and
*                       return.
*
                        SAME = .TRUE.
                        DO 40 I = 1, NARGS
                           SAME = SAME.AND.ISAME( I )
                           IF( .NOT.ISAME( I ) )
     $                        WRITE( NOUT, FMT = 9998 )I
   40                   CONTINUE
                        IF( .NOT.SAME )THEN
                           FATAL = .TRUE.
                           GO TO 120
                        END IF
*
                        IF( .NOT.NULL )THEN
                           IF( SNAME( 4: 5 ).EQ.'MV' )THEN
*
*                             Check the result.
*
                              CALL DMVCH( TRANS, N, N, ONE, A, NMAX, X,
     $                                    INCX, ZERO, Z, INCX, XT, G,
     $                                    XX, EPS, ERR, FATAL, NOUT,
     $                                    .TRUE. )
                           ELSE IF( SNAME( 4: 5 ).EQ.'SV' )THEN
*
*                             Compute approximation to original vector.
*
                              DO 50 I = 1, N
                                 Z( I ) = XX( 1 + ( I - 1 )*
     $                                    ABS( INCX ) )
                                 XX( 1 + ( I - 1 )*ABS( INCX ) )
     $                              = X( I )
   50                         CONTINUE
                              CALL DMVCH( TRANS, N, N, ONE, A, NMAX, Z,
     $                                    INCX, ZERO, X, INCX, XT, G,
     $                                    XX, EPS, ERR, FATAL, NOUT,
     $                                    .FALSE. )
                           END IF
                           ERRMAX = MAX( ERRMAX, ERR )
*                          If got really bad answer, report and return.
                           IF( FATAL )
     $                        GO TO 120
                        ELSE
*                          Avoid repeating tests with N.le.0.
                           GO TO 110
                        END IF
*
   60                CONTINUE
*
   70             CONTINUE
*
   80          CONTINUE
*
   90       CONTINUE
*
  100    CONTINUE
*
  110 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 130
*
  120 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      IF( FULL )THEN
         WRITE( NOUT, FMT = 9993 )NC, SNAME, UPLO, TRANS, DIAG, N, LDA,
     $      INCX
      ELSE IF( BANDED )THEN
         WRITE( NOUT, FMT = 9994 )NC, SNAME, UPLO, TRANS, DIAG, N, K,
     $      LDA, INCX
      ELSE IF( PACKED )THEN
         WRITE( NOUT, FMT = 9995 )NC, SNAME, UPLO, TRANS, DIAG, N, INCX
      END IF
*
  130 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( 1X, I6, ': ', A6, '(', 3( '''', A1, ''',' ), I3, ', AP, ',
     $      'X,', I2, ')                        .' )
 9994 FORMAT( 1X, I6, ': ', A6, '(', 3( '''', A1, ''',' ), 2( I3, ',' ),
     $      ' A,', I3, ', X,', I2, ')                 .' )
 9993 FORMAT( 1X, I6, ': ', A6, '(', 3( '''', A1, ''',' ), I3, ', A,',
     $      I3, ', X,', I2, ')                     .' )
 9992 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK3.
*
      END
      SUBROUTINE DCHK4( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC, NMAX,
     $                  INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS, YT, G,
     $                  Z )
*
*  Tests DGER.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF, ONE
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0, ONE = 1.0D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NALF, NIDIM, NINC, NMAX, NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ), ALF( NALF ),
     $                   AS( NMAX*NMAX ), G( NMAX ), X( NMAX ),
     $                   XS( NMAX*INCMAX ), XX( NMAX*INCMAX ),
     $                   Y( NMAX ), YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX ), Z( NMAX )
      INTEGER            IDIM( NIDIM ), INC( NINC )
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, ALS, ERR, ERRMAX, TRANSL
      INTEGER            I, IA, IM, IN, INCX, INCXS, INCY, INCYS, IX,
     $                   IY, J, LAA, LDA, LDAS, LX, LY, M, MS, N, NARGS,
     $                   NC, ND, NS
      LOGICAL            NULL, RESET, SAME
*     .. Local Arrays ..
      DOUBLE PRECISION   W( 1 )
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DGER, DMAKE, DMVCH
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Executable Statements ..
*     Define the number of arguments.
      NARGS = 9
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*
      DO 120 IN = 1, NIDIM
         N = IDIM( IN )
         ND = N/2 + 1
*
         DO 110 IM = 1, 2
            IF( IM.EQ.1 )
     $         M = MAX( N - ND, 0 )
            IF( IM.EQ.2 )
     $         M = MIN( N + ND, NMAX )
*
*           Set LDA to 1 more than minimum value if room.
            LDA = M
            IF( LDA.LT.NMAX )
     $         LDA = LDA + 1
*           Skip tests if not enough room.
            IF( LDA.GT.NMAX )
     $         GO TO 110
            LAA = LDA*N
            NULL = N.LE.0.OR.M.LE.0
*
            DO 100 IX = 1, NINC
               INCX = INC( IX )
               LX = ABS( INCX )*M
*
*              Generate the vector X.
*
               TRANSL = HALF
               CALL DMAKE( 'GE', ' ', ' ', 1, M, X, 1, XX, ABS( INCX ),
     $                     0, M - 1, RESET, TRANSL )
               IF( M.GT.1 )THEN
                  X( M/2 ) = ZERO
                  XX( 1 + ABS( INCX )*( M/2 - 1 ) ) = ZERO
               END IF
*
               DO 90 IY = 1, NINC
                  INCY = INC( IY )
                  LY = ABS( INCY )*N
*
*                 Generate the vector Y.
*
                  TRANSL = ZERO
                  CALL DMAKE( 'GE', ' ', ' ', 1, N, Y, 1, YY,
     $                        ABS( INCY ), 0, N - 1, RESET, TRANSL )
                  IF( N.GT.1 )THEN
                     Y( N/2 ) = ZERO
                     YY( 1 + ABS( INCY )*( N/2 - 1 ) ) = ZERO
                  END IF
*
                  DO 80 IA = 1, NALF
                     ALPHA = ALF( IA )
*
*                    Generate the matrix A.
*
                     TRANSL = ZERO
                     CALL DMAKE( SNAME( 2: 3 ), ' ', ' ', M, N, A, NMAX,
     $                           AA, LDA, M - 1, N - 1, RESET, TRANSL )
*
                     NC = NC + 1
*
*                    Save every datum before calling the subroutine.
*
                     MS = M
                     NS = N
                     ALS = ALPHA
                     DO 10 I = 1, LAA
                        AS( I ) = AA( I )
   10                CONTINUE
                     LDAS = LDA
                     DO 20 I = 1, LX
                        XS( I ) = XX( I )
   20                CONTINUE
                     INCXS = INCX
                     DO 30 I = 1, LY
                        YS( I ) = YY( I )
   30                CONTINUE
                     INCYS = INCY
*
*                    Call the subroutine.
*
                     IF( TRACE )
     $                  WRITE( NTRA, FMT = 9994 )NC, SNAME, M, N,
     $                  ALPHA, INCX, INCY, LDA
                     IF( REWI )
     $                  REWIND NTRA
                     CALL DGER( M, N, ALPHA, XX, INCX, YY, INCY, AA,
     $                          LDA )
*
*                    Check if error-exit was taken incorrectly.
*
                     IF( .NOT.OK )THEN
                        WRITE( NOUT, FMT = 9993 )
                        FATAL = .TRUE.
                        GO TO 140
                     END IF
*
*                    See what data changed inside subroutine.
*
                     ISAME( 1 ) = MS.EQ.M
                     ISAME( 2 ) = NS.EQ.N
                     ISAME( 3 ) = ALS.EQ.ALPHA
                     ISAME( 4 ) = LDE( XS, XX, LX )
                     ISAME( 5 ) = INCXS.EQ.INCX
                     ISAME( 6 ) = LDE( YS, YY, LY )
                     ISAME( 7 ) = INCYS.EQ.INCY
                     IF( NULL )THEN
                        ISAME( 8 ) = LDE( AS, AA, LAA )
                     ELSE
                        ISAME( 8 ) = LDERES( 'GE', ' ', M, N, AS, AA,
     $                               LDA )
                     END IF
                     ISAME( 9 ) = LDAS.EQ.LDA
*
*                    If data was incorrectly changed, report and return.
*
                     SAME = .TRUE.
                     DO 40 I = 1, NARGS
                        SAME = SAME.AND.ISAME( I )
                        IF( .NOT.ISAME( I ) )
     $                     WRITE( NOUT, FMT = 9998 )I
   40                CONTINUE
                     IF( .NOT.SAME )THEN
                        FATAL = .TRUE.
                        GO TO 140
                     END IF
*
                     IF( .NOT.NULL )THEN
*
*                       Check the result column by column.
*
                        IF( INCX.GT.0 )THEN
                           DO 50 I = 1, M
                              Z( I ) = X( I )
   50                      CONTINUE
                        ELSE
                           DO 60 I = 1, M
                              Z( I ) = X( M - I + 1 )
   60                      CONTINUE
                        END IF
                        DO 70 J = 1, N
                           IF( INCY.GT.0 )THEN
                              W( 1 ) = Y( J )
                           ELSE
                              W( 1 ) = Y( N - J + 1 )
                           END IF
                           CALL DMVCH( 'N', M, 1, ALPHA, Z, NMAX, W, 1,
     $                                 ONE, A( 1, J ), 1, YT, G,
     $                                 AA( 1 + ( J - 1 )*LDA ), EPS,
     $                                 ERR, FATAL, NOUT, .TRUE. )
                           ERRMAX = MAX( ERRMAX, ERR )
*                          If got really bad answer, report and return.
                           IF( FATAL )
     $                        GO TO 130
   70                   CONTINUE
                     ELSE
*                       Avoid repeating tests with M.le.0 or N.le.0.
                        GO TO 110
                     END IF
*
   80             CONTINUE
*
   90          CONTINUE
*
  100       CONTINUE
*
  110    CONTINUE
*
  120 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 150
*
  130 CONTINUE
      WRITE( NOUT, FMT = 9995 )J
*
  140 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      WRITE( NOUT, FMT = 9994 )NC, SNAME, M, N, ALPHA, INCX, INCY, LDA
*
  150 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( '      THESE ARE THE RESULTS FOR COLUMN ', I3 )
 9994 FORMAT( 1X, I6, ': ', A6, '(', 2( I3, ',' ), F4.1, ', X,', I2,
     $      ', Y,', I2, ', A,', I3, ')                  .' )
 9993 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK4.
*
      END
      SUBROUTINE DCHK5( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC, NMAX,
     $                  INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS, YT, G,
     $                  Z )
*
*  Tests DSYR and DSPR.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF, ONE
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0, ONE = 1.0D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NALF, NIDIM, NINC, NMAX, NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ), ALF( NALF ),
     $                   AS( NMAX*NMAX ), G( NMAX ), X( NMAX ),
     $                   XS( NMAX*INCMAX ), XX( NMAX*INCMAX ),
     $                   Y( NMAX ), YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX ), Z( NMAX )
      INTEGER            IDIM( NIDIM ), INC( NINC )
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, ALS, ERR, ERRMAX, TRANSL
      INTEGER            I, IA, IC, IN, INCX, INCXS, IX, J, JA, JJ, LAA,
     $                   LDA, LDAS, LJ, LX, N, NARGS, NC, NS
      LOGICAL            FULL, NULL, PACKED, RESET, SAME, UPPER
      CHARACTER*1        UPLO, UPLOS
      CHARACTER*2        ICH
*     .. Local Arrays ..
      DOUBLE PRECISION   W( 1 )
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DMAKE, DMVCH, DSPR, DSYR
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Data statements ..
      DATA               ICH/'UL'/
*     .. Executable Statements ..
      FULL = SNAME( 3: 3 ).EQ.'Y'
      PACKED = SNAME( 3: 3 ).EQ.'P'
*     Define the number of arguments.
      IF( FULL )THEN
         NARGS = 7
      ELSE IF( PACKED )THEN
         NARGS = 6
      END IF
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*
      DO 100 IN = 1, NIDIM
         N = IDIM( IN )
*        Set LDA to 1 more than minimum value if room.
         LDA = N
         IF( LDA.LT.NMAX )
     $      LDA = LDA + 1
*        Skip tests if not enough room.
         IF( LDA.GT.NMAX )
     $      GO TO 100
         IF( PACKED )THEN
            LAA = ( N*( N + 1 ) )/2
         ELSE
            LAA = LDA*N
         END IF
*
         DO 90 IC = 1, 2
            UPLO = ICH( IC: IC )
            UPPER = UPLO.EQ.'U'
*
            DO 80 IX = 1, NINC
               INCX = INC( IX )
               LX = ABS( INCX )*N
*
*              Generate the vector X.
*
               TRANSL = HALF
               CALL DMAKE( 'GE', ' ', ' ', 1, N, X, 1, XX, ABS( INCX ),
     $                     0, N - 1, RESET, TRANSL )
               IF( N.GT.1 )THEN
                  X( N/2 ) = ZERO
                  XX( 1 + ABS( INCX )*( N/2 - 1 ) ) = ZERO
               END IF
*
               DO 70 IA = 1, NALF
                  ALPHA = ALF( IA )
                  NULL = N.LE.0.OR.ALPHA.EQ.ZERO
*
*                 Generate the matrix A.
*
                  TRANSL = ZERO
                  CALL DMAKE( SNAME( 2: 3 ), UPLO, ' ', N, N, A, NMAX,
     $                        AA, LDA, N - 1, N - 1, RESET, TRANSL )
*
                  NC = NC + 1
*
*                 Save every datum before calling the subroutine.
*
                  UPLOS = UPLO
                  NS = N
                  ALS = ALPHA
                  DO 10 I = 1, LAA
                     AS( I ) = AA( I )
   10             CONTINUE
                  LDAS = LDA
                  DO 20 I = 1, LX
                     XS( I ) = XX( I )
   20             CONTINUE
                  INCXS = INCX
*
*                 Call the subroutine.
*
                  IF( FULL )THEN
                     IF( TRACE )
     $                  WRITE( NTRA, FMT = 9993 )NC, SNAME, UPLO, N,
     $                  ALPHA, INCX, LDA
                     IF( REWI )
     $                  REWIND NTRA
                     CALL DSYR( UPLO, N, ALPHA, XX, INCX, AA, LDA )
                  ELSE IF( PACKED )THEN
                     IF( TRACE )
     $                  WRITE( NTRA, FMT = 9994 )NC, SNAME, UPLO, N,
     $                  ALPHA, INCX
                     IF( REWI )
     $                  REWIND NTRA
                     CALL DSPR( UPLO, N, ALPHA, XX, INCX, AA )
                  END IF
*
*                 Check if error-exit was taken incorrectly.
*
                  IF( .NOT.OK )THEN
                     WRITE( NOUT, FMT = 9992 )
                     FATAL = .TRUE.
                     GO TO 120
                  END IF
*
*                 See what data changed inside subroutines.
*
                  ISAME( 1 ) = UPLO.EQ.UPLOS
                  ISAME( 2 ) = NS.EQ.N
                  ISAME( 3 ) = ALS.EQ.ALPHA
                  ISAME( 4 ) = LDE( XS, XX, LX )
                  ISAME( 5 ) = INCXS.EQ.INCX
                  IF( NULL )THEN
                     ISAME( 6 ) = LDE( AS, AA, LAA )
                  ELSE
                     ISAME( 6 ) = LDERES( SNAME( 2: 3 ), UPLO, N, N, AS,
     $                            AA, LDA )
                  END IF
                  IF( .NOT.PACKED )THEN
                     ISAME( 7 ) = LDAS.EQ.LDA
                  END IF
*
*                 If data was incorrectly changed, report and return.
*
                  SAME = .TRUE.
                  DO 30 I = 1, NARGS
                     SAME = SAME.AND.ISAME( I )
                     IF( .NOT.ISAME( I ) )
     $                  WRITE( NOUT, FMT = 9998 )I
   30             CONTINUE
                  IF( .NOT.SAME )THEN
                     FATAL = .TRUE.
                     GO TO 120
                  END IF
*
                  IF( .NOT.NULL )THEN
*
*                    Check the result column by column.
*
                     IF( INCX.GT.0 )THEN
                        DO 40 I = 1, N
                           Z( I ) = X( I )
   40                   CONTINUE
                     ELSE
                        DO 50 I = 1, N
                           Z( I ) = X( N - I + 1 )
   50                   CONTINUE
                     END IF
                     JA = 1
                     DO 60 J = 1, N
                        W( 1 ) = Z( J )
                        IF( UPPER )THEN
                           JJ = 1
                           LJ = J
                        ELSE
                           JJ = J
                           LJ = N - J + 1
                        END IF
                        CALL DMVCH( 'N', LJ, 1, ALPHA, Z( JJ ), LJ, W,
     $                              1, ONE, A( JJ, J ), 1, YT, G,
     $                              AA( JA ), EPS, ERR, FATAL, NOUT,
     $                              .TRUE. )
                        IF( FULL )THEN
                           IF( UPPER )THEN
                              JA = JA + LDA
                           ELSE
                              JA = JA + LDA + 1
                           END IF
                        ELSE
                           JA = JA + LJ
                        END IF
                        ERRMAX = MAX( ERRMAX, ERR )
*                       If got really bad answer, report and return.
                        IF( FATAL )
     $                     GO TO 110
   60                CONTINUE
                  ELSE
*                    Avoid repeating tests if N.le.0.
                     IF( N.LE.0 )
     $                  GO TO 100
                  END IF
*
   70          CONTINUE
*
   80       CONTINUE
*
   90    CONTINUE
*
  100 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 130
*
  110 CONTINUE
      WRITE( NOUT, FMT = 9995 )J
*
  120 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      IF( FULL )THEN
         WRITE( NOUT, FMT = 9993 )NC, SNAME, UPLO, N, ALPHA, INCX, LDA
      ELSE IF( PACKED )THEN
         WRITE( NOUT, FMT = 9994 )NC, SNAME, UPLO, N, ALPHA, INCX
      END IF
*
  130 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( '      THESE ARE THE RESULTS FOR COLUMN ', I3 )
 9994 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', X,',
     $      I2, ', AP)                           .' )
 9993 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', X,',
     $      I2, ', A,', I3, ')                        .' )
 9992 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK5.
*
      END
      SUBROUTINE DCHK6( SNAME, EPS, THRESH, NOUT, NTRA, TRACE, REWI,
     $                  FATAL, NIDIM, IDIM, NALF, ALF, NINC, INC, NMAX,
     $                  INCMAX, A, AA, AS, X, XX, XS, Y, YY, YS, YT, G,
     $                  Z )
*
*  Tests DSYR2 and DSPR2.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, HALF, ONE
      PARAMETER          ( ZERO = 0.0D0, HALF = 0.5D0, ONE = 1.0D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   EPS, THRESH
      INTEGER            INCMAX, NALF, NIDIM, NINC, NMAX, NOUT, NTRA
      LOGICAL            FATAL, REWI, TRACE
      CHARACTER*6        SNAME
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, NMAX ), AA( NMAX*NMAX ), ALF( NALF ),
     $                   AS( NMAX*NMAX ), G( NMAX ), X( NMAX ),
     $                   XS( NMAX*INCMAX ), XX( NMAX*INCMAX ),
     $                   Y( NMAX ), YS( NMAX*INCMAX ), YT( NMAX ),
     $                   YY( NMAX*INCMAX ), Z( NMAX, 2 )
      INTEGER            IDIM( NIDIM ), INC( NINC )
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, ALS, ERR, ERRMAX, TRANSL
      INTEGER            I, IA, IC, IN, INCX, INCXS, INCY, INCYS, IX,
     $                   IY, J, JA, JJ, LAA, LDA, LDAS, LJ, LX, LY, N,
     $                   NARGS, NC, NS
      LOGICAL            FULL, NULL, PACKED, RESET, SAME, UPPER
      CHARACTER*1        UPLO, UPLOS
      CHARACTER*2        ICH
*     .. Local Arrays ..
      DOUBLE PRECISION   W( 2 )
      LOGICAL            ISAME( 13 )
*     .. External Functions ..
      LOGICAL            LDE, LDERES
      EXTERNAL           LDE, LDERES
*     .. External Subroutines ..
      EXTERNAL           DMAKE, DMVCH, DSPR2, DSYR2
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Data statements ..
      DATA               ICH/'UL'/
*     .. Executable Statements ..
      FULL = SNAME( 3: 3 ).EQ.'Y'
      PACKED = SNAME( 3: 3 ).EQ.'P'
*     Define the number of arguments.
      IF( FULL )THEN
         NARGS = 9
      ELSE IF( PACKED )THEN
         NARGS = 8
      END IF
*
      NC = 0
      RESET = .TRUE.
      ERRMAX = ZERO
*
      DO 140 IN = 1, NIDIM
         N = IDIM( IN )
*        Set LDA to 1 more than minimum value if room.
         LDA = N
         IF( LDA.LT.NMAX )
     $      LDA = LDA + 1
*        Skip tests if not enough room.
         IF( LDA.GT.NMAX )
     $      GO TO 140
         IF( PACKED )THEN
            LAA = ( N*( N + 1 ) )/2
         ELSE
            LAA = LDA*N
         END IF
*
         DO 130 IC = 1, 2
            UPLO = ICH( IC: IC )
            UPPER = UPLO.EQ.'U'
*
            DO 120 IX = 1, NINC
               INCX = INC( IX )
               LX = ABS( INCX )*N
*
*              Generate the vector X.
*
               TRANSL = HALF
               CALL DMAKE( 'GE', ' ', ' ', 1, N, X, 1, XX, ABS( INCX ),
     $                     0, N - 1, RESET, TRANSL )
               IF( N.GT.1 )THEN
                  X( N/2 ) = ZERO
                  XX( 1 + ABS( INCX )*( N/2 - 1 ) ) = ZERO
               END IF
*
               DO 110 IY = 1, NINC
                  INCY = INC( IY )
                  LY = ABS( INCY )*N
*
*                 Generate the vector Y.
*
                  TRANSL = ZERO
                  CALL DMAKE( 'GE', ' ', ' ', 1, N, Y, 1, YY,
     $                        ABS( INCY ), 0, N - 1, RESET, TRANSL )
                  IF( N.GT.1 )THEN
                     Y( N/2 ) = ZERO
                     YY( 1 + ABS( INCY )*( N/2 - 1 ) ) = ZERO
                  END IF
*
                  DO 100 IA = 1, NALF
                     ALPHA = ALF( IA )
                     NULL = N.LE.0.OR.ALPHA.EQ.ZERO
*
*                    Generate the matrix A.
*
                     TRANSL = ZERO
                     CALL DMAKE( SNAME( 2: 3 ), UPLO, ' ', N, N, A,
     $                           NMAX, AA, LDA, N - 1, N - 1, RESET,
     $                           TRANSL )
*
                     NC = NC + 1
*
*                    Save every datum before calling the subroutine.
*
                     UPLOS = UPLO
                     NS = N
                     ALS = ALPHA
                     DO 10 I = 1, LAA
                        AS( I ) = AA( I )
   10                CONTINUE
                     LDAS = LDA
                     DO 20 I = 1, LX
                        XS( I ) = XX( I )
   20                CONTINUE
                     INCXS = INCX
                     DO 30 I = 1, LY
                        YS( I ) = YY( I )
   30                CONTINUE
                     INCYS = INCY
*
*                    Call the subroutine.
*
                     IF( FULL )THEN
                        IF( TRACE )
     $                     WRITE( NTRA, FMT = 9993 )NC, SNAME, UPLO, N,
     $                     ALPHA, INCX, INCY, LDA
                        IF( REWI )
     $                     REWIND NTRA
                        CALL DSYR2( UPLO, N, ALPHA, XX, INCX, YY, INCY,
     $                              AA, LDA )
                     ELSE IF( PACKED )THEN
                        IF( TRACE )
     $                     WRITE( NTRA, FMT = 9994 )NC, SNAME, UPLO, N,
     $                     ALPHA, INCX, INCY
                        IF( REWI )
     $                     REWIND NTRA
                        CALL DSPR2( UPLO, N, ALPHA, XX, INCX, YY, INCY,
     $                              AA )
                     END IF
*
*                    Check if error-exit was taken incorrectly.
*
                     IF( .NOT.OK )THEN
                        WRITE( NOUT, FMT = 9992 )
                        FATAL = .TRUE.
                        GO TO 160
                     END IF
*
*                    See what data changed inside subroutines.
*
                     ISAME( 1 ) = UPLO.EQ.UPLOS
                     ISAME( 2 ) = NS.EQ.N
                     ISAME( 3 ) = ALS.EQ.ALPHA
                     ISAME( 4 ) = LDE( XS, XX, LX )
                     ISAME( 5 ) = INCXS.EQ.INCX
                     ISAME( 6 ) = LDE( YS, YY, LY )
                     ISAME( 7 ) = INCYS.EQ.INCY
                     IF( NULL )THEN
                        ISAME( 8 ) = LDE( AS, AA, LAA )
                     ELSE
                        ISAME( 8 ) = LDERES( SNAME( 2: 3 ), UPLO, N, N,
     $                               AS, AA, LDA )
                     END IF
                     IF( .NOT.PACKED )THEN
                        ISAME( 9 ) = LDAS.EQ.LDA
                     END IF
*
*                    If data was incorrectly changed, report and return.
*
                     SAME = .TRUE.
                     DO 40 I = 1, NARGS
                        SAME = SAME.AND.ISAME( I )
                        IF( .NOT.ISAME( I ) )
     $                     WRITE( NOUT, FMT = 9998 )I
   40                CONTINUE
                     IF( .NOT.SAME )THEN
                        FATAL = .TRUE.
                        GO TO 160
                     END IF
*
                     IF( .NOT.NULL )THEN
*
*                       Check the result column by column.
*
                        IF( INCX.GT.0 )THEN
                           DO 50 I = 1, N
                              Z( I, 1 ) = X( I )
   50                      CONTINUE
                        ELSE
                           DO 60 I = 1, N
                              Z( I, 1 ) = X( N - I + 1 )
   60                      CONTINUE
                        END IF
                        IF( INCY.GT.0 )THEN
                           DO 70 I = 1, N
                              Z( I, 2 ) = Y( I )
   70                      CONTINUE
                        ELSE
                           DO 80 I = 1, N
                              Z( I, 2 ) = Y( N - I + 1 )
   80                      CONTINUE
                        END IF
                        JA = 1
                        DO 90 J = 1, N
                           W( 1 ) = Z( J, 2 )
                           W( 2 ) = Z( J, 1 )
                           IF( UPPER )THEN
                              JJ = 1
                              LJ = J
                           ELSE
                              JJ = J
                              LJ = N - J + 1
                           END IF
                           CALL DMVCH( 'N', LJ, 2, ALPHA, Z( JJ, 1 ),
     $                                 NMAX, W, 1, ONE, A( JJ, J ), 1,
     $                                 YT, G, AA( JA ), EPS, ERR, FATAL,
     $                                 NOUT, .TRUE. )
                           IF( FULL )THEN
                              IF( UPPER )THEN
                                 JA = JA + LDA
                              ELSE
                                 JA = JA + LDA + 1
                              END IF
                           ELSE
                              JA = JA + LJ
                           END IF
                           ERRMAX = MAX( ERRMAX, ERR )
*                          If got really bad answer, report and return.
                           IF( FATAL )
     $                        GO TO 150
   90                   CONTINUE
                     ELSE
*                       Avoid repeating tests with N.le.0.
                        IF( N.LE.0 )
     $                     GO TO 140
                     END IF
*
  100             CONTINUE
*
  110          CONTINUE
*
  120       CONTINUE
*
  130    CONTINUE
*
  140 CONTINUE
*
*     Report result.
*
      IF( ERRMAX.LT.THRESH )THEN
         WRITE( NOUT, FMT = 9999 )SNAME, NC
      ELSE
         WRITE( NOUT, FMT = 9997 )SNAME, NC, ERRMAX
      END IF
      GO TO 170
*
  150 CONTINUE
      WRITE( NOUT, FMT = 9995 )J
*
  160 CONTINUE
      WRITE( NOUT, FMT = 9996 )SNAME
      IF( FULL )THEN
         WRITE( NOUT, FMT = 9993 )NC, SNAME, UPLO, N, ALPHA, INCX,
     $      INCY, LDA
      ELSE IF( PACKED )THEN
         WRITE( NOUT, FMT = 9994 )NC, SNAME, UPLO, N, ALPHA, INCX, INCY
      END IF
*
  170 CONTINUE
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE COMPUTATIONAL TESTS (', I6, ' CALL',
     $      'S)' )
 9998 FORMAT( ' ******* FATAL ERROR - PARAMETER NUMBER ', I2, ' WAS CH',
     $      'ANGED INCORRECTLY *******' )
 9997 FORMAT( ' ', A6, ' COMPLETED THE COMPUTATIONAL TESTS (', I6, ' C',
     $      'ALLS)', /' ******* BUT WITH MAXIMUM TEST RATIO', F8.2,
     $      ' - SUSPECT *******' )
 9996 FORMAT( ' ******* ', A6, ' FAILED ON CALL NUMBER:' )
 9995 FORMAT( '      THESE ARE THE RESULTS FOR COLUMN ', I3 )
 9994 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', X,',
     $      I2, ', Y,', I2, ', AP)                     .' )
 9993 FORMAT( 1X, I6, ': ', A6, '(''', A1, ''',', I3, ',', F4.1, ', X,',
     $      I2, ', Y,', I2, ', A,', I3, ')                  .' )
 9992 FORMAT( ' ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *',
     $      '******' )
*
*     End of DCHK6.
*
      END
      SUBROUTINE DCHKE( ISNUM, SRNAMT, NOUT )
*
*  Tests the error exits from the Level 2 Blas.
*  Requires a special version of the error-handling routine XERBLA.
*  ALPHA, BETA, A, X and Y should not need to be defined.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      INTEGER            ISNUM, NOUT
      CHARACTER*6        SRNAMT
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUTC
      LOGICAL            LERR, OK
*     .. Local Scalars ..
      DOUBLE PRECISION   ALPHA, BETA
*     .. Local Arrays ..
      DOUBLE PRECISION   A( 1, 1 ), X( 1 ), Y( 1 )
*     .. External Subroutines ..
      EXTERNAL           CHKXER, DGBMV, DGEMV, DGER, DSBMV, DSPMV, DSPR,
     $                   DSPR2, DSYMV, DSYR, DSYR2, DTBMV, DTBSV, DTPMV,
     $                   DTPSV, DTRMV, DTRSV
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUTC, OK, LERR
*     .. Executable Statements ..
*     OK is set to .FALSE. by the special version of XERBLA or by CHKXER
*     if anything is wrong.
      OK = .TRUE.
*     LERR is set to .TRUE. by the special version of XERBLA each time
*     it is called, and is then tested and re-set by CHKXER.
      LERR = .FALSE.
      GO TO ( 10, 20, 30, 40, 50, 60, 70, 80,
     $        90, 100, 110, 120, 130, 140, 150,
     $        160 )ISNUM
   10 INFOT = 1
      CALL DGEMV( '/', 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DGEMV( 'N', -1, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DGEMV( 'N', 0, -1, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 6
      CALL DGEMV( 'N', 2, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL DGEMV( 'N', 0, 0, ALPHA, A, 1, X, 0, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 11
      CALL DGEMV( 'N', 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   20 INFOT = 1
      CALL DGBMV( '/', 0, 0, 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DGBMV( 'N', -1, 0, 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DGBMV( 'N', 0, -1, 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DGBMV( 'N', 0, 0, -1, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DGBMV( 'N', 2, 0, 0, -1, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL DGBMV( 'N', 0, 0, 1, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 10
      CALL DGBMV( 'N', 0, 0, 0, 0, ALPHA, A, 1, X, 0, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 13
      CALL DGBMV( 'N', 0, 0, 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   30 INFOT = 1
      CALL DSYMV( '/', 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSYMV( 'U', -1, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DSYMV( 'U', 2, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DSYMV( 'U', 0, ALPHA, A, 1, X, 0, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 10
      CALL DSYMV( 'U', 0, ALPHA, A, 1, X, 1, BETA, Y, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   40 INFOT = 1
      CALL DSBMV( '/', 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSBMV( 'U', -1, 0, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DSBMV( 'U', 0, -1, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 6
      CALL DSBMV( 'U', 0, 1, ALPHA, A, 1, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL DSBMV( 'U', 0, 0, ALPHA, A, 1, X, 0, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 11
      CALL DSBMV( 'U', 0, 0, ALPHA, A, 1, X, 1, BETA, Y, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   50 INFOT = 1
      CALL DSPMV( '/', 0, ALPHA, A, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSPMV( 'U', -1, ALPHA, A, X, 1, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 6
      CALL DSPMV( 'U', 0, ALPHA, A, X, 0, BETA, Y, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 9
      CALL DSPMV( 'U', 0, ALPHA, A, X, 1, BETA, Y, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   60 INFOT = 1
      CALL DTRMV( '/', 'N', 'N', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTRMV( 'U', '/', 'N', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTRMV( 'U', 'N', '/', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTRMV( 'U', 'N', 'N', -1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 6
      CALL DTRMV( 'U', 'N', 'N', 2, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL DTRMV( 'U', 'N', 'N', 0, A, 1, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   70 INFOT = 1
      CALL DTBMV( '/', 'N', 'N', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTBMV( 'U', '/', 'N', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTBMV( 'U', 'N', '/', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTBMV( 'U', 'N', 'N', -1, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DTBMV( 'U', 'N', 'N', 0, -1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DTBMV( 'U', 'N', 'N', 0, 1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 9
      CALL DTBMV( 'U', 'N', 'N', 0, 0, A, 1, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   80 INFOT = 1
      CALL DTPMV( '/', 'N', 'N', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTPMV( 'U', '/', 'N', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTPMV( 'U', 'N', '/', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTPMV( 'U', 'N', 'N', -1, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DTPMV( 'U', 'N', 'N', 0, A, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
   90 INFOT = 1
      CALL DTRSV( '/', 'N', 'N', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTRSV( 'U', '/', 'N', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTRSV( 'U', 'N', '/', 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTRSV( 'U', 'N', 'N', -1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 6
      CALL DTRSV( 'U', 'N', 'N', 2, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL DTRSV( 'U', 'N', 'N', 0, A, 1, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  100 INFOT = 1
      CALL DTBSV( '/', 'N', 'N', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTBSV( 'U', '/', 'N', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTBSV( 'U', 'N', '/', 0, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTBSV( 'U', 'N', 'N', -1, 0, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DTBSV( 'U', 'N', 'N', 0, -1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DTBSV( 'U', 'N', 'N', 0, 1, A, 1, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 9
      CALL DTBSV( 'U', 'N', 'N', 0, 0, A, 1, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  110 INFOT = 1
      CALL DTPSV( '/', 'N', 'N', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DTPSV( 'U', '/', 'N', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL DTPSV( 'U', 'N', '/', 0, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL DTPSV( 'U', 'N', 'N', -1, A, X, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DTPSV( 'U', 'N', 'N', 0, A, X, 0 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  120 INFOT = 1
      CALL DGER( -1, 0, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DGER( 0, -1, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DGER( 0, 0, ALPHA, X, 0, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DGER( 0, 0, ALPHA, X, 1, Y, 0, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 9
      CALL DGER( 2, 0, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  130 INFOT = 1
      CALL DSYR( '/', 0, ALPHA, X, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSYR( 'U', -1, ALPHA, X, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DSYR( 'U', 0, ALPHA, X, 0, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DSYR( 'U', 2, ALPHA, X, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  140 INFOT = 1
      CALL DSPR( '/', 0, ALPHA, X, 1, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSPR( 'U', -1, ALPHA, X, 1, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DSPR( 'U', 0, ALPHA, X, 0, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  150 INFOT = 1
      CALL DSYR2( '/', 0, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSYR2( 'U', -1, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DSYR2( 'U', 0, ALPHA, X, 0, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DSYR2( 'U', 0, ALPHA, X, 1, Y, 0, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 9
      CALL DSYR2( 'U', 2, ALPHA, X, 1, Y, 1, A, 1 )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      GO TO 170
  160 INFOT = 1
      CALL DSPR2( '/', 0, ALPHA, X, 1, Y, 1, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL DSPR2( 'U', -1, ALPHA, X, 1, Y, 1, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL DSPR2( 'U', 0, ALPHA, X, 0, Y, 1, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL DSPR2( 'U', 0, ALPHA, X, 1, Y, 0, A )
      CALL CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
*
  170 IF( OK )THEN
         WRITE( NOUT, FMT = 9999 )SRNAMT
      ELSE
         WRITE( NOUT, FMT = 9998 )SRNAMT
      END IF
      RETURN
*
 9999 FORMAT( ' ', A6, ' PASSED THE TESTS OF ERROR-EXITS' )
 9998 FORMAT( ' ******* ', A6, ' FAILED THE TESTS OF ERROR-EXITS *****',
     $      '**' )
*
*     End of DCHKE.
*
      END
      SUBROUTINE DMAKE( TYPE, UPLO, DIAG, M, N, A, NMAX, AA, LDA, KL,
     $                  KU, RESET, TRANSL )
*
*  Generates values for an M by N matrix A within the bandwidth
*  defined by KL and KU.
*  Stores the values in the array AA in the data structure required
*  by the routine, with unwanted elements set to rogue value.
*
*  TYPE is 'GE', 'GB', 'SY', 'SB', 'SP', 'TR', 'TB' OR 'TP'.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
      DOUBLE PRECISION   ROGUE
      PARAMETER          ( ROGUE = -1.0D10 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   TRANSL
      INTEGER            KL, KU, LDA, M, N, NMAX
      LOGICAL            RESET
      CHARACTER*1        DIAG, UPLO
      CHARACTER*2        TYPE
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, * ), AA( * )
*     .. Local Scalars ..
      INTEGER            I, I1, I2, I3, IBEG, IEND, IOFF, J, KK
      LOGICAL            GEN, LOWER, SYM, TRI, UNIT, UPPER
*     .. External Functions ..
      DOUBLE PRECISION   DBEG
      EXTERNAL           DBEG
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     .. Executable Statements ..
      GEN = TYPE( 1: 1 ).EQ.'G'
      SYM = TYPE( 1: 1 ).EQ.'S'
      TRI = TYPE( 1: 1 ).EQ.'T'
      UPPER = ( SYM.OR.TRI ).AND.UPLO.EQ.'U'
      LOWER = ( SYM.OR.TRI ).AND.UPLO.EQ.'L'
      UNIT = TRI.AND.DIAG.EQ.'U'
*
*     Generate data in array A.
*
      DO 20 J = 1, N
         DO 10 I = 1, M
            IF( GEN.OR.( UPPER.AND.I.LE.J ).OR.( LOWER.AND.I.GE.J ) )
     $          THEN
               IF( ( I.LE.J.AND.J - I.LE.KU ).OR.
     $             ( I.GE.J.AND.I - J.LE.KL ) )THEN
                  A( I, J ) = DBEG( RESET ) + TRANSL
               ELSE
                  A( I, J ) = ZERO
               END IF
               IF( I.NE.J )THEN
                  IF( SYM )THEN
                     A( J, I ) = A( I, J )
                  ELSE IF( TRI )THEN
                     A( J, I ) = ZERO
                  END IF
               END IF
            END IF
   10    CONTINUE
         IF( TRI )
     $      A( J, J ) = A( J, J ) + ONE
         IF( UNIT )
     $      A( J, J ) = ONE
   20 CONTINUE
*
*     Store elements in array AS in data structure required by routine.
*
      IF( TYPE.EQ.'GE' )THEN
         DO 50 J = 1, N
            DO 30 I = 1, M
               AA( I + ( J - 1 )*LDA ) = A( I, J )
   30       CONTINUE
            DO 40 I = M + 1, LDA
               AA( I + ( J - 1 )*LDA ) = ROGUE
   40       CONTINUE
   50    CONTINUE
      ELSE IF( TYPE.EQ.'GB' )THEN
         DO 90 J = 1, N
            DO 60 I1 = 1, KU + 1 - J
               AA( I1 + ( J - 1 )*LDA ) = ROGUE
   60       CONTINUE
            DO 70 I2 = I1, MIN( KL + KU + 1, KU + 1 + M - J )
               AA( I2 + ( J - 1 )*LDA ) = A( I2 + J - KU - 1, J )
   70       CONTINUE
            DO 80 I3 = I2, LDA
               AA( I3 + ( J - 1 )*LDA ) = ROGUE
   80       CONTINUE
   90    CONTINUE
      ELSE IF( TYPE.EQ.'SY'.OR.TYPE.EQ.'TR' )THEN
         DO 130 J = 1, N
            IF( UPPER )THEN
               IBEG = 1
               IF( UNIT )THEN
                  IEND = J - 1
               ELSE
                  IEND = J
               END IF
            ELSE
               IF( UNIT )THEN
                  IBEG = J + 1
               ELSE
                  IBEG = J
               END IF
               IEND = N
            END IF
            DO 100 I = 1, IBEG - 1
               AA( I + ( J - 1 )*LDA ) = ROGUE
  100       CONTINUE
            DO 110 I = IBEG, IEND
               AA( I + ( J - 1 )*LDA ) = A( I, J )
  110       CONTINUE
            DO 120 I = IEND + 1, LDA
               AA( I + ( J - 1 )*LDA ) = ROGUE
  120       CONTINUE
  130    CONTINUE
      ELSE IF( TYPE.EQ.'SB'.OR.TYPE.EQ.'TB' )THEN
         DO 170 J = 1, N
            IF( UPPER )THEN
               KK = KL + 1
               IBEG = MAX( 1, KL + 2 - J )
               IF( UNIT )THEN
                  IEND = KL
               ELSE
                  IEND = KL + 1
               END IF
            ELSE
               KK = 1
               IF( UNIT )THEN
                  IBEG = 2
               ELSE
                  IBEG = 1
               END IF
               IEND = MIN( KL + 1, 1 + M - J )
            END IF
            DO 140 I = 1, IBEG - 1
               AA( I + ( J - 1 )*LDA ) = ROGUE
  140       CONTINUE
            DO 150 I = IBEG, IEND
               AA( I + ( J - 1 )*LDA ) = A( I + J - KK, J )
  150       CONTINUE
            DO 160 I = IEND + 1, LDA
               AA( I + ( J - 1 )*LDA ) = ROGUE
  160       CONTINUE
  170    CONTINUE
      ELSE IF( TYPE.EQ.'SP'.OR.TYPE.EQ.'TP' )THEN
         IOFF = 0
         DO 190 J = 1, N
            IF( UPPER )THEN
               IBEG = 1
               IEND = J
            ELSE
               IBEG = J
               IEND = N
            END IF
            DO 180 I = IBEG, IEND
               IOFF = IOFF + 1
               AA( IOFF ) = A( I, J )
               IF( I.EQ.J )THEN
                  IF( UNIT )
     $               AA( IOFF ) = ROGUE
               END IF
  180       CONTINUE
  190    CONTINUE
      END IF
      RETURN
*
*     End of DMAKE.
*
      END
      SUBROUTINE DMVCH( TRANS, M, N, ALPHA, A, NMAX, X, INCX, BETA, Y,
     $                  INCY, YT, G, YY, EPS, ERR, FATAL, NOUT, MV )
*
*  Checks the results of the computational tests.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D0, ONE = 1.0D0 )
*     .. Scalar Arguments ..
      DOUBLE PRECISION   ALPHA, BETA, EPS, ERR
      INTEGER            INCX, INCY, M, N, NMAX, NOUT
      LOGICAL            FATAL, MV
      CHARACTER*1        TRANS
*     .. Array Arguments ..
      DOUBLE PRECISION   A( NMAX, * ), G( * ), X( * ), Y( * ), YT( * ),
     $                   YY( * )
*     .. Local Scalars ..
      DOUBLE PRECISION   ERRI
      INTEGER            I, INCXL, INCYL, IY, J, JX, KX, KY, ML, NL
      LOGICAL            TRAN
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, SQRT
*     .. Executable Statements ..
      TRAN = TRANS.EQ.'T'.OR.TRANS.EQ.'C'
      IF( TRAN )THEN
         ML = N
         NL = M
      ELSE
         ML = M
         NL = N
      END IF
      IF( INCX.LT.0 )THEN
         KX = NL
         INCXL = -1
      ELSE
         KX = 1
         INCXL = 1
      END IF
      IF( INCY.LT.0 )THEN
         KY = ML
         INCYL = -1
      ELSE
         KY = 1
         INCYL = 1
      END IF
*
*     Compute expected result in YT using data in A, X and Y.
*     Compute gauges in G.
*
      IY = KY
      DO 30 I = 1, ML
         YT( IY ) = ZERO
         G( IY ) = ZERO
         JX = KX
         IF( TRAN )THEN
            DO 10 J = 1, NL
               YT( IY ) = YT( IY ) + A( J, I )*X( JX )
               G( IY ) = G( IY ) + ABS( A( J, I )*X( JX ) )
               JX = JX + INCXL
   10       CONTINUE
         ELSE
            DO 20 J = 1, NL
               YT( IY ) = YT( IY ) + A( I, J )*X( JX )
               G( IY ) = G( IY ) + ABS( A( I, J )*X( JX ) )
               JX = JX + INCXL
   20       CONTINUE
         END IF
         YT( IY ) = ALPHA*YT( IY ) + BETA*Y( IY )
         G( IY ) = ABS( ALPHA )*G( IY ) + ABS( BETA*Y( IY ) )
         IY = IY + INCYL
   30 CONTINUE
*
*     Compute the error ratio for this result.
*
      ERR = ZERO
      DO 40 I = 1, ML
         ERRI = ABS( YT( I ) - YY( 1 + ( I - 1 )*ABS( INCY ) ) )/EPS
         IF( G( I ).NE.ZERO )
     $      ERRI = ERRI/G( I )
         ERR = MAX( ERR, ERRI )
         IF( ERR*SQRT( EPS ).GE.ONE )
     $      GO TO 50
   40 CONTINUE
*     If the loop completes, all results are at least half accurate.
      GO TO 70
*
*     Report fatal error.
*
   50 FATAL = .TRUE.
      WRITE( NOUT, FMT = 9999 )
      DO 60 I = 1, ML
         IF( MV )THEN
            WRITE( NOUT, FMT = 9998 )I, YT( I ),
     $         YY( 1 + ( I - 1 )*ABS( INCY ) )
         ELSE
            WRITE( NOUT, FMT = 9998 )I,
     $         YY( 1 + ( I - 1 )*ABS( INCY ) ), YT( I )
         END IF
   60 CONTINUE
*
   70 CONTINUE
      RETURN
*
 9999 FORMAT( ' ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HAL',
     $      'F ACCURATE *******', /'           EXPECTED RESULT   COMPU',
     $      'TED RESULT' )
 9998 FORMAT( 1X, I7, 2G18.6 )
*
*     End of DMVCH.
*
      END
      LOGICAL FUNCTION LDE( RI, RJ, LR )
*
*  Tests if two arrays are identical.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      INTEGER            LR
*     .. Array Arguments ..
      DOUBLE PRECISION   RI( * ), RJ( * )
*     .. Local Scalars ..
      INTEGER            I
*     .. Executable Statements ..
      DO 10 I = 1, LR
         IF( RI( I ).NE.RJ( I ) )
     $      GO TO 20
   10 CONTINUE
      LDE = .TRUE.
      GO TO 30
   20 CONTINUE
      LDE = .FALSE.
   30 RETURN
*
*     End of LDE.
*
      END
      LOGICAL FUNCTION LDERES( TYPE, UPLO, M, N, AA, AS, LDA )
*
*  Tests if selected elements in two arrays are equal.
*
*  TYPE is 'GE', 'SY' or 'SP'.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      INTEGER            LDA, M, N
      CHARACTER*1        UPLO
      CHARACTER*2        TYPE
*     .. Array Arguments ..
      DOUBLE PRECISION   AA( LDA, * ), AS( LDA, * )
*     .. Local Scalars ..
      INTEGER            I, IBEG, IEND, J
      LOGICAL            UPPER
*     .. Executable Statements ..
      UPPER = UPLO.EQ.'U'
      IF( TYPE.EQ.'GE' )THEN
         DO 20 J = 1, N
            DO 10 I = M + 1, LDA
               IF( AA( I, J ).NE.AS( I, J ) )
     $            GO TO 70
   10       CONTINUE
   20    CONTINUE
      ELSE IF( TYPE.EQ.'SY' )THEN
         DO 50 J = 1, N
            IF( UPPER )THEN
               IBEG = 1
               IEND = J
            ELSE
               IBEG = J
               IEND = N
            END IF
            DO 30 I = 1, IBEG - 1
               IF( AA( I, J ).NE.AS( I, J ) )
     $            GO TO 70
   30       CONTINUE
            DO 40 I = IEND + 1, LDA
               IF( AA( I, J ).NE.AS( I, J ) )
     $            GO TO 70
   40       CONTINUE
   50    CONTINUE
      END IF
*
      LDERES = .TRUE.
      GO TO 80
   70 CONTINUE
      LDERES = .FALSE.
   80 RETURN
*
*     End of LDERES.
*
      END
      DOUBLE PRECISION FUNCTION DBEG( RESET )
*
*  Generates random numbers uniformly distributed between -0.5 and 0.5.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      LOGICAL            RESET
*     .. Local Scalars ..
      INTEGER            I, IC, MI
*     .. Save statement ..
      SAVE               I, IC, MI
*     .. Intrinsic Functions ..
      INTRINSIC          DBLE
*     .. Executable Statements ..
      IF( RESET )THEN
*        Initialize local variables.
         MI = 891
         I = 7
         IC = 0
         RESET = .FALSE.
      END IF
*
*     The sequence of values of I is bounded between 1 and 999.
*     If initial I = 1,2,3,6,7 or 9, the period will be 50.
*     If initial I = 4 or 8, the period will be 25.
*     If initial I = 5, the period will be 10.
*     IC is used to break up the period by skipping 1 value of I in 6.
*
      IC = IC + 1
   10 I = I*MI
      I = I - 1000*( I/1000 )
      IF( IC.GE.5 )THEN
         IC = 0
         GO TO 10
      END IF
      DBEG = DBLE( I - 500 )/1001.0D0
      RETURN
*
*     End of DBEG.
*
      END
      DOUBLE PRECISION FUNCTION DDIFF( X, Y )
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   X, Y
*     .. Executable Statements ..
      DDIFF = X - Y
      RETURN
*
*     End of DDIFF.
*
      END
      SUBROUTINE CHKXER( SRNAMT, INFOT, NOUT, LERR, OK )
*
*  Tests whether XERBLA has detected an error when it should.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      INTEGER            INFOT, NOUT
      LOGICAL            LERR, OK
      CHARACTER*6        SRNAMT
*     .. Executable Statements ..
      IF( .NOT.LERR )THEN
         WRITE( NOUT, FMT = 9999 )INFOT, SRNAMT
         OK = .FALSE.
      END IF
      LERR = .FALSE.
      RETURN
*
 9999 FORMAT( ' ***** ILLEGAL VALUE OF PARAMETER NUMBER ', I2, ' NOT D',
     $      'ETECTED BY ', A6, ' *****' )
*
*     End of CHKXER.
*
      END
      SUBROUTINE XERBLA( SRNAME, INFO )
*
*  This is a special version of XERBLA to be used only as part of
*  the test program for testing error exits from the Level 2 BLAS
*  routines.
*
*  XERBLA  is an error handler for the Level 2 BLAS routines.
*
*  It is called by the Level 2 BLAS routines if an input parameter is
*  invalid.
*
*  Auxiliary routine for test program for Level 2 Blas.
*
*  -- Written on 10-August-1987.
*     Richard Hanson, Sandia National Labs.
*     Jeremy Du Croz, NAG Central Office.
*
*     .. Scalar Arguments ..
      INTEGER            INFO
      CHARACTER*6        SRNAME
*     .. Scalars in Common ..
      INTEGER            INFOT, NOUT
      LOGICAL            LERR, OK
      CHARACTER*6        SRNAMT
*     .. Common blocks ..
      COMMON             /INFOC/INFOT, NOUT, OK, LERR
      COMMON             /SRNAMC/SRNAMT
*     .. Executable Statements ..
      LERR = .TRUE.
      IF( INFO.NE.INFOT )THEN
         IF( INFOT.NE.0 )THEN
            WRITE( NOUT, FMT = 9999 )INFO, INFOT
         ELSE
            WRITE( NOUT, FMT = 9997 )INFO
         END IF
         OK = .FALSE.
      END IF
      IF( SRNAME.NE.SRNAMT )THEN
         WRITE( NOUT, FMT = 9998 )SRNAME, SRNAMT
         OK = .FALSE.
      END IF
      RETURN
*
 9999 FORMAT( ' ******* XERBLA WAS CALLED WITH INFO = ', I6, ' INSTEAD',
     $      ' OF ', I2, ' *******' )
 9998 FORMAT( ' ******* XERBLA WAS CALLED WITH SRNAME = ', A6, ' INSTE',
     $      'AD OF ', A6, ' *******' )
 9997 FORMAT( ' ******* XERBLA WAS CALLED WITH INFO = ', I6,
     $      ' *******' )
*
*     End of XERBLA
*
      END

