*> \brief \b CROT applies a plane rotation with real cosine and complex sine to a pair of complex vectors.
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> \htmlonly
*> Download CROT + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/crot.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/crot.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/crot.f">
*> [TXT]</a>
*> \endhtmlonly
*
*  Definition:
*  ===========
*
*       SUBROUTINE CROT( N, CX, INCX, CY, INCY, C, S )
*
*       .. Scalar Arguments ..
*       INTEGER            INCX, INCY, N
*       REAL               C
*       COMPLEX            S
*       ..
*       .. Array Arguments ..
*       COMPLEX            CX( * ), CY( * )
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> CROT   applies a plane rotation, where the cos (C) is real and the
*> sin (S) is complex, and the vectors CX and CY are complex.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The number of elements in the vectors CX and CY.
*> \endverbatim
*>
*> \param[in,out] CX
*> \verbatim
*>          CX is COMPLEX array, dimension (N)
*>          On input, the vector X.
*>          On output, CX is overwritten with C*X + S*Y.
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>          The increment between successive values of CX.  INCX <> 0.
*> \endverbatim
*>
*> \param[in,out] CY
*> \verbatim
*>          CY is COMPLEX array, dimension (N)
*>          On input, the vector Y.
*>          On output, CY is overwritten with -CONJG(S)*X + C*Y.
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>          The increment between successive values of CY.  INCX <> 0.
*> \endverbatim
*>
*> \param[in] C
*> \verbatim
*>          C is REAL
*> \endverbatim
*>
*> \param[in] S
*> \verbatim
*>          S is COMPLEX
*>          C and S define a rotation
*>             [  C          S  ]
*>             [ -conjg(S)   C  ]
*>          where C*C + S*CONJG(S) = 1.0.
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
*> \ingroup complexOTHERauxiliary
*
*  =====================================================================
      SUBROUTINE CROT( N, CX, INCX, CY, INCY, C, S )
*
*  -- LAPACK auxiliary routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      INTEGER            INCX, INCY, N
      REAL               C
      COMPLEX            S
*     ..
*     .. Array Arguments ..
      COMPLEX            CX( * ), CY( * )
*     ..
*
* =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, IX, IY
      COMPLEX            STEMP
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          CONJG
*     ..
*     .. Executable Statements ..
*
      IF( N.LE.0 )
     $   RETURN
      IF( INCX.EQ.1 .AND. INCY.EQ.1 )
     $   GO TO 20
*
*     Code for unequal increments or equal increments not equal to 1
*
      IX = 1
      IY = 1
      IF( INCX.LT.0 )
     $   IX = ( -N+1 )*INCX + 1
      IF( INCY.LT.0 )
     $   IY = ( -N+1 )*INCY + 1
      DO 10 I = 1, N
         STEMP = C*CX( IX ) + S*CY( IY )
         CY( IY ) = C*CY( IY ) - CONJG( S )*CX( IX )
         CX( IX ) = STEMP
         IX = IX + INCX
         IY = IY + INCY
   10 CONTINUE
      RETURN
*
*     Code for both increments equal to 1
*
   20 CONTINUE
      DO 30 I = 1, N
         STEMP = C*CX( I ) + S*CY( I )
         CY( I ) = C*CY( I ) - CONJG( S )*CX( I )
         CX( I ) = STEMP
   30 CONTINUE
      RETURN
      END
