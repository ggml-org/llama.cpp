


#define k_even_bpack_16(jr) \
            *bdest++ = bp[ p_idx*rs_b     + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+1)*rs_b + (j+jr)*cs_b ]; \

#define k_odd_bpack_16(jr) \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 2); \
            bdest++; \

#define BIT16_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, b) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
\
    int k_odd = k%2; \
    int p_idx; \
\
    DTYPE_IN* bdest = bpack; \
\
    for( int j=0; j<n; j += NR ) \
    { \
        int jb = bli_min(NR, n-j); \
\
        if ( jb == NR ) /* Full column width micro-panel.*/  \
        { \
            p_idx = 0; \
            for ( int p=0; p<(k/2); p++ ) \
            { \
                k_even_bpack_16(0); \
                k_even_bpack_16(1); \
                k_even_bpack_16(2); \
                k_even_bpack_16(3); \
                k_even_bpack_16(4); \
                k_even_bpack_16(5); \
                k_even_bpack_16(6); \
                k_even_bpack_16(7); \
                k_even_bpack_16(8); \
                k_even_bpack_16(9); \
                k_even_bpack_16(10); \
                k_even_bpack_16(11); \
                k_even_bpack_16(12); \
                k_even_bpack_16(13); \
                k_even_bpack_16(14); \
                k_even_bpack_16(15); \
                p_idx += 2; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_odd) \
            { \
                k_odd_bpack_16(0); \
                k_odd_bpack_16(1); \
                k_odd_bpack_16(2); \
                k_odd_bpack_16(3); \
                k_odd_bpack_16(4); \
                k_odd_bpack_16(5); \
                k_odd_bpack_16(6); \
                k_odd_bpack_16(7); \
                k_odd_bpack_16(8); \
                k_odd_bpack_16(9); \
                k_odd_bpack_16(10); \
                k_odd_bpack_16(11); \
                k_odd_bpack_16(12); \
                k_odd_bpack_16(13); \
                k_odd_bpack_16(14); \
                k_odd_bpack_16(15); \
            } \
        } \
\
        else /* Not a full row size micro-panel.  We pad with zeroes. */ \
        { \
            p_idx = 0; \
            for ( int p=0; p<(k/2); p++ )  \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_even_bpack_16(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_16(bdest); \
                } \
                p_idx += 2; \
            } \
\
            if(k_odd) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_odd_bpack_16(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_16(bdest); \
                } \
            } \
        } \
    } \
};


#define k_even_bpack_8(jr) \
            *bdest++ = bp[ p_idx*rs_b     + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+1)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+3)*rs_b + (j+jr)*cs_b ]; 

#define k_left3_bpack_8(jr) \
            *bdest++ = bp[ (k-3)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 1); \
            bdest++;

#define k_left2_bpack_8(jr) \
            *bdest++ = bp[ (k-2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 2); \
            bdest+=2;

#define k_left1_bpack_8(jr) \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 3); \
            bdest+=3;


#define BIT8_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, b) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
    int k_left = k%4; \
    int k_iter = k/4; \
    int p_idx; \
\
    DTYPE_IN* bdest = bpack; \
\
    for( int j=0; j<n; j += NR ) \
    { \
        int jb = bli_min(NR, n-j); \
\
        if ( jb == NR ) /* Full column width micro-panel.*/ \
        { \
            p_idx = 0; \
            for ( int p=0; p<k_iter; p++ ) \
            { \
                k_even_bpack_8(0); \
                k_even_bpack_8(1); \
                k_even_bpack_8(2); \
                k_even_bpack_8(3); \
                k_even_bpack_8(4); \
                k_even_bpack_8(5); \
                k_even_bpack_8(6); \
                k_even_bpack_8(7); \
                k_even_bpack_8(8); \
                k_even_bpack_8(9); \
                k_even_bpack_8(10); \
                k_even_bpack_8(11); \
                k_even_bpack_8(12); \
                k_even_bpack_8(13); \
                k_even_bpack_8(14); \
                k_even_bpack_8(15); \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                k_left3_bpack_8(0); \
                k_left3_bpack_8(1); \
                k_left3_bpack_8(2); \
                k_left3_bpack_8(3); \
                k_left3_bpack_8(4); \
                k_left3_bpack_8(5); \
                k_left3_bpack_8(6); \
                k_left3_bpack_8(7); \
                k_left3_bpack_8(8); \
                k_left3_bpack_8(9); \
                k_left3_bpack_8(10); \
                k_left3_bpack_8(11); \
                k_left3_bpack_8(12); \
                k_left3_bpack_8(13); \
                k_left3_bpack_8(14); \
                k_left3_bpack_8(15); \
            } \
            else if(k_left==2) \
            { \
                k_left2_bpack_8(0); \
                k_left2_bpack_8(1); \
                k_left2_bpack_8(2); \
                k_left2_bpack_8(3); \
                k_left2_bpack_8(4); \
                k_left2_bpack_8(5); \
                k_left2_bpack_8(6); \
                k_left2_bpack_8(7); \
                k_left2_bpack_8(8); \
                k_left2_bpack_8(9); \
                k_left2_bpack_8(10); \
                k_left2_bpack_8(11); \
                k_left2_bpack_8(12); \
                k_left2_bpack_8(13); \
                k_left2_bpack_8(14); \
                k_left2_bpack_8(15); \
            } \
            else if(k_left==1)  \
            { \
                k_left1_bpack_8(0); \
                k_left1_bpack_8(1); \
                k_left1_bpack_8(2); \
                k_left1_bpack_8(3); \
                k_left1_bpack_8(4); \
                k_left1_bpack_8(5); \
                k_left1_bpack_8(6); \
                k_left1_bpack_8(7); \
                k_left1_bpack_8(8); \
                k_left1_bpack_8(9); \
                k_left1_bpack_8(10); \
                k_left1_bpack_8(11); \
                k_left1_bpack_8(12); \
                k_left1_bpack_8(13); \
                k_left1_bpack_8(14); \
                k_left1_bpack_8(15); \
            } \
        } \
\
        else /* Not a full row size micro-panel.  We pad with zeroes. */ \
        { \
            p_idx = 0; \
            for ( int p=0; p<k_iter; p++ ) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_even_bpack_8(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_8(bdest); \
                } \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left3_bpack_8(jr); \
                } \
            } \
            else if(k_left==2) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left2_bpack_8(jr); \
                } \
            } \
            else if(k_left==1) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left1_bpack_8(jr); \
                } \
            } \
            if (k_left!=0) \
            { \
                for ( int jr=jb; jr<NR; jr++ ) { \
                    pad_macro_8(bdest); \
                } \
            } \
        } \
    } \
}

#include "i4_macros.h"

#define BIT4_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, b) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
\
    int p_idx, k_left, k_iter; \
    DTYPE_IN* bdest = bpack; \
\
    k_left = k%8; \
    k_iter = k/8; \
\
    int j = 0; \
\
    for(int int4_j=0; int4_j<n; int4_j+=NR) { /* pack panels */ \
        int jb = bli_min(NR, n-int4_j); \
\
        p_idx = 0; \
        if (jb == NR) { /* full size */ \
            for (int p=0; p<k_iter; p++) { \
                col_m_order_1(bdest, bp, (j+0), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+0), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+1), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+1), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+2), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+2), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+3), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+3), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+4), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+4), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+5), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+5), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+6), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+6), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+7), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+7), cs_b, rs_b); \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                bpad_col_kleft7(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 6) { \
                bpad_col_kleft6(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 5) { \
                bpad_col_kleft5(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 4) { \
                bpad_col_kleft4(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 3) { \
                bpad_col_kleft3(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 2) { \
                bpad_col_kleft2(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 1) { \
                bpad_col_kleft1(bdest, bp, cs_b, rs_b); \
            } \
        } \
        else { /* not full size */ \
            for (int p=0; p<k_iter; p++) { \
                for (int jr=0; jr<jb; jr++) { \
                    if (jr%2==0) { \
                        col_m_order_1(bdest, bp, (j+jr/2), cs_b, rs_b); \
                    } \
                    else { \
                        col_m_order_2(bdest, bp, (j+jr/2), cs_b, rs_b); \
                    } \
                } \
                for (int jr=jb; jr<NR; jr++) { \
                    zero_out_dest(bdest); \
                } \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                edge7(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 6) { \
                edge6(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 5) { \
                edge5(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 4) { \
                edge4(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 3) { \
                edge3(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 2) { \
                edge2(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 1) { \
                edge1(bdest, bp, j, jb, cs_b, rs_b); \
            } \
\
            /* fill in zeros when an edge case occurs */ \
            if(k_left!=0) \
            { \
                for (int ir=jb; ir<NR; ir++) \
                    zero_out_dest(bdest); \
            } \
        } \
        j += (NR/2); \
    } \
}
