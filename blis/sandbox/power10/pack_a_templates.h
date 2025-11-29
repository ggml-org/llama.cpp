


#define k_even_apack_16(ir) \
            *adest++ = ap[ (i+ir)*rs_a + p_idx*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+1)*cs_a ];

#define k_odd_apack_16(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 2); \
            adest++;
    
#define pad_macro_16(dest_matrix) \
            memset(dest_matrix, 0, 4); \
            dest_matrix+=2; 

#define BIT16_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, a) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int k_odd = k%2; \
    int p_idx; \
\
    DTYPE_IN* adest = apack; \
    for (int i=0; i<m; i+=MR) \
    { \
        int ib = bli_min(MR, m-i); \
        if (ib == MR) /* Full size column height */ \
        { \
            p_idx = 0; \
            for (int p=0; p<(k/2); p++) \
            {  \
                k_even_apack_16(0); \
                k_even_apack_16(1); \
                k_even_apack_16(2); \
                k_even_apack_16(3); \
                k_even_apack_16(4); \
                k_even_apack_16(5); \
                k_even_apack_16(6); \
                k_even_apack_16(7); \
                p_idx += 2; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_odd) \
            { \
                k_odd_apack_16(0); \
                k_odd_apack_16(1); \
                k_odd_apack_16(2); \
                k_odd_apack_16(3); \
                k_odd_apack_16(4); \
                k_odd_apack_16(5); \
                k_odd_apack_16(6); \
                k_odd_apack_16(7); \
            } \
        } \
\
        else /* Not full size, pad with zeros */ \
        { \
            p_idx = 0; \
            for (int p=0; p<(k/2); p++) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_even_apack_16(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_16(adest); \
                } \
                p_idx += 2; \
            } \
\
            if(k_odd) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_odd_apack_16(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_16(adest); \
                } \
            } \
        } \
    } \
} 

/* 8 bit packing routines */

#define k_even_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + p_idx*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+1)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+3)*cs_a ];

#define k_left3_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-3)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 1); \
            adest++;

#define k_left2_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 2); \
            adest += 2;

#define k_left1_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 3); \
            adest += 3;
    
#define pad_macro_8(dest_matrix) \
            memset(dest_matrix, 0, 4); \
            dest_matrix += 4;


#define BIT8_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, a) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int k_left = k%4; \
    int k_iter = k/4; \
    int p_idx; \
\
    DTYPE_IN* adest = apack; \
\
    /* Each panel must be packed in this format */ \
    for (int i=0; i<m; i+=MR) \
    { \
        int ib = bli_min(MR, m-i); \
\
        if (ib == MR) /* Full size column height */ \
        { \
            p_idx = 0; \
            for (int p=0; p<k_iter; p++) \
            {  \
                k_even_apack_8(0); \
                k_even_apack_8(1); \
                k_even_apack_8(2); \
                k_even_apack_8(3); \
                k_even_apack_8(4); \
                k_even_apack_8(5); \
                k_even_apack_8(6); \
                k_even_apack_8(7); \
                p_idx += 4; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_left==3) \
            { \
                k_left3_apack_8(0); \
                k_left3_apack_8(1); \
                k_left3_apack_8(2); \
                k_left3_apack_8(3); \
                k_left3_apack_8(4); \
                k_left3_apack_8(5); \
                k_left3_apack_8(6); \
                k_left3_apack_8(7); \
            } \
            else if(k_left==2) \
            { \
                k_left2_apack_8(0); \
                k_left2_apack_8(1); \
                k_left2_apack_8(2); \
                k_left2_apack_8(3); \
                k_left2_apack_8(4); \
                k_left2_apack_8(5); \
                k_left2_apack_8(6); \
                k_left2_apack_8(7); \
            } \
            else if(k_left==1) \
            { \
                k_left1_apack_8(0); \
                k_left1_apack_8(1); \
                k_left1_apack_8(2); \
                k_left1_apack_8(3); \
                k_left1_apack_8(4); \
                k_left1_apack_8(5); \
                k_left1_apack_8(6); \
                k_left1_apack_8(7); \
            } \
        } \
\
        else /* Not full size, pad with zeros */ \
        { \
            p_idx = 0; \
            for (int p=0; p<k_iter; p++) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_even_apack_8(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_8(adest); \
                } \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left3_apack_8(ir); \
                } \
            } \
            else if(k_left==2) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left2_apack_8(ir); \
                } \
            } \
            else if(k_left==1) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left1_apack_8(ir); \
                } \
            } \
            if(k_left!=0) \
            { \
                for (int ir=ib; ir<MR; ir++) { \
                    pad_macro_8(adest); \
                } \
            } \
        } \
    } \
}


////////////////////////////////////////////////////////////////////////////////
/*                            Packing Routines                                */
////////////////////////////////////////////////////////////////////////////////

/*

    Memory is byte-addressed. This results in two options when dealing with 
    int4. Either store 1 int4 value in a byte, or store 2 int4 values in 1 
    byte. The former is wasteful in storage, but it makes for a simpler
    packing routine. However, we want to not waste any storage if possible. 
    Therefore I went with the latter when designing my int4 kernel. 

    The int4 outerproduct instruction expects a 4x8 matrix in row major order 
    to be loaded into the vector. In order to achieve this 4x8 row major 
    matrix, we pack as many 4x8 panels from the src matrix into the pack matrix.

    To illustrate how my packing routine works:

    x0  x1  x2  x3  x4  x5  x6  x7
    x9  x10 x11 x12 x13 x14 x15 x16
    x17 x18 x19 x20 x21 x22 x23 x24
    x25 x26 x27 x28 x29 x30 x31 x32

    Assume we have a 4x8 matrix that is stored in column major order. Also 
    since we are dealing with int4 values, the values are stored as pairs 
    within a union struct. i.e. (x0, x9) are stored together in the same struct.
    
    Therefore in order to get the desired 4x8 row major matrix, we must go 
    through the first row of structs and grab the first int4 value and insert
    it into the appropriate spot in the pack matrix. This means that after 
    packing, (x0, x1) will be stored together in the same struct.

    This process then repeats until the entire src matrix is packed in these
    4x8 row major matrix panels. 

    To handle edge cases, the packing routine will fill in zeros where it is
    appropriate. 
    
*/

#include "i4_macros.h"

#define PACK_A(ch) \
void PACK_FUNC_NAME(ch, a) \
    ( \
        dim_t MR, \
        int m, int k, \
        uint32_t* ap, int rs_a, int cs_a, \
        uint32_t* apack \
    ) \
{ \
    uint32_t* restrict adest = apack; \
    for( int i=0; i<m; i += MR ) \
    { \
        int ib = min(MR, m-i); \
        if ( ib == MR ) { \
            for ( int p=0; p<k; p++ ) \
                for ( int ir=0; ir<MR; ir++ ) \
                    *adest++ = ap[ (i+ir)*rs_a + p*cs_a ]; \
        } \
        else { \
            for ( int p=0; p<k; p++ ) { \
                for ( int ir=0; ir<ib; ir++ ) \
                    *adest++ = ap[ (i+ir)*rs_a + p*cs_a ]; \
                for ( int ir=ib; ir<MR; ir++ ) \
                    *adest++ = 0.0; \
            } \
        } \
    } \
} 


#define BIT4_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, a) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int p_idx, k_left, k_iter; \
    DTYPE_IN* adest = apack; \
\
    k_left = k%8; \
    k_iter = k/8; \
\
    int i = 0; /* i is used for byte addressing */ \
    for(int int4_i=0; int4_i<m; int4_i+=MR) { /* pack panels */ \
\
        int ib = bli_min(MR, m-int4_i); \
        p_idx = 0; \
\
        if (ib == MR) { /* full size */ \
            for (int p=0; p<k_iter; p++) { \
                col_m_order_1(adest, ap, (i+0), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+0), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+1), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+1), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+2), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+2), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+3), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+3), rs_a, cs_a); \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                apad_col_kleft7(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 6) { \
                apad_col_kleft6(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 5) { \
                apad_col_kleft5(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 4) { \
                apad_col_kleft4(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 3) { \
                apad_col_kleft3(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 2) { \
                apad_col_kleft2(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 1) { \
                apad_col_kleft1(adest, ap, rs_a, cs_a); \
            } \
        } \
\
        else { /* not full size */ \
            for (int p=0; p<k_iter; p++) { \
                for (int ir=0; ir<ib; ir++) { \
                    if (ir%2==0) { \
                        col_m_order_1(adest, ap, (i+ir/2), rs_a, cs_a); \
                    } \
                    else { \
                        col_m_order_2(adest, ap, (i+ir/2), rs_a, cs_a); \
                    } \
                } \
                for (int ir=ib; ir<MR; ir++) { \
                    zero_out_dest(adest); \
                } \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                edge7(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 6) { \
                edge6(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 5) { \
                edge5(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 4) { \
                edge4(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 3) { \
                edge3(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 2) { \
                edge2(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 1) { \
                edge1(adest, ap, i, ib, rs_a, cs_a); \
            } \
\
            /* fill in zeros when an edge case occurs */ \
            if(k_left!=0) \
            { \
                for (int ir=ib; ir<MR; ir++) \
                    zero_out_dest(adest); \
            } \
        } \
        i += (MR/2); \
    } \
}
