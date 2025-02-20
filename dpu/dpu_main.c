/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#define PIM_KERNEL_DPU 1
#include "../ggml/include/ggml.h"
#define GGML_COMMON_DECL_C
#include "../ggml/src/ggml-common.h"

#define PRINT 0

__mram_ptr float *ptable_f32_f16;

inline static float lookup_fp16_to_fp32(uint16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    uint16_t alignedOffset;
    float temp[8];

    alignedOffset = s & 0xfff8;
    mram_read((__mram_ptr void const*) (DPU_MRAM_HEAP_POINTER+sizeof(float)*alignedOffset), temp, sizeof(float)*8);
    return temp[s & 0x7];
}
#define FP16_TO_FP32(x) lookup_fp16_to_fp32(x)

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

/*
DPU MRAM Memory:

|--Quantify-tbl--  |--DPU0-weight-Metadata--  |--layer0-subweight0--pading--  |--layer1-subweight0--pading--  |...|--layer31-subweight0--pading--  |--input-output-metadata--|--input-token--|---output0--pading--|
|--Quantify-tbl--  |--DPU1-weight-Metadata--  |--layer0-subweight1--pading--  |--layer1-subweight1--pading--  |...|--layer31-subweight1--pading--  |--input-output-metadata--|--input-token--|---output1--pading--|
......
|--Quantify-tbl--  |--DPU127-weight-Metadata--|--layer0-subweight127--pading--|--layer1-subweight127--pading--|...|--layer31-subweight127--pading--|--input-output-metadata--|--input-token--|---output127--pading--|
*/
#define BLOCK_SIZE (1 << BL)

int mram2wram(__mram_ptr void *pmram,void *pwram,uint32_t size)
{
    uint32_t rest_size = size;
    uint32_t index = 0;
    __mram_ptr void *from;
    void *to;
    while (rest_size >= BLOCK_SIZE) {
        from = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        to = (void *)(((unsigned char *)pwram) + index);
        mram_read(from, to, BLOCK_SIZE);
        rest_size -= BLOCK_SIZE;
        index += BLOCK_SIZE;
    }

    if (rest_size) {
        from = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        to = (void *)(((unsigned char *)pwram) + index);
        mram_read(from, to, rest_size);
    }
    return 0;
}

int wram2mram(__mram_ptr void *pmram,void *pwram,uint32_t size)
{
    uint32_t rest_size = size;
    uint32_t index = 0;
    __mram_ptr void *to;
    void *from;
    while (rest_size >= BLOCK_SIZE) {
        to = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        from = (void *)(((unsigned char *)pwram) + index);
        mram_write(from, to, BLOCK_SIZE);
        rest_size -= BLOCK_SIZE;
        index += BLOCK_SIZE;
    }

    if (rest_size) {
        to = (__mram_ptr void *)(((unsigned char *)pmram) + index);
        from = (void *)(((unsigned char *)pwram) + index);
        mram_write(from, to, rest_size);
    }
    return 0;
}


static float *psumf = NULL;

void init(unsigned int tasklet_id) {
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap

        ptable_f32_f16 = (__mram_ptr float *)DPU_MRAM_HEAP_POINTER;
    }
    // Barrier
    barrier_wait(&my_barrier);

    // ptable_f32_f16 = (__mram_ptr float *)DPU_MRAM_HEAP_POINTER;
}

// main
int main() {

    unsigned int tasklet_id = me();
    
    init(tasklet_id);
    
    //fp32->fp16 table
    ptable_f32_f16 = (__mram_ptr float *)DPU_MRAM_HEAP_POINTER;
    uint32_t table_f32_f16_len = (1 << 16)*sizeof(float);
    uint32_t offset = table_f32_f16_len;
    int input_row_size = 0;
    int input_cols = 0;
    

#if PRINT
    printf("table_f32_f16_len=%d\n",table_f32_f16_len);

    for (int uuu=0;uuu<16;uuu++) {
        printf("FP16_TO_FP32[%d]=%f\n",uuu,FP16_TO_FP32(uuu));
    }
#endif

    //weight metadata
    uint32_t weightmetadatabase = (uint32_t) (DPU_MRAM_HEAP_POINTER + offset);
    struct pim_meta *cache_meta = (struct pim_meta *) mem_alloc(sizeof(struct pim_meta));
    mram_read((__mram_ptr void const*) (weightmetadatabase), cache_meta, sizeof(struct pim_meta));

#if PRINT
    printf("layer_num: %d, weight_type=%d, rows_per_dpu=%d, rest_rows=%d, input_offset=%d",
        cache_meta->layer_num,cache_meta->weight_type,cache_meta->rows_per_dpu,cache_meta->rest_rows,cache_meta->input_offset);
#endif

    // 先不考虑尾行
    uint16_t weight_rows_per_thread = cache_meta->rows_per_dpu / NR_TASKLETS;
    uint16_t weight_start_row = tasklet_id * weight_rows_per_thread;
    uint16_t weight_end_row = weight_start_row + weight_rows_per_thread;

    // todo:rest row is existed, first thread in every dpu can one more row
    uint16_t weight_rows_cur_thread;
    if (cache_meta->rest_rows) {
        ;
    }
    else
    {
        weight_rows_cur_thread = cache_meta->rows_per_dpu;
    }
    offset += sizeof(struct pim_meta);

    //input metadata
    offset += (cache_meta->layer_len * cache_meta->layer_num);
#if PRINT
    printf("layer_len=%d, input metadata offset=%d\n",cache_meta->layer_len,offset);
#endif
    uint32_t inputmetadatabase = weightmetadatabase + sizeof(struct pim_meta) + cache_meta->layer_len * cache_meta->layer_num;
    pim_matrix_des *pinputcache = (pim_matrix_des *) mem_alloc(sizeof(pim_matrix_des));
    mram_read((__mram_ptr void const*) (inputmetadatabase), pinputcache, sizeof(pim_matrix_des));
    input_cols = pinputcache->ne[1];
    assert(input_cols == 1 && "Only support vector as input.");
#if PRINT
    printf("input_type=%d, layerID=%d\n",pinputcache->type,pinputcache->layerid);
    for(int nn=0;nn<GGML_MAX_DIMS;nn++) {
        printf("ne[%d]=%lld\n",nn,pinputcache->ne[nn]);
    }
#endif

    assert(cache_meta->weight_type == ((uint16_t)GGML_TYPE_Q4_0) && "Only support Q4_0 weight.");
    //weight info: GGML_TYPE_Q4_0 default
    if (cache_meta->weight_type == ((uint16_t)GGML_TYPE_Q4_0)) {
        if (pinputcache->type != GGML_TYPE_Q8_0) {
            printf("weight type is GGML_TYPE_Q4_0,input must be GGML_TYPE_Q8_0,now input is %d\n",pinputcache->type);
            return -1;
        }
        int nb = pinputcache->ne[0]/QK8_0;
        int qk = QK8_0;
        input_row_size = nb*sizeof(block_q8_0);
        __mram_ptr void *pweight_base = (__mram_ptr void *)(weightmetadatabase + sizeof(struct pim_meta));
        __mram_ptr void *pinput_base = DPU_MRAM_HEAP_POINTER + cache_meta->input_offset + sizeof(pim_matrix_des);

        if (tasklet_id == 0) {
            psumf = (float *)mem_alloc(sizeof(float)*input_cols*weight_rows_cur_thread);
        }
        barrier_wait(&my_barrier);

        // psumf = (float *)mem_alloc(sizeof(float)*input_cols*weight_rows_cur_thread);
        memset(psumf, 0 ,sizeof(float)*input_cols*weight_rows_cur_thread);
#if PRINT
        printf("input_cols=%d, rows_cur_thread=%d, nb=%d, input_row_size=%d\n",input_cols,weight_rows_cur_thread,nb,input_row_size);
#endif
        block_q4_0 *pweight_cache = (block_q4_0 *) mem_alloc(sizeof(block_q4_0)*nb);
        block_q8_0 *pinput_cache = (block_q8_0 *) mem_alloc(sizeof(block_q8_0)*nb);

        // weight_rows_cur_thread = 16;
        for(int l = 0;l < input_cols;l++) {
          __mram_ptr block_q8_0 *pinput = pinput_base + l * nb * sizeof(block_q8_0);
            mram2wram(pinput, pinput_cache, sizeof(block_q8_0)*nb);
#if PRINT
            printf("input:\n");
            for (int i = 0; i < nb; i++) {
              printf("d=%u\n",pinput[i].d);
              for (int kkk=0;kkk<QK8_0;kkk++) {
                printf("%d ",pinput[i].qs[kkk]);
              }
            printf("\n");
            }
            printf("pweight_base: %p\n", pweight_base);
#endif
            // for(int k = 0;k < weight_rows_cur_thread;k++) {
            for (int k = weight_start_row; k < weight_end_row; ++k) {
              __mram_ptr block_q4_0 *pweight = pweight_base + pinputcache->layerid * cache_meta->layer_len + k * nb * sizeof(block_q4_0);
                mram2wram(pweight, pweight_cache, sizeof(block_q4_0)*nb);
#if PRINT
                if (k % 64 == 0) {
                  printf("pweight_cache[%d].d=%d\n pweight_cache[%d].qs=", k*128, pweight_cache[0].d, k*128);
                  for (int kkk=0;kkk<QK4_0/2;kkk++) {
                    int v0 = (pweight_cache[0].qs[kkk] & 0x0f) - 8;
                    int v1 = (pweight_cache[0].qs[kkk]  >> 4) - 8;
                    printf(" %d, %d", v0, v1);
                  }
                  printf("\n");
                }
#endif

                for (int i = 0; i < nb; i++) {
                    //printf("input_col:%d, current inner weight row idx:%d\n",l,k);

                    int sumi = 0;
                    for (int j = 0; j < qk/2; ++j) {
                        const int v0 = (pweight_cache[i].qs[j] & 0x0F) - 8;
                        const int v1 = (pweight_cache[i].qs[j] >>   4) - 8;

                        sumi += (v0 * pinput_cache[i].qs[j]) + (v1 * pinput_cache[i].qs[j + qk/2]);
                    }

                    psumf[l*weight_rows_cur_thread + k] += sumi*FP16_TO_FP32(pweight_cache[i].d)*FP16_TO_FP32(pinput_cache[i].d);
                }
            }
        }
    }

    offset += (sizeof(pim_matrix_des) + input_row_size * input_cols);
#if PRINT
    for(int iii=0;iii<cache_meta->rows_per_dpu;iii+=128) {
        printf("psumf[%d]=%f\n",iii,psumf[iii]);
    }

    printf("output offset=%d\n",offset);
#endif
    // Write C Matrix to current MRAM block
    // Note: with input_cols > 1, the results should be rearranged on host
    wram2mram((__mram_ptr void *) (DPU_MRAM_HEAP_POINTER + offset), psumf, sizeof(float)*input_cols*weight_rows_cur_thread);
    return 0;
}
