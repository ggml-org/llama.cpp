#include "ggml.h"
#include "ggml-log.h"

#include "ggml-blas.h"

#include "ggml_cpp_wrapper.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <string>
#include <algorithm>

#if defined(GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif

#if defined(GGML_BLAS_USE_FLEXIBLAS)
#   include <flexiblas_api.h>
#endif

#ifdef GGML_USE_OPENMP
#   include <omp.h>
#endif

namespace ggml::backend::blas {
    
    static constexpr std::size_t MEMORY_ALIGNMENT = 64; // 512 bits

    // backend class
    class backend : public ggml::cpp::backend::backend {

        int n_threads = GGML_DEFAULT_N_THREADS;

        std::unique_ptr<char[]> work_data;
        size_t work_size = 0;

        // for tensor convert (TODO: remove work_data)
        //  TODO: have a stack off buffer if we need 2+ work_data
        void* m_work_data = nullptr;
        std::size_t m_work_size = 0;
        template<typename T>
        T* get_work(std::size_t size) {
            std::size_t nb_byte = size * sizeof(T);
            if (nb_byte > m_work_size) {
                nb_byte = std::max(nb_byte , 2*m_work_size);
                // force "aligned size"
                nb_byte  = ((nb_byte-1)/MEMORY_ALIGNMENT)+1;
                nb_byte *= MEMORY_ALIGNMENT;
                if (m_work_data) std::free(m_work_data);
                m_work_size = nb_byte;
                m_work_data = aligned_alloc(MEMORY_ALIGNMENT, m_work_size);
            }
            return (T*) m_work_data;
        }

#ifndef GGML_USE_OPENMP
        std::vector<std::future<void>> tasks;
#endif

    private:
    
    //void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, 
    //                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
    //                 OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
    //                 OPENBLAS_CONST blasint M,
    //                 OPENBLAS_CONST blasint N,
    //                 OPENBLAS_CONST blasint K,
    //                 OPENBLAS_CONST float alpha,
    //                 OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda,
    //                 OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb,
    //                 OPENBLAS_CONST float beta,
    //                                float *C, OPENBLAS_CONST blasint ldc);
    
    //void cblas_sgemm_batch(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransA_array, OPENBLAS_CONST enum CBLAS_TRANSPOSE * TransB_array, OPENBLAS_CONST blasint * M_array, OPENBLAS_CONST blasint * N_array, OPENBLAS_CONST blasint * K_array,
    //               OPENBLAS_CONST float * alpha_array, OPENBLAS_CONST float ** A_array, OPENBLAS_CONST blasint * lda_array, OPENBLAS_CONST float ** B_array, OPENBLAS_CONST blasint * ldb_array, OPENBLAS_CONST float * beta_array, float ** C_array, OPENBLAS_CONST blasint * ldc_array, OPENBLAS_CONST blasint group_count, OPENBLAS_CONST blasint * group_size);
    //void cblas_sgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST float * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);

    //void cblas_sbgemm_batch_strided(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 * A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST blasint stridea, OPENBLAS_CONST bfloat16 * B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST blasint strideb, OPENBLAS_CONST float beta, float * C, OPENBLAS_CONST blasint ldc, OPENBLAS_CONST blasint stridec, OPENBLAS_CONST blasint group_size);
#ifdef GGML_BLAS_USE_SBGEMM
        void sbgemm(const ggml_tensor & A, const ggml_tensor & B, ggml_tensor & C) {
            GGML_ASSERT(A.ne[0] == B.ne[0]); // K
            GGML_ASSERT(B.ne[1] == C.ne[1]); // N
            GGML_ASSERT(A.ne[1] == C.ne[0]); // M
            // for now!
            GGML_ASSERT(A.type == GGML_TYPE_BF16);
            GGML_ASSERT(B.type == GGML_TYPE_F32);
            GGML_ASSERT(C.type == GGML_TYPE_F32);
            
            // convert B to BF16:
            // - B contigue: (TODO: other case?)
            GGML_ASSERT(((size_t)4*B.ne[0]*B.ne[1]*B.ne[2]) == B.nb[3]);
            std::size_t sizeB = B.ne[0]*B.ne[1]*B.ne[2]*B.ne[3];
            auto* B_work = get_work<bfloat16>(std::max(sizeB, B.ne[0]*(std::size_t)256));
            cblas_sbstobf16(sizeB, (const float*)B.data, 1, B_work, 1);
            
            // compute:
            if (B.ne[2]*B.ne[3] == 1) {
                if (B.ne[1] == 1) {
                    cblas_sbgemv(CblasRowMajor, CblasNoTrans,
                                 A.ne[1], A.ne[0],
                                 1.0f,   (const bfloat16*)A.data, A.nb[1]/2,
                                                          B_work,         1,
                                 0.0f,   (      float*   )C.data,         1);
                } else {
                    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                 C.ne[1], C.ne[0], B.ne[0],
                                 1.0f,                    B_work, B.nb[1]/4,
                                         (const bfloat16*)A.data, A.nb[1]/2,
                                 0.0f,   (      float*   )C.data, C.nb[1]/4);
                }
            } else {
                // use batched gemm
                const blasint nb_group = C.ne[2]*C.ne[3];
                auto transB = CblasNoTrans;
                auto transA = CblasTrans;
                const blasint M = C.ne[1];
                const blasint N = C.ne[0];
                const blasint K = B.ne[0];
                const blasint ldA = A.nb[1]/2;
                const blasint ldB = B.nb[1]/4;
                const blasint ldC = C.nb[1]/4;
                static constexpr float alpha = 1;
                static constexpr float beta = 0;
                // TODO: reduce A.nb[2] => 1
                std::vector<      float    *> C_fp32_v (nb_group, nullptr);
                std::vector<const bfloat16 *> A_bf16_v (nb_group, nullptr);
                std::vector<const bfloat16 *> B_bf16_v (nb_group, nullptr);

                // Tensors config to batched params
                const bfloat16 * A_bf16 = (const bfloat16 *) A.data;
                const bfloat16 * B_bf16 = (const bfloat16 *) B_work;
                      float    * C_fp32 = (      float    *) C.data;
                // A[3/2] is broadcasted...
                const int64_t r2 = B.ne[2]/A.ne[2];
                const int64_t r3 = B.ne[3]/A.ne[3];
                const std::size_t lda2 = A.nb[2]/A.nb[0];
                const std::size_t lda3 = A.nb[3]/A.nb[0];
                const std::size_t ldb2 = B.nb[2]/4;
                const std::size_t ldb3 = B.nb[3]/4;
                const std::size_t ldc2 = C.nb[2]/C.nb[0];
                const std::size_t ldc3 = C.nb[3]/C.nb[0];

                for (int64_t j3 = 0; j3 < C.ne[3]; ++j3) {
                    for (int64_t j2 = 0; j2 < C.ne[2]; ++j2) {
                        auto lda = (j2/r2)*lda2+(j3/r3)*lda3;
                        auto ldb = j2*ldb2+j3*ldb3;
                        auto ldc = j2*ldc2+j3*ldc3;
                        A_bf16_v[j2+j3*C.ne[2]] = A_bf16+lda;
                        B_bf16_v[j2+j3*C.ne[2]] = B_bf16+ldb;
                        C_fp32_v[j2+j3*C.ne[2]] = C_fp32+ldc;
                    }
                }

                cblas_sbgemm_batch(CblasRowMajor, &transB, &transA,
                                   &M, &N, &K,
                                   &alpha, B_bf16_v.data(), &ldB,
                                           A_bf16_v.data(), &ldA,
                                   &beta , C_fp32_v.data(), &ldC,
                                   1, &nb_group);
            }
        }
#endif

        void sgemm(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
            GGML_TENSOR_BINARY_OP_LOCALS

            const enum ggml_type type = src0->type;

            GGML_ASSERT(ne0 == ne01);
            GGML_ASSERT(ne1 == ne11);
            GGML_ASSERT(ne2 == ne12);
            GGML_ASSERT(ne3 == ne13);

            // we don't support permuted src0 or src1
            GGML_ASSERT(nb00 == ggml_type_size(type));
            GGML_ASSERT(nb10 == ggml_type_size(src1->type));

            // dst cannot be transposed or permuted
            GGML_ASSERT(nb0 == sizeof(float));
            GGML_ASSERT(nb0 <= nb1);
            GGML_ASSERT(nb1 <= nb2);
            GGML_ASSERT(nb2 <= nb3);

            // broadcast factors
            const int64_t r2 = ne12/ne02;
            const int64_t r3 = ne13/ne03;

            const int64_t ne_plane      = ne01*ne00;
            const size_t  desired_wsize = type == GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*sizeof(float);

            if (work_size < desired_wsize) {
                work_data.reset(new char[desired_wsize]);
                work_size = desired_wsize;
            }
            void * wdata = work_data.get();

            // convert src0 to float
            if (type != GGML_TYPE_F32) {
                const auto * type_traits = ggml_get_type_traits(type);
                ggml_to_float_t const to_float = type_traits->to_float;

#ifdef GGML_USE_OPENMP
                const char  *       x      = (char *)  src0->data;
                      float * const wplane = (float *) wdata;
                const int64_t nf_plane01      = ne00;
                const int64_t nf_plane02      = ne01*ne00;
                const int64_t nf_plane03      = ne02*ne01*ne00;

                #pragma omp parallel for collapse(3) num_threads(this->n_threads) schedule(static)
                for (int64_t i03 = 0; i03 < ne03; i03++) {
                    for (int64_t i02 = 0; i02 < ne02; i02++) {
                        for (int64_t i01 = 0; i01 < ne01; i01++) {
                            to_float(x      + i03*nb03       + i02*nb02       + i01*nb01, 
                                     wplane + i03*nf_plane03 + i02*nf_plane02 + i01*nf_plane01,
                                     ne00);
                        }
                    }
                }
#else
                for (int64_t i03 = 0; i03 < ne03; i03++) {
                    for (int64_t i02 = 0; i02 < ne02; i02++) {
                        const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                              float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                        const int min_cols_per_thread = 4096;
                        const int min_rows_per_thread = std::max((int)(min_cols_per_thread/ne00), 1);
                        const int n_threads = std::max(std::min(this->n_threads, (int)(ne01/min_rows_per_thread)), 1);

                        for (int i = 1; i < n_threads; i++) {
                            const int64_t start =       i*ne01/n_threads;
                            const int64_t end   = (i + 1)*ne01/n_threads;
                            if (start < end) {
                                tasks.push_back(std::async(std::launch::async, [=]() {
                                    for (int64_t i01 = start; i01 < end; i01++) {
                                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                                    }
                                }));
                            }
                        }
                        {
                            // reuse the current thread for the first task
                            const int64_t start = 0;
                            const int64_t end   = ne01/n_threads;
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }
                    }
                }
                // wait for all tasks to finish
                for (auto & task : tasks) {
                    task.get();
                }
                tasks.clear();
#endif
            }

            for (int64_t i13 = 0; i13 < ne13; i13++) {
                for (int64_t i12 = 0; i12 < ne12; i12++) {
                    const int64_t i03 = i13/r3;
                    const int64_t i02 = i12/r2;

                    const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
                    const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                          float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

                    if (type != GGML_TYPE_F32) {
                        x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
                    }
                    if (ne1 == 1) {
                        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                    ne01, ne00,
                                    1.0f,   x, ne00,
                                            y,    1,
                                    0.0f,   d,    1);
                    } else {
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                    ne1, ne01, ne10,
                                    1.0f,   y, ne10,
                                            x, ne00,
                                    0.0f,   d, ne01);
                    }
                }
            }
        }

        void out_prod(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
            GGML_TENSOR_BINARY_OP_LOCALS

            GGML_ASSERT(ne0  == ne00);
            GGML_ASSERT(ne1  == ne10);
            GGML_ASSERT(ne2  == ne02);
            GGML_ASSERT(ne02 == ne12);
            GGML_ASSERT(ne3  == ne13);
            GGML_ASSERT(ne03 == ne13);

            // we don't support permuted src0 or src1
            GGML_ASSERT(nb00 == sizeof(float));

            // dst cannot be transposed or permuted
            GGML_ASSERT(nb0 == sizeof(float));
            // GGML_ASSERT(nb0 <= nb1);
            // GGML_ASSERT(nb1 <= nb2);
            // GGML_ASSERT(nb2 <= nb3);

            // Arguments to ggml_compute_forward_out_prod (expressed as major,minor)
            // src0: (k,n)
            // src1: (k,m)
            // dst:  (m,n)
            //
            // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
            // Also expressed as (major,minor)
            // a: (m,k): so src1 transposed
            // b: (k,n): so src0
            // c: (m,n)
            //
            // However, if ggml_is_transposed(src1) is true, then
            // src1->data already contains a transposed version, so sgemm mustn't
            // transpose it further.

            int n = src0->ne[0];
            int k = src0->ne[1];
            int m = src1->ne[0];

            CBLAS_TRANSPOSE transposeA;
            int lda;

            if (!ggml_is_transposed(src1)) {
                transposeA = CblasTrans;
                lda = m;
            } else {
                transposeA = CblasNoTrans;
                lda = k;
            }

            float * a = (float *) ((char *) src1->data);
            float * b = (float *) ((char *) src0->data);
            float * c = (float *) ((char *) dst->data);

            cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);
        }

    private:
        const std::string m_name{"BLAS"};

    public:
        static constexpr ggml_guid s_guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
            
        backend(const std::string& /*params*/, ggml::cpp::backend::device& dev) :
            ggml::cpp::backend::backend(dev)
        { }

        virtual ~backend() {
            if (m_work_data) std::free(m_work_data);
        }

        const std::string& get_name() override {
            return m_name;
        }

        const ggml_guid* get_guid() override {
            return &s_guid;
        }

        ggml_status graph_compute(ggml_cgraph & cgraph) override {
            for (int i = 0; i < cgraph.n_nodes; i++) {
                ggml_tensor * node = cgraph.nodes[i];
                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }
                switch (node->op) {
                    case GGML_OP_MUL_MAT:
#ifdef GGML_BLAS_USE_SBGEMM
                        if (node->src[0]->type == GGML_TYPE_BF16) {
                            sbgemm(*node->src[0], *node->src[1], *node);
                        } else
#endif
#ifdef GGML_BLAS_USE_SHGEMM
                        //if (node->src[0].type == GGML_TYPE_BF16) {
                        //    shgemm(*node->src[0], *node->src[1], *node);
                        //} else
#endif
                        {
                            sgemm(node->src[0], node->src[1], node);
                        }
                        break;

                    case GGML_OP_OUT_PROD:
                        out_prod(node->src[0], node->src[1], node);
                        break;

                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_PERMUTE:
                    case GGML_OP_TRANSPOSE:
                        break;

                    default:
                        GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                }
            }
            return GGML_STATUS_SUCCESS;
        }

        void set_n_threads(int n_threads) override {
            this->n_threads = n_threads;
#if defined(GGML_BLAS_USE_OPENBLAS)
            openblas_set_num_threads(n_threads);
#    if defined(GGML_USE_OPENMP)
            omp_set_num_threads(n_threads);
#    endif
#elif defined(GGML_BLAS_USE_BLIS)
            bli_thread_set_num_threads(n_threads);
#elif defined(GGML_BLAS_USE_FLEXIBLAS)
            flexiblas_set_num_threads(n_threads);
#elif defined(GGML_BLAS_USE_NVPL)
            nvpl_blas_set_num_threads(n_threads);
#endif
        }

    };

    // device class
    class device : public ggml::cpp::backend::device {
        const std::string m_name;
        const std::string m_desc;
        ggml::cpp::backend::buffer_type* m_cpu_buffer_type;           // ggml_backend_cpu_buffer_type
        ggml::cpp::backend::buffer_type* m_cpu_buffer_from_ptr_type;  // ggml_backend_cpu_buffer_from_ptr_type

    public:
        device() : m_name("BLAS"), m_desc(
            #if defined(GGML_BLAS_USE_ACCELERATE)
                "Accelerate"
            #elif defined(GGML_BLAS_USE_MKL)
                "MKL"
            #elif defined(GGML_BLAS_USE_BLIS)
                "BLIS"
            #elif defined(GGML_BLAS_USE_NVPL)
                "NVPL"
            #elif defined(GGML_BLAS_USE_OPENBLAS)
                "OpenBLAS"
            #else
                "BLAS"
            #endif
        ) {
            m_cpu_buffer_type          = ggml::cpp::backend::new_cpu_buffer_type("BLAS",        false, MEMORY_ALIGNMENT);
            m_cpu_buffer_from_ptr_type = ggml::cpp::backend::new_cpu_buffer_type("BLAS_Mapped", true , MEMORY_ALIGNMENT);
        }

        virtual ~device() {
            delete m_cpu_buffer_type;
            delete m_cpu_buffer_from_ptr_type;
        }

        const std::string& get_name() override {
            return m_name;
        }

        const std::string& get_description() override {
            return m_desc;
        }

        void get_memory(std::size_t & free, std::size_t & total) override {
            // no memory to report
            total = 0;
            free = 0;
        }

        enum ggml_backend_dev_type get_type() override {
            return GGML_BACKEND_DEVICE_TYPE_ACCEL;
        }

        ggml::cpp::backend::backend& init_backend(const std::string& params) override {
            auto back = new backend(params, *this);
            return *back;
        }

        ggml::cpp::backend::buffer_type& get_buffer_type() override {
            return *m_cpu_buffer_type;
        }

        bool caps_buffer_from_host_ptr() override { return true; }
        ggml::cpp::backend::buffer_type* get_from_host_ptr_buffer_type() override { 
            return m_cpu_buffer_from_ptr_type;
        }

        bool supports_op(const ggml_tensor & op) override {
            const struct ggml_tensor * src0 = op.src[0];
            const struct ggml_tensor * src1 = op.src[1];

            switch (op.op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    return true;

                case GGML_OP_MUL_MAT:
                {
                    // BLAS usually is only faster for large matrices
                    const int64_t ne10 = src1->ne[0];

                    const int64_t ne0 = op.ne[0];
                    const int64_t ne1 = op.ne[1];

                    // TODO: find the optimal value
                    const int64_t min_batch = 448;

                    return ggml_is_contiguous(src0) &&
                           ggml_is_contiguous(src1) &&
                           src1->type == GGML_TYPE_F32 &&
                           (    src0->type == GGML_TYPE_F32 ||
#ifdef GGML_BLAS_USE_SBGEMM
                                (   src0->type == GGML_TYPE_BF16 &&
                                    ne1 >= min_batch 
                                ) ||
#endif
                                (   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                                    ggml_get_type_traits(src0->type)->to_float != NULL
                                )
                            );
                }

                case GGML_OP_OUT_PROD:
                    return src0->type == GGML_TYPE_F32 &&
                           src1->type == GGML_TYPE_F32 &&
                           ggml_is_matrix(src0) &&
                           ggml_is_matrix(src1) &&
                           ggml_is_contiguous(src0) &&
                           (ggml_is_contiguous(src1) || ggml_is_transposed(src1)) &&
                           (src0->type == GGML_TYPE_F32 || ggml_get_type_traits(src0->type)->to_float != NULL);

                default:
                    return false;

            }
        }

        bool supports_buft(ggml_backend_buffer_type_t buffer_type) override {
            return ggml_backend_buft_is_host(buffer_type);
        }

    };

    // backend reg class
    class reg: public ggml::cpp::backend::reg {
        const std::string m_name{"BLAS"};
        device* m_device;

    public:
        reg() {
            m_device = new device();
#if defined(GGML_BLAS_USE_OPENBLAS) 
            if (openblas_get_parallel() == OPENBLAS_SEQUENTIAL) {
                GGML_LOG_WARN("%s: warning: OpenBLAS was compiled without parallel support\n", __func__);
            }
#    if defined(GGML_USE_OPENMP)
            else if (openblas_get_parallel() != OPENBLAS_OPENMP) {
                GGML_LOG_WARN("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
            }
#    else
            if (openblas_get_parallel() == OPENBLAS_OPENMP) {
                GGML_LOG_WARN("%s: warning: ggml is not using OpenMP, but OpenBLAS was compiled with OpenMP support\n", __func__);
            }
#    endif
#endif
#if defined(BLIS_ENABLE_CBLAS) && defined(GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
            GGML_LOG_WARN("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif
        }

        virtual ~reg() { }

        const std::string& get_name() override {
            return m_name;
        }

        std::size_t get_device_count() override {
            return 1;
        }

        device& get_device(std::size_t index) override {
            GGML_ASSERT(index == 0);
            return *m_device;
        }
    };

    static ggml::backend::blas::reg& get_reg(void) {
        static ggml::backend::blas::reg ctx;
        return ctx;
    }

}

// extern API:
ggml_backend_t ggml_backend_blas_init(void) {
    auto* _reg = ggml_backend_blas_reg();
    auto* _device = _reg->iface.get_device(_reg, 0);
    return _device->iface.init_backend(_device, "");
}

bool ggml_backend_is_blas(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, &ggml::backend::blas::backend::s_guid);
}

void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads) {
    GGML_ASSERT(ggml_backend_is_blas(backend_blas));
    auto& ctx = *((ggml::cpp::backend::backend*) (backend_blas->context));
    ctx.set_n_threads(n_threads);
}

ggml_backend_reg_t ggml_backend_blas_reg(void) {
    return ggml::cpp::backend::c_wrapper(&ggml::backend::blas::get_reg());
}

GGML_BACKEND_DL_IMPL(ggml_backend_blas_reg)
