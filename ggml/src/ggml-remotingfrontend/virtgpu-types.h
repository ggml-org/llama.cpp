#pragma once
#include "virtgpu.h"

struct vn_cs_encoder {
   char* cur;
   const char* end;
};

struct vn_cs_decoder {
  const char* cur;
  const char* end;
};

/*
 * encode peek
 */

static inline bool
vn_cs_decoder_peek_internal(const struct vn_cs_decoder *dec,
                            size_t size,
                            void *val,
                            size_t val_size)
{
   assert(val_size <= size);

   if (unlikely(size > (size_t) (dec->end - dec->cur))) {
      FATAL("DECODER IS FULL :/");
      //vn_cs_decoder_set_fatal(dec);
      memset(val, 0, val_size);
      return false;
   }

   /* we should not rely on the compiler to optimize away memcpy... */
   memcpy(val, dec->cur, val_size);
   return true;
}

static inline void
vn_cs_decoder_peek(const struct vn_cs_decoder *dec,
                   size_t size,
                   void *val,
                   size_t val_size)
{
   vn_cs_decoder_peek_internal(dec, size, val, val_size);
}

/*
 * read/write
 */

static inline void
vn_cs_decoder_read(struct vn_cs_decoder *dec,
                   size_t size,
                   void *val,
                   size_t val_size)
{
   if (vn_cs_decoder_peek_internal(dec, size, val, val_size))
      dec->cur += size;
}

static inline void
vn_cs_encoder_write(struct vn_cs_encoder *enc,
                    size_t size,
                    const void *val,
                    size_t val_size)
{
   assert(val_size <= size);
   assert(size <= ((size_t) (enc->end - enc->cur)));

   /* we should not rely on the compiler to optimize away memcpy... */
   memcpy(enc->cur, val, val_size);
   enc->cur += size;
}

/*
 * encode/decode
 */

static inline void
vn_decode(struct vn_cs_decoder *dec, size_t size, void *data, size_t data_size)
{
   assert(size % 4 == 0);
   vn_cs_decoder_read(dec, size, data, data_size);
}

static inline void
vn_encode(struct vn_cs_encoder *enc, size_t size, const void *data, size_t data_size)
{
   assert(size % 4 == 0);
   /* TODO check if the generated code is optimal */
   vn_cs_encoder_write(enc, size, data, data_size);
}

/*
 * typed encode/decode
 */

/* uint64_t */

static inline size_t
vn_sizeof_uint64_t(const uint64_t *val)
{
    assert(sizeof(*val) == 8);
    return 8;
}

static inline void
vn_encode_uint64_t(struct vn_cs_encoder *enc, const uint64_t *val)
{
    vn_encode(enc, 8, val, sizeof(*val));
}

static inline void
vn_decode_uint64_t(struct vn_cs_decoder *dec, uint64_t *val)
{
    vn_decode(dec, 8, val, sizeof(*val));
}

static inline size_t
vn_sizeof_uint64_t_array(const uint64_t *val, uint32_t count)
{
    assert(sizeof(*val) == 8);
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    return size;
}

static inline void
vn_encode_uint64_t_array(struct vn_cs_encoder *enc, const uint64_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_encode(enc, size, val, size);
}

static inline void
vn_decode_uint64_t_array(struct vn_cs_decoder *dec, uint64_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_decode(dec, size, val, size);
}

/* int32_t */

static inline size_t
vn_sizeof_int32_t(const int32_t *val)
{
    assert(sizeof(*val) == 4);
    return 4;
}

static inline void
vn_encode_int32_t(struct vn_cs_encoder *enc, const int32_t *val)
{
    vn_encode(enc, 4, val, sizeof(*val));
}

static inline void
vn_decode_int32_t(struct vn_cs_decoder *dec, int32_t *val)
{
    vn_decode(dec, 4, val, sizeof(*val));
}

static inline size_t
vn_sizeof_int32_t_array(const int32_t *val, uint32_t count)
{
    assert(sizeof(*val) == 4);
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    return size;
}

static inline void
vn_encode_int32_t_array(struct vn_cs_encoder *enc, const int32_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_encode(enc, size, val, size);
}

static inline void
vn_decode_int32_t_array(struct vn_cs_decoder *dec, int32_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_decode(dec, size, val, size);
}

/* array size (uint64_t) */

static inline size_t
vn_sizeof_array_size(uint64_t size)
{
    return vn_sizeof_uint64_t(&size);
}

static inline void
vn_encode_array_size(struct vn_cs_encoder *enc, uint64_t size)
{
    vn_encode_uint64_t(enc, &size);
}

static inline uint64_t
vn_decode_array_size(struct vn_cs_decoder *dec, uint64_t expected_size)
{
    uint64_t size;
    vn_decode_uint64_t(dec, &size);
    if (size != expected_size) {
        FATAL("ENCODER IS FULL :/");
        //vn_cs_decoder_set_fatal(dec);
        size = 0;
    }
    return size;
}

static inline uint64_t
vn_decode_array_size_unchecked(struct vn_cs_decoder *dec)
{
    uint64_t size;
    vn_decode_uint64_t(dec, &size);
    return size;
}

static inline uint64_t
vn_peek_array_size(struct vn_cs_decoder *dec)
{
    uint64_t size;
    vn_cs_decoder_peek(dec, sizeof(size), &size, sizeof(size));
    return size;
}

/* non-array pointer */

static inline size_t
vn_sizeof_simple_pointer(const void *val)
{
    return vn_sizeof_array_size(val ? 1 : 0);
}

static inline bool
vn_encode_simple_pointer(struct vn_cs_encoder *enc, const void *val)
{
    vn_encode_array_size(enc, val ? 1 : 0);
    return val;
}

static inline bool
vn_decode_simple_pointer(struct vn_cs_decoder *dec)
{
    return vn_decode_array_size_unchecked(dec);
}

/* uint32_t */

static inline size_t
vn_sizeof_uint32_t(const uint32_t *val)
{
    assert(sizeof(*val) == 4);
    return 4;
}

static inline void
vn_encode_uint32_t(struct vn_cs_encoder *enc, const uint32_t *val)
{
    vn_encode(enc, 4, val, sizeof(*val));
}

static inline void
vn_decode_uint32_t(struct vn_cs_decoder *dec, uint32_t *val)
{
    vn_decode(dec, 4, val, sizeof(*val));
}

static inline size_t
vn_sizeof_uint32_t_array(const uint32_t *val, uint32_t count)
{
    assert(sizeof(*val) == 4);
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    return size;
}

static inline void
vn_encode_uint32_t_array(struct vn_cs_encoder *enc, const uint32_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_encode(enc, size, val, size);
}

static inline void
vn_decode_uint32_t_array(struct vn_cs_decoder *dec, uint32_t *val, uint32_t count)
{
    const size_t size = sizeof(*val) * count;
    assert(size >= count);
    vn_decode(dec, size, val, size);
}
