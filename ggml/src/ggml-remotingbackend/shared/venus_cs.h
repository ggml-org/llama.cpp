#pragma once

#include <cassert>
#include <cstring>

// needs UNUSED to be defined
// needs FATAL to be defined

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct vn_cs_encoder {
  char* cur;
  const char *start;
  const char* end;
};

struct vn_cs_decoder {
  const char* cur;
  const char* end;
};

/*
 * new encoder and decoder
 */

static struct vn_cs_decoder
vn_cs_new_decoder(const char *ptr, size_t size) {
  struct vn_cs_decoder dec = {
      .cur = ptr,
      .end = ptr + size,
  };

  return dec;
}

static struct vn_cs_encoder
vn_cs_new_encoder(char *ptr, size_t size) {
  struct vn_cs_encoder enc = {
      .cur = ptr,
      .start = ptr,
      .end = ptr + size,
  };

  return enc;
}

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
    FATAL("READING TOO MUCH FROM THE DECODER :/");
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

static inline const void *
vn_cs_decoder_use_inplace(struct vn_cs_decoder *dec,
			  size_t size)
{
  if (unlikely(size > (size_t) (dec->end - dec->cur))) {
    FATAL("READING TOO MUCH FROM THE DECODER :/");
  }
  const void *addr = dec->cur;
  dec->cur += size;

  return addr;
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

static inline char *
vn_cs_encoder_write(struct vn_cs_encoder *enc,
                    size_t size,
                    const void *val,
                    size_t val_size)
{
  assert(val_size <= size);
  assert(size <= ((size_t) (enc->end - enc->cur)));

  char *write_addr = enc->cur;
  /* we should not rely on the compiler to optimize away memcpy... */
  memcpy(write_addr, val, val_size);
  enc->cur += size;

  return write_addr;
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

/* uint8_t */

static inline void
vn_encode_uint8_t(struct vn_cs_encoder *enc, const uint8_t *val)
{
  vn_encode(enc, sizeof(int), val, sizeof(*val));
}

static inline void
vn_decode_uint8_t(struct vn_cs_decoder *dec, uint8_t *val)
{
  vn_decode(dec, sizeof(int), val, sizeof(*val));
}

/* uint64_t */

static inline size_t
vn_sizeof_uint64_t(const uint64_t *val)
{
  assert(sizeof(*val) == 8);
#ifdef NDEBUG
  UNUSED(val);
#endif
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

static inline const uint64_t *
vn_decode_uint64_t_array_inplace(struct vn_cs_decoder *dec, uint32_t count)
{
  return (uint64_t *)(uintptr_t) vn_cs_decoder_use_inplace(dec, count * sizeof(uint64_t));
}

/* int32_t */

static inline size_t
vn_sizeof_int32_t(const int32_t *val)
{
  assert(sizeof(*val) == 4);
#ifdef NDEBUG
  UNUSED(val);
#endif
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
#ifdef NDEBUG
  UNUSED(val);
#endif
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

/* size_t */

static inline size_t
vn_sizeof_size_t(const size_t *val)
{
    return sizeof(*val);
}

static inline void
vn_encode_size_t(struct vn_cs_encoder *enc, const size_t *val)
{
    const uint64_t tmp = *val;
    vn_encode_uint64_t(enc, &tmp);
}

static inline void
vn_decode_size_t(struct vn_cs_decoder *dec, size_t *val)
{
    uint64_t tmp;
    vn_decode_uint64_t(dec, &tmp);
    *val = tmp;
}

static inline size_t
vn_sizeof_size_t_array(const size_t *val, uint32_t count)
{
    return vn_sizeof_size_t(val) * count;
}

static inline void
vn_encode_size_t_array(struct vn_cs_encoder *enc, const size_t *val, uint32_t count)
{
    if (sizeof(size_t) == sizeof(uint64_t)) {
        vn_encode_uint64_t_array(enc, (const uint64_t *)val, count);
    } else {
        for (uint32_t i = 0; i < count; i++)
            vn_encode_size_t(enc, &val[i]);
    }
}

static inline void
vn_decode_size_t_array(struct vn_cs_decoder *dec, size_t *val, uint32_t count)
{
    if (sizeof(size_t) == sizeof(uint64_t)) {
        vn_decode_uint64_t_array(dec, (uint64_t *)val, count);
    } else {
        for (uint32_t i = 0; i < count; i++)
            vn_decode_size_t(dec, &val[i]);
    }
}

/* opaque blob */

static inline size_t
vn_sizeof_blob_array(const void *val, size_t size)
{
  UNUSED(val);
  return (size + 3) & ~3;
}

static inline void
vn_encode_blob_array(struct vn_cs_encoder *enc, const void *val, size_t size)
{
  vn_encode(enc, (size + 3) & ~3, val, size);
}

static inline void
vn_decode_blob_array(struct vn_cs_decoder *dec, void *val, size_t size)
{
  vn_decode(dec, (size + 3) & ~3, val, size);
}

/* string */

static inline size_t
vn_sizeof_char_array(const char *val, size_t size)
{
  return vn_sizeof_blob_array(val, size);
}

static inline void
vn_encode_char_array(struct vn_cs_encoder *enc, const char *val, size_t size)
{
  assert(size && strlen(val) < size);
  vn_encode_blob_array(enc, val, size);
}

static inline void
vn_decode_char_array(struct vn_cs_decoder *dec, char *val, size_t size)
{
  vn_decode_blob_array(dec, val, size);
  if (size)
    val[size - 1] = '\0';
  else {
    //vn_cs_decoder_set_fatal(dec);
    FATAL("Couldn't decode the blog array");
  }
}

/* (temp) buffer allocation */

static inline void *
vkr_cs_decoder_alloc_array(struct vkr_cs_decoder *dec, size_t size, size_t count)
{
  UNUSED(dec);
  size_t alloc_size;
  if (unlikely(__builtin_mul_overflow(size, count, &alloc_size))) {
    FATAL("overflow in array allocation of %zu * %zu bytes", size, count);
    return NULL;
  }

  return malloc(alloc_size);
}

static inline void *
vn_cs_decoder_alloc_array(struct vn_cs_decoder *dec, size_t size, size_t count)
{
  struct vkr_cs_decoder *d = (struct vkr_cs_decoder *)dec;
  return vkr_cs_decoder_alloc_array(d, size, count);
}

/* bool */

static inline void
vn_encode_bool_t(struct vn_cs_encoder *enc, const bool *val)
{
  vn_encode(enc, sizeof(int), val, sizeof(bool));
}

static inline void
vn_decode_bool_t(struct vn_cs_decoder *dec, bool *val)
{
  vn_decode(dec, sizeof(int), val, sizeof(bool));
}

/* apir_buffer_type_host_handle_t */

static inline void
vn_encode_apir_buffer_type_host_handle_t(struct vn_cs_encoder *enc, const apir_buffer_type_host_handle_t *val)
{
  vn_encode(enc, sizeof(apir_buffer_type_host_handle_t), val, sizeof(apir_buffer_type_host_handle_t));
}

static inline void
vn_decode_apir_buffer_type_host_handle_t(struct vn_cs_decoder *dec, apir_buffer_type_host_handle_t *val)
{
  vn_decode(dec, sizeof(apir_buffer_type_host_handle_t), val, sizeof(apir_buffer_type_host_handle_t));
}

/* apir_buffer_host_handle_t */

static inline void
vn_encode_apir_buffer_host_handle_t(struct vn_cs_encoder *enc, const apir_buffer_host_handle_t *val)
{
  vn_encode(enc, sizeof(apir_buffer_host_handle_t), val, sizeof(apir_buffer_host_handle_t));
}

static inline void
vn_decode_apir_buffer_host_handle_t(struct vn_cs_decoder *dec, apir_buffer_host_handle_t *val)
{
  vn_decode(dec, sizeof(apir_buffer_host_handle_t), val, sizeof(apir_buffer_host_handle_t));
}

/* uintptr_t */

static inline void
vn_encode_uintptr_t(struct vn_cs_encoder *enc, const uintptr_t *val)
{
  vn_encode(enc, sizeof(*val), val, sizeof(*val));
}

static inline void
vn_decode_uintptr_t(struct vn_cs_decoder *dec, uintptr_t *val)
{
  vn_decode(dec, sizeof(*val), val, sizeof(*val));
}
