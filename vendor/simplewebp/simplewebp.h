/**
 * @file simplewebp.h
 * @author Google Inc., Miku AuahDark
 * @brief A simple WebP decoder.
 * @version 20251007
 * See license at the bottom of the file.
 */

#ifndef _SIMPLE_WEBP_H_
#define _SIMPLE_WEBP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdlib.h>
#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901)
#include <stdint.h>

typedef uint8_t simplewebp_u8;
typedef int8_t simplewebp_i8;
typedef uint16_t simplewebp_u16;
typedef int16_t simplewebp_i16;
typedef uint32_t simplewebp_u32;
typedef int32_t simplewebp_i32;

#ifdef __cplusplus
typedef bool simplewebp_bool;
#else
#include <stdbool.h>
typedef _Bool simplewebp_bool;
#endif

#else /* Yeah this must be good enough */
typedef char simplewebp_bool;
typedef unsigned char simplewebp_u8;
typedef signed char simplewebp_i8;
typedef unsigned short simplewebp_u16;
typedef short simplewebp_i16;
typedef unsigned int simplewebp_u32;
typedef int simplewebp_i32;
#endif

#define SIMPLEWEBP_VERSION 20251007

/**
 * @brief SimpleWebP "input stream".
 * 
 * For user-defined "input stream", populate the struct accordingly.
 * This struct can be allocated on stack, and is assumed so.
 */
typedef struct simplewebp_input
{
	/**
	 * @brief Function-specific userdata.
	 */
	void *userdata;

	/**
	 * @brief Function to close **and** deallocate `userdata`.
	 * @param userdata Function-specific userdata.
	 */
	void (*close)(void *userdata);

	/**
	 * @brief Function to read from "input stream"
	 * @param size Amount of bytes to read.
	 * @param dest Output buffer.
	 * @param userdata Function-specific userdata.
	 * @return Amount of bytes read.
	 */
	size_t (*read)(size_t size, void *dest, void *userdata);

	/**
	 * @brief Function to seek "input stream". Stream must support seeking!
	 * @param pos New position of the stream, based on the beginning of the file.
	 * @param userdata Function-specific userdata.
	 * @return Non-zero on success, zero on failure.
	 */
	simplewebp_bool (*seek)(size_t pos, void *userdata);

	/**
	 * @brief Function to get current "input stream" position. Stream must support this!
	 * @return Stream position relative to the beginning.
	 */
	size_t (*tell)(void *userdata);
} simplewebp_input;

/**
 * @brief SimpleWebP allocator structure.
 * 
 * SimpleWebP functions that needs an allocation need to pass this structure.
 * This struct can be allocated on stack.
 */
typedef struct simplewebp_allocator
{
	/**
	 * @brief Allocate block of memory.
	 * @param userdata Allocator-specific data.
	 * @param size Amount of bytes to allocate.
	 * @return Pointer to tbe memory block, or `NULL` on failure.
	 */
	void *(*alloc)(void *userdata, size_t size);

	/**
	 * @brief Free allocated memory.
	 * @param userdata Allocator-specific data.
	 * @param mem Valid pointer to the memory allocated by `alloc` function.
	 * @note Passing `NULL` is undefined behavior, although most implementation treat it as no-op.
	 */
	void (*free)(void *userdata, void *mem);

	/**
	 * @brief Allocator-specific data.
	 * 
	 * If your allocator is stateless, simply pass `NULL`.
	 */
	void *userdata;
} simplewebp_allocator;

/**
 * @brief SimpleWebP opaque handle.
 */
typedef struct simplewebp simplewebp;

typedef enum simplewebp_error
{
	/** No error */
	SIMPLEWEBP_NO_ERROR = 0,
	/** Failed to allocate memory */
	SIMPLEWEBP_ALLOC_ERROR,
	/** Input read error (such as EOF) */
	SIMPLEWEBP_IO_ERROR,
	/** Not a WebP image */
	SIMPLEWEBP_NOT_WEBP_ERROR,
	/** WebP image corrupt */
	SIMPLEWEBP_CORRUPT_ERROR,
	/** WebP image unsupported */
	SIMPLEWEBP_UNSUPPORTED_ERROR,
	/** WebP image is lossless */
	SIMPLEWEBP_IS_LOSSLESS_ERROR
} simplewebp_error;

/**
 * @brief Get runtime SimpleWebP version.
 * In most cases, this should be equivalent to `SIMPLEWEBP_VERSION`
 * 
 * @return SimpleWebP numeric version.
 */
size_t simplewebp_version(void);
/**
 * @brief Get error message associated by SimpleWebP error code.
 * 
 * @param error Error code.
 * @return Error message as null-terminated string.
 */
const char *simplewebp_get_error_text(simplewebp_error error);

/**
 * @brief Initialize `simplewebp_input` structure to load from memory.
 * 
 * @param data Pointer to the WebP-encoded image in memory.
 * @param size Size of `data` in bytes.
 * @param out Destination structure.
 * @param allocator Allocator structure, or `NULL` to use C default.
 * @return Error codes.
 */
simplewebp_error simplewebp_input_from_memory(void *data, size_t size, simplewebp_input *out, const simplewebp_allocator *allocator);
/**
 * @brief Close `simplewebp_input`, freeing the `userdata` and setting its `userdata` to `NULL`.
 * @param input Pointer to the `simplewebp_input` structure.
 * @note Application should not use the same `simplewebp_input` unless re-initialized again.
 */
void simplewebp_close_input(simplewebp_input *input);

/**
 * @brief Load a WebP image.
 * 
 * @param input Pointer to the `simplewebp_input` structure. This function will take the ownership of the `simplewebp_input`.
 * @param allocator Allocator structure, or `NULL` to use C default.
 * @param out Pointer to the `simplewebp` opaque handle.
 * @return Error codes. 
 */
simplewebp_error simplewebp_load(simplewebp_input *input, const simplewebp_allocator *allocator, simplewebp **out);

/**
 * @brief Load a WebP image from memory.
 * 
 * @param data Pointer to the WebP-encoded image in memory.
 * @param size Size of `data` in bytes.
 * @param allocator Allocator structure, or `NULL` to use C default.
 * @param out Pointer to the `simplewebp` opaque handle.
 * @return Error codes. 
 */
simplewebp_error simplewebp_load_from_memory(void *data, size_t size, const simplewebp_allocator *allocator, simplewebp **out);

/**
 * @brief Frees and unload associated memory and closes the "input stream".
 * @param simplewebp `simplewebp` opaque handle.
 */
void simplewebp_unload(simplewebp *simplewebp);

/**
 * @brief Get WebP image dimensions.
 * @param simplewebp `simplewebp` opaque handle.
 * @param width Where to store the image width.
 * @param height Where to store the image height.
 */
void simplewebp_get_dimensions(simplewebp *simplewebp, size_t *width, size_t *height);

/**
 * @brief Check if the WebP image is lossless or lossy.
 * @param simplewebp `simplewebp` opaque handle.
 * @return 1 if lossless, 0 if lossy.
 */
simplewebp_bool simplewebp_is_lossless(simplewebp *simplewebp);

/**
 * @brief Decode WebP image to raw RGBA8 pixels data.
 * @param simplewebp `simplewebp` opaque handle.
 * @param buffer Block of memory with size of `width * height * 4` bytes. This is where the RGBA is stored.
 * @param settings Ignored. Set this to `NULL`.
 * @return Error codes. 
 */
simplewebp_error simplewebp_decode(simplewebp *simplewebp, void *buffer, void *settings);

/**
 * @brief Decode WebP image to raw planar YUVA420 pixels data. Can only be used on lossy WebP image.
 * @param simplewebp `simplewebp` opaque handle.
 * @param y_buffer Block of memory with size of `width * height` bytes.
 * @param u_buffer Block of memory with size of `((width + 1) / 2) * ((height + 1) / 2)` bytes.
 * @param v_buffer Block of memory with size of `((width + 1) / 2) * ((height + 1) / 2)` bytes.
 * @param a_buffer Block of memory with size of `width * height` bytes.
 * @param settings Ignored. Set this to `NULL`.
 * @return simplewebp_error Error codes.
 */
simplewebp_error simplewebp_decode_yuva(simplewebp *simplewebp, void *y_buffer, void *u_buffer, void *v_buffer, void *a_buffer, void *settings);

#ifndef SIMPLEWEBP_DISABLE_STDIO
#include <stdio.h>

/**
 * @brief Initialize `simplewebp_input` structure to load from `FILE*`.
 * 
 * @param file C stdio `FILE*`.
 * @param out Destination structure.
 * @return Error codes.
 */
simplewebp_error simplewebp_input_from_file(FILE *file, simplewebp_input *out);

/**
 * @brief Initialize `simplewebp_input` structure to load from file path using `fopen`.
 * 
 * @param filename Path to file.
 * @param out Destination structure.
 * @return Error codes.
 */
simplewebp_error simplewebp_input_from_filename(const char *filename, simplewebp_input *out);

/**
 * @brief Load a WebP image from `FILE*`.
 * 
 * @param file C stdio `FILE*`.
 * @param allocator Allocator structure, or `NULL` to use C default.
 * @param out Pointer to the `simplewebp` opaque handle.
 * @return Error codes. 
 */
simplewebp_error simplewebp_load_from_file(FILE *file, const simplewebp_allocator *allocator, simplewebp **out);

/**
 * @brief Load a WebP image from file path using `fopen`.
 * 
 * @param filename Path to file.
 * @param allocator Allocator structure, or `NULL` to use C default.
 * @param out Pointer to the `simplewebp` opaque handle.
 * @return Error codes. 
 */
simplewebp_error simplewebp_load_from_filename(const char *filename, const simplewebp_allocator *allocator, simplewebp **out);
#endif /* SIMPLEWEBP_DISABLE_STDIO */


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _SIMPLE_WEBP_H_ */

#ifdef SIMPLEWEBP_IMPLEMENTATION

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <string.h>

struct swebp__picture_header
{
	simplewebp_u16 width, height;
	simplewebp_u8 xscale, yscale, colorspace, clamp_type;
};

struct swebp__finfo
{
	simplewebp_u8 limit, ilevel, inner, hev_thresh;
};

struct swebp__topsmp
{
	simplewebp_u8 y[16], u[8], v[8];
};

struct swebp__mblock
{
	simplewebp_u8 nz, nz_dc;
};

struct swebp__mblockdata
{
	simplewebp_i16 coeffs[384];
	simplewebp_u32 nonzero_y, nonzero_uv;
	simplewebp_u8 imodes[16], is_i4x4, uvmode, dither, skip, segment;
};

typedef simplewebp_u8 swebp__probarray[11];

struct swebp__bandprobas
{
	swebp__probarray probas[3];
};

struct swebp__proba
{
	simplewebp_u8 segments[3];
	struct swebp__bandprobas bands[4][8];
	const struct swebp__bandprobas *bands_ptr[4][17];
};

struct swebp__frame_header
{
	simplewebp_u8 key_frame, profile, show;
	simplewebp_u32 partition_length;
};

struct swebp__filter_header
{
	simplewebp_u8 simple, level, sharpness, use_lf_delta;
	simplewebp_i32 ref_lf_delta[4], mode_lf_delta[4];
};

struct swebp__segment_header
{
	simplewebp_u8 use_segment, update_map, absolute_delta;
	simplewebp_i8 quantizer[4], filter_strength[4];
};

/* Bit reader and boolean decoder */
struct swebp__bdec
{
	const simplewebp_u8 *buf, *buf_end, *buf_max;
	simplewebp_u32 value;
	simplewebp_u8 range, eof;
	simplewebp_i8 bits;
};

typedef simplewebp_i32 swebp__quant_t[2];
struct swebp__quantmat
{
	swebp__quant_t y1_mat, y2_mat, uv_mat;
	simplewebp_i32 uv_quant, dither;
};

struct swebp__alpha
{
	simplewebp_u8 filter_method;
	simplewebp_bool is_lossless_compressed;
};

struct swebp__vp8
{
	simplewebp_u8 ready;

	struct swebp__bdec br;
	struct swebp__frame_header frame_header;
	struct swebp__picture_header picture_header;
	struct swebp__filter_header filter_header;
	struct swebp__segment_header segment_header;

	simplewebp_i32 mb_w, mb_h;
	simplewebp_i32 tl_mb_x, tl_mb_y;
	simplewebp_i32 br_mb_x, br_mb_y;

	simplewebp_u32 nparts_minus_1;
	struct swebp__bdec parts[8];

	struct swebp__quantmat dqm[4];

	struct swebp__proba proba;
	simplewebp_u8 use_skip_proba, skip_proba;

	simplewebp_u8 *intra_t, intra_l[4];

	struct swebp__topsmp *yuv_t;

	struct swebp__mblock *mb_info;
	struct swebp__finfo *f_info;
	simplewebp_u8 *yuv_b;

	
	simplewebp_u8 *cache_y, *cache_u, *cache_v;
	simplewebp_i32 cache_y_stride, cache_uv_stride;

	
	simplewebp_u8* mem;
	size_t mem_size;

	simplewebp_i32 mb_x, mb_y;
	struct swebp__mblockdata *mb_data;

	simplewebp_i8 filter_type;
	struct swebp__finfo fstrengths[4][2];

	simplewebp_u8 *alpha_plane;
};

struct swebp__vp8l_bdec
{
	simplewebp_u8 *buf;
	size_t len, bit_pos;
	simplewebp_u8 eos;
};

struct swebp__vp8l_decoder
{
	simplewebp_u32 width, height;
	simplewebp_bool has_alpha;
};

union swebp__decoder_list
{
	struct swebp__vp8 vp8;
	struct swebp__vp8l_decoder vp8l;
};

struct swebp__yuvdst
{
	simplewebp_u8 *y, *u, *v, *a;
};

struct swebp__pixel
{
	simplewebp_u8 r, g, b, a;
};

struct swebp__chroma
{
	simplewebp_u8 u, v;
};

struct simplewebp
{
	simplewebp_input input, riff_input, vp8_input, vp8l_input, alph_input;
	simplewebp_allocator allocator;
	struct swebp__alpha alpha_decoder;

	simplewebp_u8 webp_type; /* Simple lossy (0), Lossless (1) */
	union swebp__decoder_list decoder;
};

struct simplewebp_memoryinput_data
{
	simplewebp_allocator allocator;
	void *data;
	size_t size, pos;
};

static void *swebp__malloc(void *_unused, size_t size)
{
	(void)_unused;
	return malloc(size);
}

static void swebp__free(void *_unused, void *mem)
{
	(void)_unused;
	free(mem);
}

static void *swebp__alloc(simplewebp *simplewebp, size_t size)
{
	return simplewebp->allocator.alloc(simplewebp->allocator.userdata, size);
}

static void swebp__dealloc(simplewebp *simplewebp, void *ptr)
{
	if (ptr)
		simplewebp->allocator.free(simplewebp->allocator.userdata, ptr);
}

static simplewebp_allocator swebp__default_allocator = {swebp__malloc, swebp__free, NULL};

size_t simplewebp_version(void)
{
	return SIMPLEWEBP_VERSION;
}

const char *simplewebp_get_error_text(simplewebp_error error)
{
	switch (error)
	{
		case SIMPLEWEBP_NO_ERROR:
			return "No error";
		case SIMPLEWEBP_ALLOC_ERROR:
			return "Failed to allocate memory";
		case SIMPLEWEBP_IO_ERROR:
			return "Input read error (such as EOF)";
		case SIMPLEWEBP_NOT_WEBP_ERROR:
			return "Not a WebP image";
		case SIMPLEWEBP_CORRUPT_ERROR:
			return "WebP image corrupt";
		case SIMPLEWEBP_UNSUPPORTED_ERROR:
			return "WebP image unsupported";
		case SIMPLEWEBP_IS_LOSSLESS_ERROR:
			return "WebP image is lossless";
		default:
			return "Unknown error";
	}
}

static size_t swebp__memoryinput_read(size_t size, void *dest, void *userdata)
{
	struct simplewebp_memoryinput_data *input_data;
	size_t nextpos, readed;

	input_data = (struct simplewebp_memoryinput_data *) userdata;
	nextpos = input_data->pos + size;

	if (nextpos >= input_data->size)
	{
		readed = input_data->size - input_data->pos;
		nextpos = input_data->size;
	}
	else
		readed = size;

	if (readed > 0)
	{
		memcpy(dest, ((char *) input_data->data) + input_data->pos, readed);
		input_data->pos = nextpos;
	}

	return readed;
}

static simplewebp_bool swebp__memoryinput_seek(size_t pos, void *userdata)
{
	struct simplewebp_memoryinput_data *input_data = (struct simplewebp_memoryinput_data *) userdata;

	if (pos >= input_data->size)
		input_data->pos = input_data->size;
	else
		input_data->pos = pos;

	return 1;
}

static void swebp__memoryinput_close(void *userdata)
{
	struct simplewebp_memoryinput_data *input_data = (struct simplewebp_memoryinput_data *) userdata;
	input_data->allocator.free(input_data->allocator.userdata, input_data);
}

static size_t swebp__memoryinput_tell(void *userdata)
{
	struct simplewebp_memoryinput_data *input_data = (struct simplewebp_memoryinput_data *) userdata;
	return input_data->pos;
}

simplewebp_error simplewebp_input_from_memory(void *data, size_t size, simplewebp_input *out, const simplewebp_allocator *allocator)
{
	struct simplewebp_memoryinput_data *input_data;

	if (allocator == NULL)
		allocator = &swebp__default_allocator;

	input_data = (struct simplewebp_memoryinput_data *) allocator->alloc(
		allocator->userdata,
		sizeof(struct simplewebp_memoryinput_data)
	);
	if (input_data == NULL)
		return SIMPLEWEBP_ALLOC_ERROR;

	input_data->allocator = *allocator;
	input_data->data = data;
	input_data->size = size;
	input_data->pos = 0;
	out->userdata = input_data;
	out->read = swebp__memoryinput_read;
	out->seek = swebp__memoryinput_seek;
	out->tell = swebp__memoryinput_tell;
	out->close = swebp__memoryinput_close;
	return SIMPLEWEBP_NO_ERROR;
}

struct simplewebp_input_proxy
{
	simplewebp_input input;
	simplewebp_allocator allocator;

	size_t start, length;
};

static simplewebp_bool swebp__seek(size_t pos, simplewebp_input *input)
{
	return input->seek(pos, input->userdata);
}

static size_t swebp__read(size_t size, void *dest, simplewebp_input *input)
{
	return input->read(size, dest, input->userdata);
}

static simplewebp_bool swebp__read2(size_t size, void *dest, simplewebp_input *input)
{
	return swebp__read(size, dest, input) == size;
}

static size_t swebp__tell(simplewebp_input *input)
{
	return input->tell(input->userdata);
}

static size_t swebp__proxy_tell(void *userdata)
{
	struct simplewebp_input_proxy *proxy;
	size_t pos;

	proxy = (struct simplewebp_input_proxy *) userdata;
	pos = swebp__tell(&proxy->input);

	if (pos < proxy->start)
	{
		swebp__seek(proxy->start, &proxy->input);
		pos = proxy->start;
	}

	return pos - proxy->start;
}

static size_t swebp__proxy_read(size_t size, void *dest, void *userdata)
{
	struct simplewebp_input_proxy *proxy;
	size_t pos, nextpos, readed;

	proxy = (struct simplewebp_input_proxy *) userdata;
	pos = swebp__proxy_tell(userdata);
	
	nextpos = pos + size;

	if (nextpos >= proxy->length)
	{
		readed = size - (proxy->length - nextpos);
		nextpos = proxy->length;
	}
	else
		readed = size;

	if (readed > 0)
		readed = swebp__read(readed, dest, &proxy->input);

	return readed;
}

static simplewebp_bool swebp__proxy_seek(size_t pos, void *userdata)
{
	struct simplewebp_input_proxy *proxy = (struct simplewebp_input_proxy *) userdata;

	if (pos > proxy->length)
		pos = proxy->length;

	return swebp__seek(pos + proxy->start, &proxy->input);
}

static void swebp__proxy_close(void *userdata)
{
	struct simplewebp_input_proxy *proxy = (struct simplewebp_input_proxy *) userdata;
	proxy->allocator.free(proxy->allocator.userdata, proxy);
}

static size_t swebp__proxy_size(void *userdata)
{
	struct simplewebp_input_proxy *proxy = (struct simplewebp_input_proxy *) userdata;
	return proxy->length;
}

static simplewebp_error swebp__proxy_create(
	const simplewebp_allocator *allocator,
	simplewebp_input *input,
	simplewebp_input *out,
	size_t start,
	size_t length
)
{
	struct simplewebp_input_proxy *proxy = (struct simplewebp_input_proxy *) allocator->alloc(
		allocator->userdata,
		sizeof(struct simplewebp_input_proxy)
	);
	if (proxy == NULL)
		return SIMPLEWEBP_ALLOC_ERROR;

	proxy->input = *input;
	proxy->allocator = *allocator;
	proxy->start = start;
	proxy->length = length;
	out->userdata = proxy;
	out->read = swebp__proxy_read;
	out->seek = swebp__proxy_seek;
	out->tell = swebp__proxy_tell;
	out->close = swebp__proxy_close;
	return SIMPLEWEBP_NO_ERROR;
}

void simplewebp_close_input(simplewebp_input *input)
{
	input->close(input->userdata);
	input->userdata = NULL;
}

static simplewebp_u32 swebp__to_uint32(const simplewebp_u8 *buf)
{
	return buf[0] | (((simplewebp_u32) buf[1]) << 8) | (((simplewebp_u32) buf[2]) << 16) | (((simplewebp_u32) buf[3]) << 24);
}

static simplewebp_error swebp__get_input_chunk_4cc(const simplewebp_allocator *allocator, simplewebp_input *input, simplewebp_input *outproxy, void *fourcc, size_t *chunk_size)
{
	simplewebp_u8 size[4];

	if (!swebp__read2(4, fourcc, input))
		return SIMPLEWEBP_IO_ERROR;
	if (!swebp__read2(4, size, input))
		return SIMPLEWEBP_IO_ERROR;

	*chunk_size = swebp__to_uint32(size);
	return swebp__proxy_create(allocator, input, outproxy, swebp__tell(input), *chunk_size);
}

static simplewebp_u16 swebp__to_uint16(const simplewebp_u8 *buf)
{
	return buf[0] | (((simplewebp_u16) buf[1]) << 8);
}

static simplewebp_error swebp__load_lossy(simplewebp_input *vp8_input, simplewebp *result)
{
	simplewebp_u8 temp[8], profile;
	simplewebp_u32 frametag, partition_size;
	simplewebp_u16 width, height;

	if (!swebp__seek(0, vp8_input))
		return SIMPLEWEBP_IO_ERROR;

	if (!swebp__read2(3, temp, vp8_input))
		return SIMPLEWEBP_IO_ERROR;

	frametag = temp[0] | (((simplewebp_u32) temp[1]) << 8) | (((simplewebp_u32) temp[2]) << 16);
	if (frametag & 1)
		/* Intraframe in SimpleWebP? Nope */
		return SIMPLEWEBP_UNSUPPORTED_ERROR;

	profile = (frametag >> 1) & 7;
	if (profile > 3)
		/* Unsupported profile */
		return SIMPLEWEBP_UNSUPPORTED_ERROR;

	partition_size = frametag >> 5;
	if (partition_size >= swebp__proxy_size(vp8_input->userdata))
		/* Inconsistent data */
		return SIMPLEWEBP_CORRUPT_ERROR;

	if (!swebp__read2(7, temp, vp8_input))
		return SIMPLEWEBP_IO_ERROR;

	if (memcmp(temp, "\x9D\x01\x2A", 3) != 0)
		return SIMPLEWEBP_CORRUPT_ERROR;

	width = swebp__to_uint16(temp + 3);
	height = swebp__to_uint16(temp + 5);

	result->webp_type = 0;
	memset(&result->decoder.vp8, 0, sizeof(struct swebp__vp8));
	result->decoder.vp8.picture_header.width = width & 0x3FFF;
	result->decoder.vp8.picture_header.height = height & 0x3FFF;
	result->decoder.vp8.picture_header.xscale = (simplewebp_u8) (width >> 14);
	result->decoder.vp8.picture_header.yscale = (simplewebp_u8) (height >> 14);
	result->decoder.vp8.frame_header.partition_length = partition_size;

	return SIMPLEWEBP_NO_ERROR;
}

static void swebp__vp8l_bitread_init(struct swebp__vp8l_bdec *bdec, simplewebp_u8 *buf, size_t size)
{
	bdec->buf = buf;
	bdec->len = size;
	bdec->bit_pos = 0;
	bdec->eos = 0;
}

static simplewebp_u32 swebp__vp8l_bitread_read(struct swebp__vp8l_bdec *bdec, int count)
{
	int i;
	simplewebp_u32 value = 0;

	for (i = 0; i < count; i++, bdec->bit_pos++)
	{
		simplewebp_u8 b;
		size_t bytepos = bdec->bit_pos >> 3;

		if (bytepos >= bdec->len)
		{
			bdec->eos = 1;
			break;
		}

		b = bdec->buf[bytepos];
		value |= ((b >> (bdec->bit_pos & 7)) & 1) << i;
	}

	return value;
}

static simplewebp_error swebp__load_lossless(simplewebp_input *vp8l_input, simplewebp *result)
{
	simplewebp_u8 temp[5];
	struct swebp__vp8l_bdec br;

	if (!swebp__seek(0, vp8l_input))
		return SIMPLEWEBP_IO_ERROR;

	if (!swebp__read2(5, temp, vp8l_input))
		return SIMPLEWEBP_IO_ERROR;

	if (temp[0] != 0x2F)
		return SIMPLEWEBP_CORRUPT_ERROR;

	swebp__vp8l_bitread_init(&br, temp + 1, 4);

	memset(&result->decoder.vp8l, 0, sizeof(struct swebp__vp8l_decoder));
	result->decoder.vp8l.width = swebp__vp8l_bitread_read(&br, 14) + 1;
	result->decoder.vp8l.height = swebp__vp8l_bitread_read(&br, 14) + 1;
	result->decoder.vp8l.has_alpha = swebp__vp8l_bitread_read(&br, 1);
	if (swebp__vp8l_bitread_read(&br, 3) != 0)
		return SIMPLEWEBP_UNSUPPORTED_ERROR;

	result->webp_type = 1;
	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_u32 swebp__to_uint24(const simplewebp_u8 *buf)
{
	return buf[0] | (((simplewebp_u32) buf[1]) << 8) | (((simplewebp_u32) buf[2]) << 16);
}

static simplewebp_error swebp__alpha_init(struct swebp__alpha *alpha, simplewebp_input *input)
{
	simplewebp_u8 flags;

	if (!swebp__seek(0, input))
		return SIMPLEWEBP_IO_ERROR;

	if (!swebp__read2(1, &flags, input))
		return SIMPLEWEBP_IO_ERROR;

	alpha->filter_method = (flags >> 2) & 3;
	alpha->is_lossless_compressed = (flags & 3) > 0;

	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_i32 swebp__clip(simplewebp_i32 v, simplewebp_i32 m)
{
	return (v < 0) ? 0 : ((v > m) ? m : v);
}

static void swebp__alpha_apply_filters(simplewebp_u8 *ptr, size_t width, size_t height, simplewebp_u8 filter_type)
{
	size_t y, x;

	if (filter_type == 0)
		/* No filters to be applied */
		return;

	for (y = 0; y < height; y++)
	{
		for (x = (y == 0); x < width; x++)
		{
			size_t i = y * width + x;
			simplewebp_u8 predictor = 0;

			switch (filter_type)
			{
				case 0:
				default:
					/* Should not happen */
					predictor = 0;
					/* FALLTHROUGH */
				case 1:
					predictor = x == 0 ? ptr[i - width] : ptr[i - 1];
					break;
				case 2:
					predictor = y == 0 ? ptr[i - 1] : ptr[i - width];
					break;
				case 3:
				{
					simplewebp_i16 a, b, c;
					simplewebp_i16 val;

					a = x == 0 ? ptr[i - width] : ptr[i - 1];
					b = y == 0 ? ptr[i - 1] : ptr[i - width];
					if (x > 0)
					{
						if (y > 0)
							c = ptr[i - width - 1];
						else
							c = ptr[i - 1];
					}
					else
					{
						if (y > 0)
							c = ptr[i - width];
						else
							c = ptr[0];
					}
					val = a + b - c;
					predictor = (simplewebp_u8) swebp__clip(val, 255);
					break;
				}
			}

			ptr[i] += predictor;
		}
	}
}

static simplewebp_error swebp__alpha_decode_simple(simplewebp *simplewebp, simplewebp_u8 *dst)
{
	size_t width, height;
	simplewebp_input *input = &simplewebp->alph_input;

	if (!swebp__seek(1, input))
		return SIMPLEWEBP_IO_ERROR;

	simplewebp_get_dimensions(simplewebp, &width, &height);
	if (!swebp__read2(width * height, dst, input))
		return SIMPLEWEBP_IO_ERROR;

	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_error swebp__decode_lossless_bitstream(
	simplewebp *simplewebp,
	simplewebp_input *input,
	size_t bitstreamsize,
	struct swebp__pixel *rgba,
	simplewebp_bool skipheader
);

static simplewebp_error swebp__alpha_decode_lossless(simplewebp *simplewebp, simplewebp_u8 *dst)
{
	size_t vp8lsize, width, height, i;
	struct swebp__pixel *rgba;
	simplewebp_error err;

	simplewebp_get_dimensions(simplewebp, &width, &height);

	vp8lsize = swebp__proxy_size(simplewebp->alph_input.userdata) - 1;
	if (!swebp__proxy_seek(1, simplewebp->alph_input.userdata))
		return SIMPLEWEBP_IO_ERROR;

	rgba = (struct swebp__pixel*) swebp__alloc(simplewebp, sizeof(struct swebp__pixel) * width * height);
	if (!rgba)
		return SIMPLEWEBP_ALLOC_ERROR;

	err = swebp__decode_lossless_bitstream(
		simplewebp,
		&simplewebp->alph_input,
		vp8lsize,
		rgba,
		1
	);
	if (err != SIMPLEWEBP_NO_ERROR)
	{
		swebp__dealloc(simplewebp, rgba);
		return err;
	}

	/* Pull green channel */
	for (i = 0; i < width * height; i++)
		dst[i] = rgba[i].g;
	
	swebp__dealloc(simplewebp, rgba);
	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_error swebp__alpha_decode(simplewebp *simplewebp, simplewebp_u8 *dst)
{
	size_t width, height;
	simplewebp_error err = SIMPLEWEBP_NO_ERROR;

	simplewebp_get_dimensions(simplewebp, &width, &height);

	if (simplewebp->alph_input.userdata)
	{
		if (simplewebp->alpha_decoder.is_lossless_compressed)
			err = swebp__alpha_decode_lossless(simplewebp, dst);
		else
			err = swebp__alpha_decode_simple(simplewebp, dst);
		
		if (err == SIMPLEWEBP_NO_ERROR)
			swebp__alpha_apply_filters(dst, width, height, simplewebp->alpha_decoder.filter_method);
	}
	else
		memset(dst, 255, width * height);

	return err;
}

static simplewebp_bool swebp__has_decoder(simplewebp *simplewebp)
{
	return simplewebp->vp8_input.userdata != NULL || simplewebp->vp8l_input.userdata != NULL;
}

simplewebp_error simplewebp_load(simplewebp_input *input, const simplewebp_allocator *allocator, simplewebp **out)
{
	simplewebp_u8 temp[4];
	simplewebp_error err;
	simplewebp_input riff_input;
	simplewebp *result;
	size_t chunk_size;

	if (allocator == NULL)
		allocator = &swebp__default_allocator;

	*out = NULL;

	/* Read "RIFF" */
	err = swebp__get_input_chunk_4cc(allocator, input, &riff_input, temp, &chunk_size);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;
	if (memcmp(temp, "RIFF", 4) != 0)
	{
		simplewebp_close_input(&riff_input);
		return SIMPLEWEBP_NOT_WEBP_ERROR;
	}

	/* Read "WEBP" */
	if (!swebp__read2(4, temp, &riff_input) || memcmp(temp, "WEBP", 4) != 0)
	{
		simplewebp_close_input(&riff_input);
		return SIMPLEWEBP_IO_ERROR;
	}

	/* Allocate simplewebp structure */
	result = (simplewebp *) allocator->alloc(allocator->userdata, sizeof(simplewebp));
	if (result == NULL)
		return SIMPLEWEBP_ALLOC_ERROR;
	memset(result, 0, sizeof(simplewebp));

	result->allocator = *allocator;
	result->riff_input = riff_input;

	while (1)
	{
		simplewebp_input chunk_input_proxy;
		size_t current_position;

		err = swebp__get_input_chunk_4cc(allocator, &result->riff_input, &chunk_input_proxy, temp, &chunk_size);
		if (err != SIMPLEWEBP_NO_ERROR)
		{
			if (swebp__has_decoder(result))
				/* Loaded successfully */
				err = SIMPLEWEBP_NO_ERROR;

			break;
		}

		current_position = swebp__tell(&result->riff_input);

		if (memcmp(temp, "VP8 ", 4) == 0)
		{
			if (swebp__has_decoder(result))
			{
				err = SIMPLEWEBP_UNSUPPORTED_ERROR;
				break;
			}

			err = swebp__load_lossy(&chunk_input_proxy, result);
			if (err != SIMPLEWEBP_NO_ERROR)
			{
				simplewebp_close_input(&chunk_input_proxy);
				break;
			}
			
			result->vp8_input = chunk_input_proxy;
		}
		else if (memcmp(temp, "VP8L", 4) == 0)
		{
			if (swebp__has_decoder(result) || result->alph_input.userdata != NULL)
			{
				/* Alpha channel is present or existing VP8(L) decoder present. */
				simplewebp_close_input(&chunk_input_proxy);
				err = SIMPLEWEBP_UNSUPPORTED_ERROR;
				break;
			}

			err = swebp__load_lossless(&chunk_input_proxy, result);
			if (err != SIMPLEWEBP_NO_ERROR)
			{
				simplewebp_close_input(&chunk_input_proxy);
				break;
			}

			result->vp8l_input = chunk_input_proxy;
		}
		else if (memcmp(temp, "ALPH", 4) == 0)
		{
			if (result->vp8l_input.userdata != NULL)
			{
				/* Alpha channel already present in VP8L */
				simplewebp_close_input(&chunk_input_proxy);
				err = SIMPLEWEBP_UNSUPPORTED_ERROR;
				break;
			}

			err = swebp__alpha_init(&result->alpha_decoder, &chunk_input_proxy);
			if (err != SIMPLEWEBP_NO_ERROR)
			{
				simplewebp_close_input(&chunk_input_proxy);
				break;
			}

			result->alph_input = chunk_input_proxy;
		}
		else
			/* Input unused. */
			simplewebp_close_input(&chunk_input_proxy);

		if (!swebp__seek(current_position + ((chunk_size + 1) & (~((size_t) 1))), &result->riff_input))
		{
			err = SIMPLEWEBP_IO_ERROR;
			break;
		}
	}

	/* Success case */
	if (err == SIMPLEWEBP_NO_ERROR)
	{
		result->input = *input;
		*out = result;
	}
	/* Failure case */
	else
	{
		simplewebp_unload(result);
		*out = NULL;
	}

	return err;
}

simplewebp_error simplewebp_load_from_memory(void *data, size_t size, const simplewebp_allocator *allocator, simplewebp **out)
{
	simplewebp_input input;
	simplewebp_error err;

	err = simplewebp_input_from_memory(data, size, &input, allocator);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;

	err = simplewebp_load(&input, allocator, out);
	if (err != SIMPLEWEBP_NO_ERROR)
		simplewebp_close_input(&input);

	return err;
}

static void swebp__close_input(simplewebp_input *inp)
{
	if (inp->userdata)
		simplewebp_close_input(inp);
}

void simplewebp_unload(simplewebp *simplewebp)
{
	swebp__close_input(&simplewebp->alph_input);
	swebp__close_input(&simplewebp->vp8_input);
	swebp__close_input(&simplewebp->vp8l_input);
	swebp__close_input(&simplewebp->riff_input);
	swebp__close_input(&simplewebp->input);

	if (simplewebp->webp_type)
	{
		/* TODO: Clear lossless data*/
	}
	else
	{

	}

	swebp__dealloc(simplewebp, simplewebp);
}

void simplewebp_get_dimensions(simplewebp *simplewebp, size_t *width, size_t *height)
{
	switch (simplewebp->webp_type)
	{
		case 0:
			*width = simplewebp->decoder.vp8.picture_header.width;
			*height = simplewebp->decoder.vp8.picture_header.height;
			break;
		case 1:
			*width = simplewebp->decoder.vp8l.width;
			*height = simplewebp->decoder.vp8l.height;
			break;
		default:
			*width = 0;
			*height = 0;
			break;
	}
}

simplewebp_bool simplewebp_is_lossless(simplewebp *simplewebp)
{
	return simplewebp->webp_type == 1;
}

static void swebp__bitread_setbuf(struct swebp__bdec *br, simplewebp_u8 *buf, size_t size)
{
	br->buf = buf;
	br->buf_end = buf + size;
	br->buf_max = (size >= sizeof(simplewebp_u32))
		? (buf + size - sizeof(simplewebp_u32) + 1)
		: buf;
}

static void swebp__bitread_load(struct swebp__bdec *br)
{
	simplewebp_u32 bits;

	if (br->buf < br->buf_max)
	{
		/* Read 24 bits at a time in big endian order */
		bits = br->buf[2] | (br->buf[1] << 8) | (br->buf[0] << 16);
		br->buf += 3; /* 24 / 8 */
		br->value = bits | (br->value << 24);
		br->bits += 24;
	}
	else
	{
		/* Only read 8 bits at a time */
		if (br->buf < br->buf_end)
		{
			br->bits += 8;
			br->value = (*br->buf++) | (br->value << 8);
		}
		else if (!br->eof)
		{
			br->value <<= 8;
			br->bits += 8;
			br->eof = 1;
		}
		else
			br->bits = 0;
	}
}

static void swebp__bitread_init(struct swebp__bdec *br, simplewebp_u8 *buf, size_t size)
{
	br->range = 254;
	br->value = 0;
	br->bits = -8;
	br->eof = 0;
	swebp__bitread_setbuf(br, buf, size);
	swebp__bitread_load(br);
}

/* https://stackoverflow.com/a/11398748 */
const simplewebp_u32 swebp__blog2_tab32[32] = {
	 0,  9,  1, 10, 13, 21,  2, 29,
	11, 14, 16, 18, 22, 25,  3, 30,
	 8, 12, 20, 28, 15, 17, 24,  7,
	19, 27, 23,  6, 26,  5,  4, 31
};

static simplewebp_u32 swebp__bitslog2floor(simplewebp_u32 value)
{
	value |= value >> 1;
	value |= value >> 2;
	value |= value >> 4;
	value |= value >> 8;
	value |= value >> 16;
	return swebp__blog2_tab32[(value * 0x07C4ACDDU) >> 27];
}

static simplewebp_u32 swebp__bitread_getbit(struct swebp__bdec *br, simplewebp_u32 prob)
{
	simplewebp_u32 bit;
	simplewebp_u8 range, split, value, shift;
	simplewebp_i8 pos;

	range = br->range;

	if (br->bits < 0)
		swebp__bitread_load(br);

	pos = br->bits;
	split = (simplewebp_u8) ((((simplewebp_u32) range) * prob) >> 8);
	value = (simplewebp_u8) (br->value >> pos);
	bit = value > split;

	if (bit)
	{
		range -= split;
		br->value -= (((simplewebp_u32) split) + 1) << pos;
	}
	else
		range = split + 1;

	shift = 7 ^ swebp__bitslog2floor(range);
	range <<= shift;
	br->bits -= shift;
	br->range = range - 1;
	return bit;
}

static simplewebp_u32 swebp__bitread_getval(struct swebp__bdec *br, simplewebp_u8 bits)
{
	simplewebp_u32 value = 0;

	while (bits--)
		value |= swebp__bitread_getbit(br, 0x80) << bits;

	return value;
}

static simplewebp_i32 swebp__bitread_getvalsigned(struct swebp__bdec *br, simplewebp_u8 bits)
{
	simplewebp_i32 value = swebp__bitread_getval(br, bits);
	return swebp__bitread_getval(br, 1) ? -value : value;
}

static simplewebp_i32 swebp__bitread_getsigned(struct swebp__bdec *br, simplewebp_i32 v)
{
	simplewebp_i8 pos;
	simplewebp_u32 split, value;
	simplewebp_i32 mask;

	if (br->bits < 0)
		swebp__bitread_load(br);

	pos = br->bits;
	split = br->range >> 1;
	value = br->value >> pos;
	mask = ((simplewebp_i32) (split - value)) >> 31;
	br->bits -= 1;
	br->range += (simplewebp_u8) mask;
	br->range |= 1;
	br->value -= (simplewebp_u32) ((split + 1) & (simplewebp_u32) mask) << pos;
	return (v ^ mask) - mask;
}

/* RFC 6386 section 14.1 */
static const simplewebp_u8 swebp__dctab[128] = {
  4,     5,   6,   7,   8,   9,  10,  10,
  11,   12,  13,  14,  15,  16,  17,  17,
  18,   19,  20,  20,  21,  21,  22,  22,
  23,   23,  24,  25,  25,  26,  27,  28,
  29,   30,  31,  32,  33,  34,  35,  36,
  37,   37,  38,  39,  40,  41,  42,  43,
  44,   45,  46,  46,  47,  48,  49,  50,
  51,   52,  53,  54,  55,  56,  57,  58,
  59,   60,  61,  62,  63,  64,  65,  66,
  67,   68,  69,  70,  71,  72,  73,  74,
  75,   76,  76,  77,  78,  79,  80,  81,
  82,   83,  84,  85,  86,  87,  88,  89,
  91,   93,  95,  96,  98, 100, 101, 102,
  104, 106, 108, 110, 112, 114, 116, 118,
  122, 124, 126, 128, 130, 132, 134, 136,
  138, 140, 143, 145, 148, 151, 154, 157
};

static const simplewebp_u16 swebp__actab[128] = {
	4,     5,   6,   7,   8,   9,  10,  11,
	12,   13,  14,  15,  16,  17,  18,  19,
	20,   21,  22,  23,  24,  25,  26,  27,
	28,   29,  30,  31,  32,  33,  34,  35,
	36,   37,  38,  39,  40,  41,  42,  43,
	44,   45,  46,  47,  48,  49,  50,  51,
	52,   53,  54,  55,  56,  57,  58,  60,
	62,   64,  66,  68,  70,  72,  74,  76,
	78,   80,  82,  84,  86,  88,  90,  92,
	94,   96,  98, 100, 102, 104, 106, 108,
	110, 112, 114, 116, 119, 122, 125, 128,
	131, 134, 137, 140, 143, 146, 149, 152,
	155, 158, 161, 164, 167, 170, 173, 177,
	181, 185, 189, 193, 197, 201, 205, 209,
	213, 217, 221, 225, 229, 234, 239, 245,
	249, 254, 259, 264, 269, 274, 279, 284
};

/* RFC 6386 section 13 */
static const simplewebp_u8 swebp__coeff_update_proba[4][8][3][11] = {
	{ { { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 176, 246, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 223, 241, 252, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 249, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 244, 252, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 234, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 246, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 239, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 251, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 251, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 253, 255, 254, 255, 255, 255, 255, 255, 255 },
			{ 250, 255, 254, 255, 254, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		}
	},
	{ { { 217, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 225, 252, 241, 253, 255, 255, 254, 255, 255, 255, 255 },
			{ 234, 250, 241, 250, 253, 255, 253, 254, 255, 255, 255 }
		},
		{ { 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 223, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 238, 253, 254, 254, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 248, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 249, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 247, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		}
	},
	{ { { 186, 251, 250, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 234, 251, 244, 254, 255, 255, 255, 255, 255, 255, 255 },
			{ 251, 251, 243, 253, 254, 255, 254, 255, 255, 255, 255 }
		},
		{ { 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 236, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 251, 253, 253, 254, 254, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		}
	},
	{ { { 248, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 250, 254, 252, 254, 255, 255, 255, 255, 255, 255, 255 },
			{ 248, 254, 249, 253, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 246, 253, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 252, 254, 251, 254, 254, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 254, 252, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 248, 254, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 253, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 245, 251, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 253, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 251, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 252, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 249, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 253, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 250, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		},
		{ { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 },
			{ 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255 }
		}
	}
};

/* RFC 6386 section 13.5 */
static const simplewebp_u8 swebp__coeff_proba0[4][8][3][11] = {
	{ { { 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 }
		},
		{ { 253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128 },
			{ 189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128 },
			{ 106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128 }
		},
		{ { 1, 98, 248, 255, 236, 226, 255, 255, 128, 128, 128 },
			{ 181, 133, 238, 254, 221, 234, 255, 154, 128, 128, 128 },
			{ 78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128 },
		},
		{ { 1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128 },
			{ 184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128 },
			{ 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128 },
		},
		{ { 1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128 },
			{ 170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128 },
			{ 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128 }
		},
		{ { 1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128 },
			{ 207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128 },
			{ 102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128 }
		},
		{ { 1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128 },
			{ 177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128 },
			{ 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128 }
		},
		{ { 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 246, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 }
		}
	},
	{ { { 198, 35, 237, 223, 193, 187, 162, 160, 145, 155, 62 },
			{ 131, 45, 198, 221, 172, 176, 220, 157, 252, 221, 1 },
			{ 68, 47, 146, 208, 149, 167, 221, 162, 255, 223, 128 }
		},
		{ { 1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128 },
			{ 184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128 },
			{ 81, 99, 181, 242, 176, 190, 249, 202, 255, 255, 128 }
		},
		{ { 1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128 },
			{ 99, 121, 210, 250, 201, 198, 255, 202, 128, 128, 128 },
			{ 23, 91, 163, 242, 170, 187, 247, 210, 255, 255, 128 }
		},
		{ { 1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128 },
			{ 109, 178, 241, 255, 231, 245, 255, 255, 128, 128, 128 },
			{ 44, 130, 201, 253, 205, 192, 255, 255, 128, 128, 128 }
		},
		{ { 1, 132, 239, 251, 219, 209, 255, 165, 128, 128, 128 },
			{ 94, 136, 225, 251, 218, 190, 255, 255, 128, 128, 128 },
			{ 22, 100, 174, 245, 186, 161, 255, 199, 128, 128, 128 }
		},
		{ { 1, 182, 249, 255, 232, 235, 128, 128, 128, 128, 128 },
			{ 124, 143, 241, 255, 227, 234, 128, 128, 128, 128, 128 },
			{ 35, 77, 181, 251, 193, 211, 255, 205, 128, 128, 128 }
		},
		{ { 1, 157, 247, 255, 236, 231, 255, 255, 128, 128, 128 },
			{ 121, 141, 235, 255, 225, 227, 255, 255, 128, 128, 128 },
			{ 45, 99, 188, 251, 195, 217, 255, 224, 128, 128, 128 }
		},
		{ { 1, 1, 251, 255, 213, 255, 128, 128, 128, 128, 128 },
			{ 203, 1, 248, 255, 255, 128, 128, 128, 128, 128, 128 },
			{ 137, 1, 177, 255, 224, 255, 128, 128, 128, 128, 128 }
		}
	},
	{ { { 253, 9, 248, 251, 207, 208, 255, 192, 128, 128, 128 },
			{ 175, 13, 224, 243, 193, 185, 249, 198, 255, 255, 128 },
			{ 73, 17, 171, 221, 161, 179, 236, 167, 255, 234, 128 }
		},
		{ { 1, 95, 247, 253, 212, 183, 255, 255, 128, 128, 128 },
			{ 239, 90, 244, 250, 211, 209, 255, 255, 128, 128, 128 },
			{ 155, 77, 195, 248, 188, 195, 255, 255, 128, 128, 128 }
		},
		{ { 1, 24, 239, 251, 218, 219, 255, 205, 128, 128, 128 },
			{ 201, 51, 219, 255, 196, 186, 128, 128, 128, 128, 128 },
			{ 69, 46, 190, 239, 201, 218, 255, 228, 128, 128, 128 }
		},
		{ { 1, 191, 251, 255, 255, 128, 128, 128, 128, 128, 128 },
			{ 223, 165, 249, 255, 213, 255, 128, 128, 128, 128, 128 },
			{ 141, 124, 248, 255, 255, 128, 128, 128, 128, 128, 128 }
		},
		{ { 1, 16, 248, 255, 255, 128, 128, 128, 128, 128, 128 },
			{ 190, 36, 230, 255, 236, 255, 128, 128, 128, 128, 128 },
			{ 149, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 }
		},
		{ { 1, 226, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 247, 192, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 240, 128, 255, 128, 128, 128, 128, 128, 128, 128, 128 }
		},
		{ { 1, 134, 252, 255, 255, 128, 128, 128, 128, 128, 128 },
			{ 213, 62, 250, 255, 255, 128, 128, 128, 128, 128, 128 },
			{ 55, 93, 255, 128, 128, 128, 128, 128, 128, 128, 128 }
		},
		{ { 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128 }
		}
	},
	{ { { 202, 24, 213, 235, 186, 191, 220, 160, 240, 175, 255 },
			{ 126, 38, 182, 232, 169, 184, 228, 174, 255, 187, 128 },
			{ 61, 46, 138, 219, 151, 178, 240, 170, 255, 216, 128 }
		},
		{ { 1, 112, 230, 250, 199, 191, 247, 159, 255, 255, 128 },
			{ 166, 109, 228, 252, 211, 215, 255, 174, 128, 128, 128 },
			{ 39, 77, 162, 232, 172, 180, 245, 178, 255, 255, 128 }
		},
		{ { 1, 52, 220, 246, 198, 199, 249, 220, 255, 255, 128 },
			{ 124, 74, 191, 243, 183, 193, 250, 221, 255, 255, 128 },
			{ 24, 71, 130, 219, 154, 170, 243, 182, 255, 255, 128 }
		},
		{ { 1, 182, 225, 249, 219, 240, 255, 224, 128, 128, 128 },
			{ 149, 150, 226, 252, 216, 205, 255, 171, 128, 128, 128 },
			{ 28, 108, 170, 242, 183, 194, 254, 223, 255, 255, 128 }
		},
		{ { 1, 81, 230, 252, 204, 203, 255, 192, 128, 128, 128 },
			{ 123, 102, 209, 247, 188, 196, 255, 233, 128, 128, 128 },
			{ 20, 95, 153, 243, 164, 173, 255, 203, 128, 128, 128 }
		},
		{ { 1, 222, 248, 255, 216, 213, 128, 128, 128, 128, 128 },
			{ 168, 175, 246, 252, 235, 205, 255, 255, 128, 128, 128 },
			{ 47, 116, 215, 255, 211, 212, 255, 255, 128, 128, 128 }
		},
		{ { 1, 121, 236, 253, 212, 214, 255, 255, 128, 128, 128 },
			{ 141, 84, 213, 252, 201, 202, 255, 219, 128, 128, 128 },
			{ 42, 80, 160, 240, 162, 185, 255, 205, 128, 128, 128 }
		},
		{ { 1, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 244, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 },
			{ 238, 1, 255, 128, 128, 128, 128, 128, 128, 128, 128 }
		}
	}
};

static const simplewebp_u8 swebp__fextrarows[3] = {0, 2, 8};

static simplewebp_u8 *swebp__align32(simplewebp_u8 *ptr)
{
	size_t uptr = (size_t) ptr;
	return (simplewebp_u8 *) ((uptr + 31) & (~(size_t)31));
}

/* Clip tables */

static const simplewebp_u8 swebp__abs0[255 + 255 + 1] = {
	0xff, 0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4,
	0xf3, 0xf2, 0xf1, 0xf0, 0xef, 0xee, 0xed, 0xec, 0xeb, 0xea, 0xe9, 0xe8,
	0xe7, 0xe6, 0xe5, 0xe4, 0xe3, 0xe2, 0xe1, 0xe0, 0xdf, 0xde, 0xdd, 0xdc,
	0xdb, 0xda, 0xd9, 0xd8, 0xd7, 0xd6, 0xd5, 0xd4, 0xd3, 0xd2, 0xd1, 0xd0,
	0xcf, 0xce, 0xcd, 0xcc, 0xcb, 0xca, 0xc9, 0xc8, 0xc7, 0xc6, 0xc5, 0xc4,
	0xc3, 0xc2, 0xc1, 0xc0, 0xbf, 0xbe, 0xbd, 0xbc, 0xbb, 0xba, 0xb9, 0xb8,
	0xb7, 0xb6, 0xb5, 0xb4, 0xb3, 0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xac,
	0xab, 0xaa, 0xa9, 0xa8, 0xa7, 0xa6, 0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa0,
	0x9f, 0x9e, 0x9d, 0x9c, 0x9b, 0x9a, 0x99, 0x98, 0x97, 0x96, 0x95, 0x94,
	0x93, 0x92, 0x91, 0x90, 0x8f, 0x8e, 0x8d, 0x8c, 0x8b, 0x8a, 0x89, 0x88,
	0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81, 0x80, 0x7f, 0x7e, 0x7d, 0x7c,
	0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71, 0x70,
	0x6f, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64,
	0x63, 0x62, 0x61, 0x60, 0x5f, 0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58,
	0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50, 0x4f, 0x4e, 0x4d, 0x4c,
	0x4b, 0x4a, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40,
	0x3f, 0x3e, 0x3d, 0x3c, 0x3b, 0x3a, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34,
	0x33, 0x32, 0x31, 0x30, 0x2f, 0x2e, 0x2d, 0x2c, 0x2b, 0x2a, 0x29, 0x28,
	0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x20, 0x1f, 0x1e, 0x1d, 0x1c,
	0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
	0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04,
	0x03, 0x02, 0x01, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
	0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
	0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
	0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c,
	0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44,
	0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50,
	0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c,
	0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74,
	0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80,
	0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c,
	0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4,
	0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0,
	0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc,
	0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
	0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0,
	0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec,
	0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff
};
static const simplewebp_u8 *const swebp__kabs0 = &swebp__abs0[255];

static const simplewebp_u8 swebp__sclip1[1020 + 1020 + 1] = {
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
	0x80, 0x80, 0x80, 0x80, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93,
	0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
	0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab,
	0xac, 0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
	0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
	0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb,
	0xdc, 0xdd, 0xde, 0xdf, 0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
	0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3,
	0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,
	0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
	0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23,
	0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b,
	0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
	0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53,
	0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
	0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b,
	0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
	0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f,
	0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f
};
static const simplewebp_i8 *const swebp__ksclip1 = (const simplewebp_i8 *) &swebp__sclip1[1020];

static const simplewebp_u8 swebp__sclip2[112 + 112 + 1] = {
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
	0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb,
	0xfc, 0xfd, 0xfe, 0xff, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
	0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f
};
static const simplewebp_i8 *const swebp__ksclip2 = (const simplewebp_i8 *) &swebp__sclip2[112];

static const simplewebp_u8 swebp__clip1[255 + 511 + 1] = {
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
	0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14,
	0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
	0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c,
	0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
	0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40, 0x41, 0x42, 0x43, 0x44,
	0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f, 0x50,
	0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c,
	0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74,
	0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80,
	0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c,
	0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4,
	0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0,
	0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbb, 0xbc,
	0xbd, 0xbe, 0xbf, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
	0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf, 0xd0, 0xd1, 0xd2, 0xd3, 0xd4,
	0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe0,
	0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xeb, 0xec,
	0xed, 0xee, 0xef, 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};
static const simplewebp_u8 *const swebp__kclip1 = &swebp__clip1[255];

/* RFC 6386 section 14.4 */

static void swebp__transform_wht(const simplewebp_i16 *in, simplewebp_i16 *out)
{
	simplewebp_i32 temp[16], i;

	for (i = 0; i < 4; i++)
	{
		simplewebp_i32 a0, a1, a2, a3;

		a0 = in[i] + in[i + 12];
		a1 = in[i + 4] + in[i + 8];
		a2 = in[i + 4] - in[i + 8];
		a3 = in[i] - in[i + 12];
		temp[i] = a0 + a1;
		temp[i + 4] = a3 + a2;
		temp[i + 8] = a0 - a1;
		temp[i + 12] = a3 - a2;
	}
	for (i = 0; i < 4; i++)
	{
		simplewebp_i32 dc, a0, a1, a2, a3;

		dc = temp[i * 4] + 3;
		a0 = dc + temp[i * 4 + 3];
		a1 = temp[i * 4 + 1] + temp[i * 4 + 2];
		a2 = temp[i * 4 + 1] - temp[i * 4 + 2];
		a3 = dc - temp[i * 4 + 3];
		out[i * 64] = (a0 + a1) >> 3;
		out[i * 64 + 16] = (a3 + a2) >> 3;
		out[i * 64 + 32] = (a0 - a1) >> 3;
		out[i * 64 + 48] = (a3 - a2) >> 3;
	}
}

static simplewebp_i32 swebp__mul1(simplewebp_i16 a)
{
	return (((simplewebp_i32) a * 20091) >> 16) + a;
}

static simplewebp_i32 swebp__mul2(simplewebp_i16 a)
{
	return ((simplewebp_i32) a * 35468) >> 16;
}


static simplewebp_u8 swebp__clip8b(simplewebp_i32 v) {
	return (!(v & ~0xff)) ? v : (v < 0) ? 0 : 255;
}

static void swebp__store(simplewebp_u8 *out, simplewebp_i32 x, simplewebp_i32 y, simplewebp_i32 v)
{
	out[y * 32 + x] = swebp__clip8b(out[y * 32 + x] + (v >> 3));
}

static void swebp__transform_one(const simplewebp_i16 *in, simplewebp_u8 *out)
{
	simplewebp_i32 tmp[16], i;

	/* Vertical pass */
	for (i = 0; i < 4; i++)
	{
		simplewebp_i32 a, b, c, d;

		a = in[i] + in[i + 8];
		b = in[i] - in[i + 8];
		c = swebp__mul2(in[i + 4]) - swebp__mul1(in[i + 12]);
		d = swebp__mul1(in[i + 4]) + swebp__mul2(in[i + 12]);
		tmp[i * 4] = a + d;
		tmp[i * 4 + 1] = b + c;
		tmp[i * 4 + 2] = b - c;
		tmp[i * 4 + 3] = a - d;
	}
	/* Horizontal pass */
	for (i = 0; i < 4; i++)
	{
		simplewebp_i32 dc, a, b, c, d;

		dc = tmp[i] + 4;
		a = dc + tmp[i + 8];
		b = dc - tmp[i + 8];
		c = swebp__mul2(tmp[i + 4]) - swebp__mul1(tmp[i + 12]);
		d = swebp__mul1(tmp[i + 4]) + swebp__mul2(tmp[i + 12]);
		swebp__store(out, 0, i, a + d);
		swebp__store(out, 1, i, b + c);
		swebp__store(out, 2, i, b - c);
		swebp__store(out, 3, i, a - d);
	}
}

static void swebp__transform(const simplewebp_i16 *in, simplewebp_u8 *out, simplewebp_u8 do_2)
{
	swebp__transform_one(in, out);
	if (do_2)
		swebp__transform_one(in + 16, out + 4);
}

static void swebp__transform_dc(const simplewebp_i16 *in, simplewebp_u8 *out)
{
	simplewebp_i32 dc, x, y;
	dc = in[0] + 4;

	for (y = 0; y < 4; y++)
	{
		for (x = 0; x < 4; x++)
			swebp__store(out, x, y, dc);
	}
}

static void swebp__store2(simplewebp_u8 *out, simplewebp_i32 y, simplewebp_i32 dc, simplewebp_i32 d, simplewebp_i32 c)
{
	swebp__store(out, 0, y, dc + d);
	swebp__store(out, 1, y, dc + c);
	swebp__store(out, 2, y, dc - c);
	swebp__store(out, 3, y, dc - d);
}

static void swebp__transform_ac3(const simplewebp_i16 *in, simplewebp_u8 *out)
{
	simplewebp_i32 a, c4, d4, c1, d1;

	a = in[0] + 4;
	c4 = swebp__mul2(in[4]);
	d4 = swebp__mul1(in[4]);
	c1 = swebp__mul2(in[1]);
	d1 = swebp__mul1(in[1]);
	swebp__store2(out, 0, a + d4, d1, c1);
	swebp__store2(out, 1, a + c4, d1, c1);
	swebp__store2(out, 2, a - c4, d1, c1);
	swebp__store2(out, 3, a - d4, d1, c1);
}

static void swebp__transform_uv(const simplewebp_i16 *in, simplewebp_u8 *out)
{
	swebp__transform(in, out, 1);
	swebp__transform(in + 32 /* 2*16 */, out + 128 /* 4*BPS */, 1);
}

static void swebp__transform_dcuv(const simplewebp_i16 *in, simplewebp_u8 *out)
{
	if (in[0])
		swebp__transform_dc(in, out);
	if (in[16])
		swebp__transform_dc(in + 16, out + 4);
	if (in[32])
		swebp__transform_dc(in + 32, out + 128);
	if (in[48])
		swebp__transform_dc(in + 48, out + 132);
}

static simplewebp_i32 swebp__needsfilter2(const simplewebp_u8 *p, simplewebp_i32 step, simplewebp_i32 t, simplewebp_i32 it)
{
	simplewebp_i32 p3, p2, p1, p0, q0, q1, q2, q3;
	p3 = p[-4 * step];
	p2 = p[-3 * step];
	p1 = p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];
	q2 = p[2 * step];
	q3 = p[3 * step];

	if ((4 * swebp__kabs0[p0 - q0] + swebp__kabs0[p1 - q1]) > t)
		return 0;

	return
		swebp__kabs0[p3 - p2] <= it &&
		swebp__kabs0[p2 - p1] <= it &&
		swebp__kabs0[p1 - p0] <= it &&
		swebp__kabs0[q3 - q2] <= it &&
		swebp__kabs0[q2 - q1] <= it &&
		swebp__kabs0[q1 - q0] <= it;
}

static simplewebp_i32 swebp__hev(const simplewebp_u8 *p, simplewebp_i32 step, simplewebp_i32 thresh)
{
	simplewebp_i32 p1, p0, q0, q1;

	p1 = p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];

	return (swebp__kabs0[p1 - p0] > thresh) || (swebp__kabs0[q1 - q0] > thresh);
}

static void swebp__do_filter2(simplewebp_u8 *p, simplewebp_i32 step)
{
	simplewebp_i32 p1, p0, q0, q1, a, a1, a2;

	p1 =  p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];
	a = 3 * (q0 - p0) + swebp__ksclip1[p1 - q1];
	a1 =  swebp__ksclip2[(a + 4) >> 3];
	a2 =  swebp__ksclip2[(a + 3) >> 3];
	p[-step] = swebp__kclip1[p0 + a2];
	p[0] = swebp__kclip1[q0 - a1];
}

static void swebp__do_filter6(simplewebp_u8 *p, simplewebp_i32 step)
{
	simplewebp_i32 p2, p1, p0, q0, q1, q2, a, a1, a2, a3;

	p2 =  p[-3 * step];
	p1 =  p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];
	q2 = p[2 * step];
	a = swebp__ksclip1[3 * (q0 - p0) + swebp__ksclip1[p1 - q1]];
	a1 = (27 * a + 63) >> 7;
	a2 = (18 * a + 63) >> 7;
	a3 = (9 * a + 63) >> 7;

	p[-3 * step] = swebp__kclip1[p2 + a3];
	p[-2 * step] = swebp__kclip1[p1 + a2];
	p[-step] = swebp__kclip1[p0 + a1];
	p[0] = swebp__kclip1[q0 - a1];
	p[step] = swebp__kclip1[q1 - a2];
	p[2 * step] = swebp__kclip1[q2 - a3];
}

static void swebp__filterloop26(simplewebp_u8 *p, simplewebp_i32 hstride, simplewebp_i32 vstride, simplewebp_i32 size, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	simplewebp_i32 thresh2 = 2 * thresh + 1;

	while (size-- > 0)
	{
		if (swebp__needsfilter2(p, hstride, thresh2, ithresh))
		{
			if (swebp__hev(p, hstride, hev_thresh))
				swebp__do_filter2(p, hstride);
			else
				swebp__do_filter6(p, hstride);
		}

		p += vstride;
	}
}

static void swebp__vfilter16(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop26(p, stride, 1, 16, thresh, ithresh, hev_thresh);
}

static void swebp__do_filter4(simplewebp_u8 *p, simplewebp_i32 step)
{
	simplewebp_i32 p1, p0, q0, q1, a, a1, a2, a3;

	p1 =  p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];
	a = 3 * (q0 - p0);
	a1 = swebp__ksclip2[(a + 4) >> 3];
	a2 = swebp__ksclip2[(a + 3) >> 3];
	a3 = (a1 + 1) >> 1;

	p[-2 * step] = swebp__kclip1[p1 + a3];
	p[-step] = swebp__kclip1[p0 + a2];
	p[0] = swebp__kclip1[q0 - a1];
	p[step] = swebp__kclip1[q1 - a3];
}

static void swebp__filterloop24(simplewebp_u8 *p, simplewebp_i32 hstride, simplewebp_i32 vstride, simplewebp_i32 size, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	simplewebp_i32 thresh2 = 2 * thresh + 1;

	while (size-- > 0)
	{
		if (swebp__needsfilter2(p, hstride, thresh2, ithresh))
		{
			if (swebp__hev(p, hstride, hev_thresh))
				swebp__do_filter2(p, hstride);
			else
				swebp__do_filter4(p, hstride);
		}

		p += vstride;
	}
}

static void swebp__vfilter16_i(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	simplewebp_i32 i;

	for (i = 3; i > 0; i--)
	{
		p += 4 * stride;
		swebp__filterloop24(p, stride, 1, 16, thresh, ithresh, hev_thresh);
	}
}

static void swebp__hfilter16(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop26(p, 1, stride, 16, thresh, ithresh, hev_thresh);
}

static void swebp__vfilter8(simplewebp_u8 *u, simplewebp_u8 *v, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop26(u, stride, 1, 8, thresh, ithresh, hev_thresh);
	swebp__filterloop26(v, stride, 1, 8, thresh, ithresh, hev_thresh);
}

static void swebp__vfilter8_i(simplewebp_u8 *u, simplewebp_u8 *v, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop24(u + 4 * stride, stride, 1, 8, thresh, ithresh, hev_thresh);
	swebp__filterloop24(v + 4 * stride, stride, 1, 8, thresh, ithresh, hev_thresh);
}

static simplewebp_bool swebp__needsfilter(const simplewebp_u8 *p, simplewebp_i32 step, simplewebp_i32 t)
{
	simplewebp_i32 p1, p0, q0, q1;

	p1 =  p[-2 * step];
	p0 = p[-step];
	q0 = p[0];
	q1 = p[step];
	return (4 * swebp__kabs0[p0 - q0] + swebp__kabs0[p1 - q1]) <= t;
}

static void swebp__simple_vfilter16(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh)
{
	simplewebp_i32 i, thresh2;

	thresh2 = 2 * thresh + 1;
	for (i = 0; i < 16; i++)
	{
		if (swebp__needsfilter(p + i, stride, thresh2))
			swebp__do_filter2(p + i, stride);
	}
}

static void swebp__simple_hfilter16(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh)
{
	simplewebp_i32 i, thresh2;
	simplewebp_u8 *target;

	thresh2 = 2 * thresh + 1;
	for (i = 0; i < 16; i++)
	{
		target = p + i * stride;

		if (swebp__needsfilter(target, 1, thresh2))
			swebp__do_filter2(target, 1);
	}
}

static void swebp__simple_vfilter16_i(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh)
{
	simplewebp_i32 k;

	for (k = 3; k > 0; k--)
	{
		p += 4 * stride;
		swebp__simple_vfilter16(p, stride, thresh);
	}
}

static void swebp__simple_hfilter16_i(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh)
{
	simplewebp_i32 k;

	for (k = 3; k > 0; k--)
	{
		p += 4;
		swebp__simple_hfilter16(p, stride, thresh);
	}
}

static void swebp__hfilter16_i(simplewebp_u8 *p, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	simplewebp_i32 k;

	for (k = 3; k > 0; k--)
	{
		p += 4;
		swebp__filterloop24(p, 1, stride, 16, thresh, ithresh, hev_thresh);
	}
}

static void swebp__hfilter8(simplewebp_u8 *u, simplewebp_u8 *v, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop26(u, 1, stride, 8, thresh, ithresh, hev_thresh);
	swebp__filterloop26(v, 1, stride, 8, thresh, ithresh, hev_thresh);
}

static void swebp__hfilter8_i(simplewebp_u8 *u, simplewebp_u8 *v, simplewebp_i32 stride, simplewebp_i32 thresh, simplewebp_i32 ithresh, simplewebp_i32 hev_thresh)
{
	swebp__filterloop24(u + 4, 1, stride, 8, thresh, ithresh, hev_thresh);
	swebp__filterloop24(v + 4, 1, stride, 8, thresh, ithresh, hev_thresh);
}

/* DC */
static void swebp__predluma4_0(simplewebp_u8 *out)
{
	simplewebp_u32 dc;
	simplewebp_i32 i;

	dc = 4;
	for (i = 0; i < 4; i++)
		dc += out[i - 32] + out[-1 + i * 32];
	dc >>= 3;
	for (i = 0; i < 4; i++)
		memset(out + i * 32, dc, 4);
}

static void swebp__truemotion(simplewebp_u8 *out, simplewebp_i32 size)
{
	const simplewebp_u8 *top, *clip0, *clip;
	simplewebp_i32 x, y;

	top = out - 32;
	clip0 = swebp__kclip1 - top[-1];

	for (y = 0; y < size; y++)
	{
		clip = clip0 + out[-1];

		for (x = 0; x < size; x++)
			out[x] = clip[top[x]];

		out += 32;
	}
}

/* TM4 */
static void swebp__predluma4_1(simplewebp_u8 *out)
{
	swebp__truemotion(out, 4);
}

static simplewebp_u8 swebp__avg3(simplewebp_u32 a, simplewebp_u32 b, simplewebp_u32 c)
{
	return (simplewebp_u8) ((a + 2 * b + c + 2) >> 2);
}

/* Vertical */
static void swebp__predluma4_2(simplewebp_u8 *out)
{
	const simplewebp_u8 *top;
	simplewebp_u8 vals[4], i;

	top = out - 32;

	for (i = 0; i < 4; i++)
		vals[i] = swebp__avg3(top[i - 1], top[i], top[i + 1]);
	for (i = 0; i < 4; i++)
		memcpy(out + i * 32, vals, sizeof(vals));
}

static void swebp__from_uint32(simplewebp_u8 *out, simplewebp_u32 value)
{
	/* Little endian */
	out[0] = (simplewebp_u8) value;
	out[1] = (simplewebp_u8) (value >> 8);
	out[2] = (simplewebp_u8) (value >> 16);
	out[3] = (simplewebp_u8) (value >> 24);
}

/* Horizontal*/
static void swebp__predluma4_3(simplewebp_u8 *out)
{
	simplewebp_i32 vals[5], i;
	for (i = -1; i < 4; i++)
		vals[i + 1] = out[-1 + i * 32];

	swebp__from_uint32(out, 0x01010101U * swebp__avg3(vals[0], vals[1], vals[2]));
	swebp__from_uint32(out + 32, 0x01010101U * swebp__avg3(vals[1], vals[2], vals[3]));
	swebp__from_uint32(out + 64, 0x01010101U * swebp__avg3(vals[2], vals[3], vals[4]));
	swebp__from_uint32(out + 96, 0x01010101U * swebp__avg3(vals[3], vals[4], vals[4]));
}

/* Right Down */
static void swebp__predluma4_4(simplewebp_u8 *out)
{
	simplewebp_u32 i, j, k, l, x, a, b, c, d;

	i = out[-1];
	j = out[-1 + 1 * 32];
	k = out[-1 + 2 * 32];
	l = out[-1 + 3 * 32];
	x = out[-1 - 32];
	a = out[0 - 32];
	b = out[1 - 32];
	c = out[2 - 32];
	d = out[3 - 32];

	out[3 * 32 + 0] = swebp__avg3(j, k, l);
	out[3 * 32 + 1] = out[2 * 32 + 0] = swebp__avg3(i, j, k);
	out[3 * 32 + 2] = out[2 * 32 + 1] = out[1 * 32 + 0] = swebp__avg3(x, i, j);
	out[3 * 32 + 3] = out[2 * 32 + 2] = out[1 * 32 + 1] = out[0 * 32 + 0] = swebp__avg3(a, x, i);
	out[2 * 32 + 3] = out[1 * 32 + 2] = out[0 * 32 + 1] = swebp__avg3(b, a, x);
	out[1 * 32 + 3] = out[0 * 32 + 2] = swebp__avg3(c, b, a);
	out[0 * 32 + 3] = swebp__avg3(d, c, b);
}

static simplewebp_u8 swebp__avg2(simplewebp_u32 a, simplewebp_u32 b)
{
	return (simplewebp_u8) ((a + b + 1) >> 1);
}

/* Vertical-Right */
static void swebp__predluma4_5(simplewebp_u8 *out)
{
	simplewebp_u32 i, j, k, x, a, b, c, d;

	i = out[-1];
	j = out[-1 + 1 * 32];
	k = out[-1 + 2 * 32];
	x = out[-1 - 32];
	a = out[0 - 32];
	b = out[1 - 32];
	c = out[2 - 32];
	d = out[3 - 32];

	out[0 * 32 + 0] = out[2 * 32 + 1] = swebp__avg2(x, a);
	out[0 * 32 + 1] = out[2 * 32 + 2] = swebp__avg2(a, b);
	out[0 * 32 + 2] = out[2 * 32 + 3] = swebp__avg2(b, c);
	out[0 * 32 + 3] = swebp__avg2(c, d);
	out[3 * 32 + 0] = swebp__avg3(k, j, i);
	out[2 * 32 + 0] = swebp__avg3(j, i, x);
	out[1 * 32 + 0] = out[3 * 32 + 1] = swebp__avg3(i, x, a);
	out[1 * 32 + 1] = out[3 * 32 + 2] = swebp__avg3(x, a, b);
	out[1 * 32 + 2] = out[3 * 32 + 3] = swebp__avg3(a, b, c);
	out[1 * 32 + 3] = swebp__avg3(b, c, d);
}

/* Left Down */
static void swebp__predluma4_6(simplewebp_u8 *out)
{
	simplewebp_i32 a, b, c, d, e, f, g, h;

	a = out[-32];
	b = out[-31];
	c = out[-30];
	d = out[-29];
	e = out[-28];
	f = out[-27];
	g = out[-26];
	h = out[-25];

	out[0 * 32 + 0] = swebp__avg3(a, b, c);
	out[0 * 32 + 1] = out[1 * 32 + 0] = swebp__avg3(b, c, d);
	out[0 * 32 + 2] = out[1 * 32 + 1] = out[2 * 32 + 0] = swebp__avg3(c, d, e);
	out[0 * 32 + 3] = out[1 * 32 + 2] = out[2 * 32 + 1] = out[3 * 32 + 0] = swebp__avg3(d, e, f);
	out[1 * 32 + 3] = out[2 * 32 + 2] = out[3 * 32 + 1] = swebp__avg3(e, f, g);
	out[2 * 32 + 3] = out[3 * 32 + 2] = swebp__avg3(f, g, h);
	out[3 * 32 + 3] = swebp__avg3(g, h, h);
}

/* Vertical-Left */
static void swebp__predluma4_7(simplewebp_u8 *out)
{
	simplewebp_i32 a, b, c, d, e, f, g, h;

	a = out[-32];
	b = out[-31];
	c = out[-30];
	d = out[-29];
	e = out[-28];
	f = out[-27];
	g = out[-26];
	h = out[-25];

	out[0 * 32 + 0] = swebp__avg2(a, b);
	out[0 * 32 + 1] = out[2 * 32 + 0] = swebp__avg2(b, c);
	out[0 * 32 + 2] = out[2 * 32 + 1] = swebp__avg2(c, d);
	out[0 * 32 + 3] = out[2 * 32 + 2] = swebp__avg2(d, e);
	out[1 * 32 + 0] = swebp__avg3(a, b, c);
	out[1 * 32 + 1] = out[3 * 32 + 0] = swebp__avg3(b, c, d);
	out[1 * 32 + 2] = out[3 * 32 + 1] = swebp__avg3(c, d, e);
	out[1 * 32 + 3] = out[3 * 32 + 2] = swebp__avg3(d, e, f);
	out[2 * 32 + 3] = swebp__avg3(e, f, g);
	out[3 * 32 + 3] = swebp__avg3(f, g, h);
}

/* Horizontal-Down */
static void swebp__predluma4_8(simplewebp_u8 *out)
{
	simplewebp_u32 i, j, k, l, x, a, b, c;

	i = out[-1];
	j = out[-1 + 1 * 32];
	k = out[-1 + 2 * 32];
	l = out[-1 + 3 * 32];
	x = out[-1 - 32];
	a = out[0 - 32];
	b = out[1 - 32];
	c = out[2 - 32];
	
	out[0 * 32 + 0] = out[1 * 32 + 2] = swebp__avg2(i, x);
	out[1 * 32 + 0] = out[2 * 32 + 2] = swebp__avg2(j, i);
	out[2 * 32 + 0] = out[3 * 32 + 2] = swebp__avg2(k, j);
	out[3 * 32 + 0] = swebp__avg2(l, k);
	out[0 * 32 + 3] = swebp__avg3(a, b, c);
	out[0 * 32 + 2] = swebp__avg3(x, a, b);
	out[0 * 32 + 1] = out[1 * 32 + 3] = swebp__avg3(i, x, a);
	out[1 * 32 + 1] = out[2 * 32 + 3] = swebp__avg3(j, i, x);
	out[2 * 32 + 1] = out[3 * 32 + 3] = swebp__avg3(k, j, i);
	out[3 * 32 + 1] = swebp__avg3(l, k, j);
}

/* Horizontal-Up */
static void swebp__predluma4_9(simplewebp_u8 *out)
{
	simplewebp_u32 i, j, k, l;

	i = out[-1];
	j = out[-1 + 1 * 32];
	k = out[-1 + 2 * 32];
	l = out[-1 + 3 * 32];

	out[0 * 32 + 0] = swebp__avg2(i, j);
	out[0 * 32 + 2] = out[1 * 32 + 0] = swebp__avg2(j, k);
	out[1 * 32 + 2] = out[2 * 32 + 0] = swebp__avg2(k, l);
	out[0 * 32 + 1] = swebp__avg3(i, j, k);
	out[0 * 32 + 3] = out[1 * 32 + 1] = swebp__avg3(j, k, l);
	out[1 * 32 + 3] = out[2 * 32 + 1] = swebp__avg3(k, l, l);
	out[2 * 32 + 3] = out[2 * 32 + 2] =
	out[3 * 32 + 0] = out[3 * 32 + 1] =
	out[3 * 32 + 2] = out[3 * 32 + 3] = l;
}

static void swebp__predluma4(simplewebp_u8 num, simplewebp_u8 *out)
{
	switch (num)
	{
		case 0:
			swebp__predluma4_0(out);
			break;
		case 1:
			swebp__predluma4_1(out);
			break;
		case 2:
			swebp__predluma4_2(out);
			break;
		case 3:
			swebp__predluma4_3(out);
			break;
		case 4:
			swebp__predluma4_4(out);
			break;
		case 5:
			swebp__predluma4_5(out);
			break;
		case 6:
			swebp__predluma4_6(out);
			break;
		case 7:
			swebp__predluma4_7(out);
			break;
		case 8:
			swebp__predluma4_8(out);
			break;
		case 9:
			swebp__predluma4_9(out);
			break;
		default:
			break;
	}
}

static void swebp__put16(simplewebp_i32 v, simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 0; j < 16; j++)
		memset(out + j * 32, v, 16);
}

/* DC */
static void swebp__predluma16_0(simplewebp_u8 *out)
{
	simplewebp_i32 dc, j;

	dc = 16;
	for (j = 0; j < 16; j++)
		dc += out[-1 + j * 32] + out[j - 32];

	swebp__put16(dc >> 5, out);
}

/* TM */
static void swebp__predluma16_1(simplewebp_u8 *out)
{
	swebp__truemotion(out, 16);
}

/* Vertical */
static void swebp__predluma16_2(simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 0; j < 16; j++)
		memcpy(out + j * 32, out - 32, 16);
}

/* Horizontal */
static void swebp__predluma16_3(simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 16; j > 0; j--)
	{
		memset(out, out[-1], 16);
		out += 32;
	}
}

/* DC w/o Top */
static void swebp__predluma16_4(simplewebp_u8 *out)
{
	simplewebp_i32 dc, j;

	dc = 8;
	for (j = 0; j < 16; j++)
		dc += out[-1 + j * 32];

	swebp__put16(dc >> 4, out);
}

/* DC w/o Left */
static void swebp__predluma16_5(simplewebp_u8 *out)
{
	simplewebp_i32 dc, j;

	dc = 8;
	for (j = 0; j < 16; j++)
		dc += out[j - 32];

	swebp__put16(dc >> 4, out);
}

/* DC w/o Top Left */
static void swebp__predluma16_6(simplewebp_u8 *out)
{
	swebp__put16(128, out);
}

static void swebp__predluma16(simplewebp_u8 num, simplewebp_u8 *out)
{
	switch (num)
	{
		case 0:
			swebp__predluma16_0(out);
			break;
		case 1:
			swebp__predluma16_1(out);
			break;
		case 2:
			swebp__predluma16_2(out);
			break;
		case 3:
			swebp__predluma16_3(out);
			break;
		case 4:
			swebp__predluma16_4(out);
			break;
		case 5:
			swebp__predluma16_5(out);
			break;
		case 6:
			swebp__predluma16_6(out);
			break;
		default:
			break;
	}
}

static void swebp__put8x8uv(simplewebp_i32 value, simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 0; j < 8; j++)
		memset(out + j * 32, value, 8);
}

/* DC */
static void swebp__predchroma8_0(simplewebp_u8 *out)
{
	simplewebp_i32 dc0, i;

	dc0 = 8;
	for (i = 0; i < 8; i++)
		dc0 += out[i - 32] + out[-1 + i * 32];

	swebp__put8x8uv(dc0 >> 4, out);
}

/* TM */
static void swebp__predchroma8_1(simplewebp_u8 *out)
{
	swebp__truemotion(out, 8);
}

/* Vertical */
static void swebp__predchroma8_2(simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 0; j < 8; j++)
		memcpy(out + j * 32, out - 32, 8);
}

/* Horizontal */
static void swebp__predchroma8_3(simplewebp_u8 *out)
{
	simplewebp_i32 j;
	for (j = 0; j < 8; j++)
		memset(out + j * 32, out[j * 32 - 1], 8);
}

/* DC w/o Top */
static void swebp__predchroma8_4(simplewebp_u8 *out)
{
	simplewebp_i32 dc0, i;

	dc0 = 4;
	for (i = 0; i < 8; i++)
		dc0 += out[-1 + i * 32];

	swebp__put8x8uv(dc0 >> 3, out);
}

/* DC w/o Left */
static void swebp__predchroma8_5(simplewebp_u8 *out)
{
	simplewebp_i32 dc0, i;

	dc0 = 4;
	for (i = 0; i < 8; i++)
		dc0 += out[i - 32];

	swebp__put8x8uv(dc0 >> 3, out);
}

/* DC w/o Top Left */
static void swebp__predchroma8_6(simplewebp_u8 *out)
{
	swebp__put8x8uv(128, out);
}

static void swebp__predchroma8(simplewebp_u8 num, simplewebp_u8 *out)
{
	switch (num)
	{
		case 0:
			swebp__predchroma8_0(out);
			break;
		case 1:
			swebp__predchroma8_1(out);
			break;
		case 2:
			swebp__predchroma8_2(out);
			break;
		case 3:
			swebp__predchroma8_3(out);
			break;
		case 4:
			swebp__predchroma8_4(out);
			break;
		case 5:
			swebp__predchroma8_5(out);
			break;
		case 6:
			swebp__predchroma8_6(out);
			break;
		default:
			break;
	}
}

/* RFC 6386 section 11.5 */
static const simplewebp_u8 swebp__modes_proba[10][10][9] = {
	{ { 231, 120, 48, 89, 115, 113, 120, 152, 112 },
		{ 152, 179, 64, 126, 170, 118, 46, 70, 95 },
		{ 175, 69, 143, 80, 85, 82, 72, 155, 103 },
		{ 56, 58, 10, 171, 218, 189, 17, 13, 152 },
		{ 114, 26, 17, 163, 44, 195, 21, 10, 173 },
		{ 121, 24, 80, 195, 26, 62, 44, 64, 85 },
		{ 144, 71, 10, 38, 171, 213, 144, 34, 26 },
		{ 170, 46, 55, 19, 136, 160, 33, 206, 71 },
		{ 63, 20, 8, 114, 114, 208, 12, 9, 226 },
		{ 81, 40, 11, 96, 182, 84, 29, 16, 36 } },
	{ { 134, 183, 89, 137, 98, 101, 106, 165, 148 },
		{ 72, 187, 100, 130, 157, 111, 32, 75, 80 },
		{ 66, 102, 167, 99, 74, 62, 40, 234, 128 },
		{ 41, 53, 9, 178, 241, 141, 26, 8, 107 },
		{ 74, 43, 26, 146, 73, 166, 49, 23, 157 },
		{ 65, 38, 105, 160, 51, 52, 31, 115, 128 },
		{ 104, 79, 12, 27, 217, 255, 87, 17, 7 },
		{ 87, 68, 71, 44, 114, 51, 15, 186, 23 },
		{ 47, 41, 14, 110, 182, 183, 21, 17, 194 },
		{ 66, 45, 25, 102, 197, 189, 23, 18, 22 } },
	{ { 88, 88, 147, 150, 42, 46, 45, 196, 205 },
		{ 43, 97, 183, 117, 85, 38, 35, 179, 61 },
		{ 39, 53, 200, 87, 26, 21, 43, 232, 171 },
		{ 56, 34, 51, 104, 114, 102, 29, 93, 77 },
		{ 39, 28, 85, 171, 58, 165, 90, 98, 64 },
		{ 34, 22, 116, 206, 23, 34, 43, 166, 73 },
		{ 107, 54, 32, 26, 51, 1, 81, 43, 31 },
		{ 68, 25, 106, 22, 64, 171, 36, 225, 114 },
		{ 34, 19, 21, 102, 132, 188, 16, 76, 124 },
		{ 62, 18, 78, 95, 85, 57, 50, 48, 51 } },
	{ { 193, 101, 35, 159, 215, 111, 89, 46, 111 },
		{ 60, 148, 31, 172, 219, 228, 21, 18, 111 },
		{ 112, 113, 77, 85, 179, 255, 38, 120, 114 },
		{ 40, 42, 1, 196, 245, 209, 10, 25, 109 },
		{ 88, 43, 29, 140, 166, 213, 37, 43, 154 },
		{ 61, 63, 30, 155, 67, 45, 68, 1, 209 },
		{ 100, 80, 8, 43, 154, 1, 51, 26, 71 },
		{ 142, 78, 78, 16, 255, 128, 34, 197, 171 },
		{ 41, 40, 5, 102, 211, 183, 4, 1, 221 },
		{ 51, 50, 17, 168, 209, 192, 23, 25, 82 } },
	{ { 138, 31, 36, 171, 27, 166, 38, 44, 229 },
		{ 67, 87, 58, 169, 82, 115, 26, 59, 179 },
		{ 63, 59, 90, 180, 59, 166, 93, 73, 154 },
		{ 40, 40, 21, 116, 143, 209, 34, 39, 175 },
		{ 47, 15, 16, 183, 34, 223, 49, 45, 183 },
		{ 46, 17, 33, 183, 6, 98, 15, 32, 183 },
		{ 57, 46, 22, 24, 128, 1, 54, 17, 37 },
		{ 65, 32, 73, 115, 28, 128, 23, 128, 205 },
		{ 40, 3, 9, 115, 51, 192, 18, 6, 223 },
		{ 87, 37, 9, 115, 59, 77, 64, 21, 47 } },
	{ { 104, 55, 44, 218, 9, 54, 53, 130, 226 },
		{ 64, 90, 70, 205, 40, 41, 23, 26, 57 },
		{ 54, 57, 112, 184, 5, 41, 38, 166, 213 },
		{ 30, 34, 26, 133, 152, 116, 10, 32, 134 },
		{ 39, 19, 53, 221, 26, 114, 32, 73, 255 },
		{ 31, 9, 65, 234, 2, 15, 1, 118, 73 },
		{ 75, 32, 12, 51, 192, 255, 160, 43, 51 },
		{ 88, 31, 35, 67, 102, 85, 55, 186, 85 },
		{ 56, 21, 23, 111, 59, 205, 45, 37, 192 },
		{ 55, 38, 70, 124, 73, 102, 1, 34, 98 } },
	{ { 125, 98, 42, 88, 104, 85, 117, 175, 82 },
		{ 95, 84, 53, 89, 128, 100, 113, 101, 45 },
		{ 75, 79, 123, 47, 51, 128, 81, 171, 1 },
		{ 57, 17, 5, 71, 102, 57, 53, 41, 49 },
		{ 38, 33, 13, 121, 57, 73, 26, 1, 85 },
		{ 41, 10, 67, 138, 77, 110, 90, 47, 114 },
		{ 115, 21, 2, 10, 102, 255, 166, 23, 6 },
		{ 101, 29, 16, 10, 85, 128, 101, 196, 26 },
		{ 57, 18, 10, 102, 102, 213, 34, 20, 43 },
		{ 117, 20, 15, 36, 163, 128, 68, 1, 26 } },
	{ { 102, 61, 71, 37, 34, 53, 31, 243, 192 },
		{ 69, 60, 71, 38, 73, 119, 28, 222, 37 },
		{ 68, 45, 128, 34, 1, 47, 11, 245, 171 },
		{ 62, 17, 19, 70, 146, 85, 55, 62, 70 },
		{ 37, 43, 37, 154, 100, 163, 85, 160, 1 },
		{ 63, 9, 92, 136, 28, 64, 32, 201, 85 },
		{ 75, 15, 9, 9, 64, 255, 184, 119, 16 },
		{ 86, 6, 28, 5, 64, 255, 25, 248, 1 },
		{ 56, 8, 17, 132, 137, 255, 55, 116, 128 },
		{ 58, 15, 20, 82, 135, 57, 26, 121, 40 } },
	{ { 164, 50, 31, 137, 154, 133, 25, 35, 218 },
		{ 51, 103, 44, 131, 131, 123, 31, 6, 158 },
		{ 86, 40, 64, 135, 148, 224, 45, 183, 128 },
		{ 22, 26, 17, 131, 240, 154, 14, 1, 209 },
		{ 45, 16, 21, 91, 64, 222, 7, 1, 197 },
		{ 56, 21, 39, 155, 60, 138, 23, 102, 213 },
		{ 83, 12, 13, 54, 192, 255, 68, 47, 28 },
		{ 85, 26, 85, 85, 128, 128, 32, 146, 171 },
		{ 18, 11, 7, 63, 144, 171, 4, 4, 246 },
		{ 35, 27, 10, 146, 174, 171, 12, 26, 128 } },
	{ { 190, 80, 35, 99, 180, 80, 126, 54, 45 },
		{ 85, 126, 47, 87, 176, 51, 41, 20, 32 },
		{ 101, 75, 128, 139, 118, 146, 116, 128, 85 },
		{ 56, 41, 15, 176, 236, 85, 37, 9, 62 },
		{ 71, 30, 17, 119, 118, 255, 17, 18, 138 },
		{ 101, 38, 60, 138, 55, 70, 43, 26, 142 },
		{ 146, 36, 19, 30, 171, 255, 97, 27, 20 },
		{ 138, 45, 61, 62, 219, 1, 81, 188, 64 },
		{ 32, 41, 20, 117, 151, 142, 20, 21, 163 },
		{ 112, 19, 12, 61, 195, 128, 48, 4, 24 } }
};

static const simplewebp_u8 swebp__bands[17] = {
	0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0
};

static simplewebp_error swebp__load_vp8_header(struct swebp__vp8 *vp8d, simplewebp_u8 *buf, size_t bufsize)
{
	struct swebp__segment_header *segmnt_hdr;
	struct swebp__filter_header *filt_hdr;
	struct swebp__bdec br;

	/* Populate picture headers */
	vp8d->mb_w = (vp8d->picture_header.width + 15) >> 4;
	vp8d->mb_h = (vp8d->picture_header.height + 15) >> 4;

	/* Reset proba */
	memset(vp8d->proba.segments, 255, sizeof(vp8d->proba.segments));
	/* Reset segment header */
	vp8d->segment_header.use_segment = 0;
	vp8d->segment_header.update_map = 0;
	vp8d->segment_header.absolute_delta = 1;
	memset(&vp8d->segment_header.quantizer, 0, sizeof(vp8d->segment_header.quantizer));
	memset(&vp8d->segment_header.filter_strength, 0, sizeof(vp8d->segment_header.filter_strength));

	/* Initialize bitreader */
	swebp__bitread_init(&br, buf, vp8d->frame_header.partition_length);
	buf += vp8d->frame_header.partition_length;
	bufsize -= vp8d->frame_header.partition_length;

	/* Read more picture headers. This is a keyframe */
	vp8d->picture_header.colorspace = swebp__bitread_getval(&br, 1);
	vp8d->picture_header.clamp_type = swebp__bitread_getval(&br, 1);

	/* Parse segment header */
	segmnt_hdr = &vp8d->segment_header;
	segmnt_hdr->use_segment = swebp__bitread_getval(&br, 1);
	if (segmnt_hdr->use_segment)
	{
		simplewebp_i32 s;
		segmnt_hdr->update_map = swebp__bitread_getval(&br, 1);

		if (swebp__bitread_getval(&br, 1) /* update data */)
		{
			segmnt_hdr->absolute_delta = swebp__bitread_getval(&br, 1);
			for (s = 0; s < 4; s++)
				segmnt_hdr->quantizer[s] = swebp__bitread_getval(&br, 1)
					? swebp__bitread_getvalsigned(&br, 7)
					: 0;
			for (s = 0; s < 4; s++)
				segmnt_hdr->filter_strength[s] = swebp__bitread_getval(&br, 1)
					? swebp__bitread_getvalsigned(&br, 6)
					: 0;
			
		}

		if (segmnt_hdr->update_map)
		{
			for (s = 0; s < 3; s++)
				vp8d->proba.segments[s] = swebp__bitread_getval(&br, 1)
					? swebp__bitread_getval(&br, 8)
					: 255;
		}
	}
	else
		segmnt_hdr->update_map = 0;

	if (br.eof)
		return SIMPLEWEBP_CORRUPT_ERROR;

	/* Parse filters */
	filt_hdr = &vp8d->filter_header;
	filt_hdr->simple = swebp__bitread_getval(&br, 1);
	filt_hdr->level = swebp__bitread_getval(&br, 6);
	filt_hdr->sharpness = swebp__bitread_getval(&br, 3);
	filt_hdr->use_lf_delta = swebp__bitread_getval(&br, 1);

	if (filt_hdr->use_lf_delta)
	{
		if (swebp__bitread_getval(&br, 1) /* update lf-delta? */)
		{
			simplewebp_i32 i;
			
			for (i = 0; i < 4; i++)
			{
				if (swebp__bitread_getval(&br, 1))
					filt_hdr->ref_lf_delta[i] = swebp__bitread_getvalsigned(&br, 6);
			}

			for (i = 0; i < 4; i++)
			{
				if (swebp__bitread_getval(&br, 1))
					filt_hdr->mode_lf_delta[i] = swebp__bitread_getvalsigned(&br, 6);
			}
		}
	}

	vp8d->filter_type = filt_hdr->level == 0 ? 0 : (filt_hdr->simple ? 1 : 2);

	if (br.eof)
		return SIMPLEWEBP_CORRUPT_ERROR;

	/* Parse partitions */
	{
		simplewebp_u8 *sz, *buf_end, *part_start;
		size_t size_left, last_part, p;

		sz = buf;
		buf_end = buf + bufsize;
		size_left = bufsize;

		last_part = vp8d->nparts_minus_1 = (1 << swebp__bitread_getval(&br, 2)) - 1;
		if (3 * last_part > bufsize)
			return SIMPLEWEBP_CORRUPT_ERROR;

		part_start = buf + last_part * 3;
		size_left -= last_part * 3;

		for (p = 0; p < last_part; p++)
		{
			size_t psize = swebp__to_uint24(sz);
			if (psize > size_left)
				psize = size_left;

			swebp__bitread_init(vp8d->parts + p, part_start, psize);
			part_start += psize;
			size_left -= psize;
			sz += 3;
		}

		swebp__bitread_init(vp8d->parts + last_part, part_start, size_left);
		if (part_start >= buf_end)
			return SIMPLEWEBP_CORRUPT_ERROR;
	}

	/* Parse quantizer */
	{
		simplewebp_i32 base_q0, dqy1_dc, dqy2_dc, dqy2_ac, dquv_dc, dquv_ac, i;
		base_q0 = swebp__bitread_getval(&br, 7);
		dqy1_dc = swebp__bitread_getval(&br, 1)
			? swebp__bitread_getvalsigned(&br, 4)
			: 0;
		dqy2_dc = swebp__bitread_getval(&br, 1)
			? swebp__bitread_getvalsigned(&br, 4)
			: 0;
		dqy2_ac = swebp__bitread_getval(&br, 1)
			? swebp__bitread_getvalsigned(&br, 4)
			: 0;
		dquv_dc = swebp__bitread_getval(&br, 1)
			? swebp__bitread_getvalsigned(&br, 4)
			: 0;
		dquv_ac = swebp__bitread_getval(&br, 1)
			? swebp__bitread_getvalsigned(&br, 4)
			: 0;

		for (i = 0; i < 4; i++)
		{
			simplewebp_i32 q;
			struct swebp__quantmat *m;

			if (segmnt_hdr->use_segment)
				q = segmnt_hdr->quantizer[i] + (!segmnt_hdr->absolute_delta) * base_q0;
			else
			{
				if (i > 0)
				{
					vp8d->dqm[i] = vp8d->dqm[0];
					continue;
				}
				else
					q = base_q0;
			}

			m = vp8d->dqm + i;
			m->y1_mat[0] = swebp__dctab[swebp__clip(q + dqy1_dc, 127)];
			m->y1_mat[1] = swebp__actab[swebp__clip(q + 0,       127)];

			m->y2_mat[0] = swebp__dctab[swebp__clip(q + dqy2_dc, 127)] * 2;
			m->y2_mat[1] = (swebp__actab[swebp__clip(q + dqy2_ac, 127)] * 101581) >> 16;
			if (m->y2_mat[1] < 8)
				m->y2_mat[1] = 8;

			m->uv_mat[0] = swebp__dctab[swebp__clip(q + dquv_dc, 117)];
			m->uv_mat[1] = swebp__actab[swebp__clip(q + dquv_ac, 127)];

			m->uv_quant = q + dquv_ac;
		}
	}

	/* Ignore "update proba" */
	swebp__bitread_getval(&br, 1);

	/* Parse proba */
	{
		struct swebp__proba *proba;
		simplewebp_i32 t, b, c, p, v;

		proba = &vp8d->proba;

		for (t = 0; t < 4; t++)
		{
			for (b = 0; b < 8; b++)
			{
				for (c = 0; c < 3; c++)
				{
					for (p = 0; p < 11; p++)
					{
						v = swebp__bitread_getbit(&br, swebp__coeff_update_proba[t][b][c][p])
							? swebp__bitread_getval(&br, 8)
							: swebp__coeff_proba0[t][b][c][p];
						proba->bands[t][b].probas[c][p] = v;
					}
				}
			}

			for (b = 0; b < 17; b++)
				proba->bands_ptr[t][b] = &proba->bands[t][swebp__bands[b]];
		}

		vp8d->use_skip_proba = swebp__bitread_getval(&br, 1);
		if (vp8d->use_skip_proba)
			vp8d->skip_proba = swebp__bitread_getval(&br, 8);
	}

	if (br.eof)
		return SIMPLEWEBP_CORRUPT_ERROR;

	vp8d->br = br;
	vp8d->ready = 1;
	return SIMPLEWEBP_NO_ERROR;
}

static void swebp__vp8_enter_critical(struct swebp__vp8 *vp8d)
{
	vp8d->tl_mb_x = vp8d->tl_mb_y = 0;
	vp8d->br_mb_x = vp8d->mb_w;
	vp8d->br_mb_y = vp8d->mb_h;

	if (vp8d->filter_type > 0)
	{
		simplewebp_i32 s;
		struct swebp__filter_header *filt_hdr;

		filt_hdr = &vp8d->filter_header;

		for (s = 0; s < 4; s++)
		{
			simplewebp_i32 i4x4, base_level;

			if (vp8d->segment_header.use_segment)
				base_level = vp8d->segment_header.filter_strength[s]
					+ filt_hdr->level * (!vp8d->segment_header.absolute_delta);
			else
				base_level = filt_hdr->level;

			for (i4x4 = 0; i4x4 <= 1; i4x4++)
			{
				struct swebp__finfo *info;
				simplewebp_i32 level;

				info = &vp8d->fstrengths[s][i4x4];
				level = base_level
					+ filt_hdr->ref_lf_delta[0] * filt_hdr->use_lf_delta
					+ filt_hdr->mode_lf_delta[0] * i4x4 * filt_hdr->use_lf_delta;

				level = (level < 0) ? 0 : ((level > 63) ? 63 : level);
				if (level > 0)
				{
					simplewebp_i32 ilevel = level;

					if (filt_hdr->sharpness > 0)
					{
						ilevel >>= 1 + (filt_hdr->sharpness > 4);

						if (ilevel > 9 - filt_hdr->sharpness)
							ilevel = 9 - filt_hdr->sharpness;
					}

					if (ilevel < 1)
						ilevel = 1;

					info->ilevel = ilevel;
					info->limit = 2 * level + ilevel;
					info->hev_thresh = (level >= 40) + (level >= 15);
				}
				else
					info->limit = 0;

				info->inner = i4x4;
			}
		}
	}
}

static void swebp__vp8_parse_intra_mode(struct swebp__vp8 *vp8d, simplewebp_i32 mb_x)
{
	simplewebp_u8 *top, *left;
	struct swebp__mblockdata *block;
	struct swebp__bdec *br;

	top = vp8d->intra_t + 4 * mb_x;
	left = vp8d->intra_l;
	block = vp8d->mb_data + mb_x;
	br = &vp8d->br;

	if (vp8d->segment_header.update_map)
		block->segment = !swebp__bitread_getbit(br, vp8d->proba.segments[0])
			? swebp__bitread_getbit(br, vp8d->proba.segments[1])
			: swebp__bitread_getbit(br, vp8d->proba.segments[2]) + 2;
	else
		block->segment = 0;

	if (vp8d->use_skip_proba)
		block->skip = swebp__bitread_getbit(br, vp8d->skip_proba);

	block->is_i4x4 = !swebp__bitread_getbit(br, 145);
	if (!block->is_i4x4)
	{
		simplewebp_i32 ymode = swebp__bitread_getbit(br, 156)
			? (swebp__bitread_getbit(br, 128) ? 1 : 3)
			: (swebp__bitread_getbit(br, 163) ? 2 : 0);

		block->imodes[0] = ymode;
		memset(top, ymode, 4);
		memset(left, ymode, 4);
	}
	else
	{
		simplewebp_u8 *modes;
		simplewebp_i32 y;

		modes = block->imodes;
		for (y = 0; y < 4; y++)
		{
			simplewebp_i32 ymode, x;

			ymode = left[y];
			for (x = 0; x < 4; x++)
			{
				const simplewebp_u8 *const prob = swebp__modes_proba[top[x]][ymode];
				ymode = !swebp__bitread_getbit(br, prob[0]) ? 0 :
					!swebp__bitread_getbit(br, prob[1]) ? 1 :
						!swebp__bitread_getbit(br, prob[2]) ? 2 :
							!swebp__bitread_getbit(br, prob[3]) ?
								(!swebp__bitread_getbit(br, prob[4]) ? 3 :
									(!swebp__bitread_getbit(br, prob[5]) ? 4
																				: 5)) :
								(!swebp__bitread_getbit(br, prob[6]) ? 6 :
									(!swebp__bitread_getbit(br, prob[7]) ? 7 :
										(!swebp__bitread_getbit(br, prob[8]) ? 8
																					: 9))
								);
				top[x] = ymode;
			}

			memcpy(modes, top, 4);
			modes += 4;
			left[y] = ymode;
		}
	}

	block->uvmode = !swebp__bitread_getbit(br, 142) ? 0 : (
		!swebp__bitread_getbit(br, 114) ? 2 : (
			swebp__bitread_getbit(br, 183) ? 1 : 3
		));
}

static simplewebp_i32 swebp__vp8_parse_intra_row(struct swebp__vp8 *vp8d)
{
	simplewebp_i32 mb_x;
	
	for (mb_x = 0; mb_x < vp8d->mb_w; mb_x++)
		swebp__vp8_parse_intra_mode(vp8d, mb_x);

	return !vp8d->br.eof;
}

static void swebp__vp8_init_scanline(struct swebp__vp8 *vp8d)
{
	struct swebp__mblock *const left = vp8d->mb_info - 1;
	left->nz = 0;
	left->nz_dc = 0;
	memset(vp8d->intra_l, 0, sizeof(vp8d->intra_l));
	vp8d->mb_x = 0;
}

static simplewebp_u8 *swebp__vp8_alloc_memory(struct swebp__vp8 *vp8d, simplewebp_allocator *allocator)
{
	simplewebp_i32 mb_w;
	size_t intra_pred_mode_size, top_size, mb_info_size, f_info_size;
	size_t yuv_size, mb_data_size, cache_height, cache_size, alpha_size, needed;
	simplewebp_u8 *orig_mem, *mem;

	mb_w = vp8d->mb_w;
	intra_pred_mode_size = 4 * mb_w;
	top_size = sizeof(struct swebp__topsmp) * mb_w;
	mb_info_size = (mb_w + 1) * sizeof(struct swebp__mblock);
	f_info_size = vp8d->filter_type > 0 ? (mb_w * sizeof(struct swebp__finfo)) : 0;
	yuv_size = (32 * 17 + 32 * 9) * sizeof(*vp8d->yuv_b);
	mb_data_size = mb_w * sizeof(*vp8d->mb_data);
	cache_height = (16 + swebp__fextrarows[vp8d->filter_type]) * 3 / 2;
	cache_size = top_size * cache_height;
	alpha_size = vp8d->picture_header.width * vp8d->picture_header.width;

	needed = intra_pred_mode_size
		+ top_size
		+ mb_info_size
		+ f_info_size
		+ yuv_size
		+ mb_data_size
		+ cache_size
		+ alpha_size + 31;

	mem = orig_mem = (simplewebp_u8 *) allocator->alloc(allocator->userdata, needed);

	if (mem)
	{
		vp8d->mem = mem;
		vp8d->mem_size = needed;

		vp8d->intra_t = mem;
		mem += intra_pred_mode_size;

		vp8d->yuv_t = (struct swebp__topsmp *) mem;
		mem += top_size;

		vp8d->mb_info = ((struct swebp__mblock *) mem) + 1;
		mem += mb_info_size;

		vp8d->f_info = f_info_size ? (struct swebp__finfo *) mem : NULL;
		mem += f_info_size;

		mem = swebp__align32(mem);
		vp8d->yuv_b = mem;
		mem += yuv_size;

		vp8d->mb_data = (struct swebp__mblockdata *) mem;
		mem += mb_data_size;

		vp8d->cache_y_stride = 16 * mb_w;
		vp8d->cache_uv_stride = 8 * mb_w;
		{
			simplewebp_i32 extra_rows, extra_y, extra_uv;

			extra_rows = swebp__fextrarows[vp8d->filter_type];
			extra_y = extra_rows * vp8d->cache_y_stride;
			extra_uv = (extra_rows / 2) * vp8d->cache_uv_stride;
			vp8d->cache_y = mem + extra_y;
			vp8d->cache_u = vp8d->cache_y + 16 * vp8d->cache_y_stride + extra_uv;
			vp8d->cache_v = vp8d->cache_u + 8 * vp8d->cache_uv_stride + extra_uv;
		}
		mem += cache_size;

		vp8d->alpha_plane = alpha_size ? mem : NULL;
		mem += alpha_size;

		memset(vp8d->mb_info - 1, 0, mb_info_size);
		swebp__vp8_init_scanline(vp8d);
		memset(vp8d->intra_t, 0, intra_pred_mode_size);
	}

	return orig_mem;
}

static const simplewebp_u8 swebp__cat3[] = { 173, 148, 140, 0 };
static const simplewebp_u8 swebp__cat4[] = { 176, 155, 140, 135, 0 };
static const simplewebp_u8 swebp__cat5[] = { 180, 157, 141, 134, 130, 0 };
static const simplewebp_u8 swebp__cat6[] = { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0 };
static const simplewebp_u8* const swebp__cat3456[] = {
	swebp__cat3,
	swebp__cat4,
	swebp__cat5,
	swebp__cat6
};
static const simplewebp_u8 swebp__zigzag[16] = {
	0, 1, 4, 8,
	5, 2, 3, 6, 
	9, 12, 13, 10,
	7, 11, 14, 15
};

static simplewebp_i32 swebp__get_large_value(struct swebp__bdec *br, const simplewebp_u8 *p)
{
	simplewebp_i32 v;

	if (!swebp__bitread_getbit(br, p[3]))
	{
		if (!swebp__bitread_getbit(br, p[4]))
			v = 2;
		else
			v = 3 + swebp__bitread_getbit(br, p[5]);
	}
	else
	{
		if (!swebp__bitread_getbit(br, p[6]))
		{
			if (!swebp__bitread_getbit(br, p[7]))
				v = 5 + swebp__bitread_getbit(br, 159);
			else
			{
				v = 7 + 2 * swebp__bitread_getbit(br, 165);
				v += swebp__bitread_getbit(br, 145);
			}
		}
		else
		{
			const simplewebp_u8 *tab;
			simplewebp_i32 bit1, bit0, cat;

			bit1 = swebp__bitread_getbit(br, p[8]);
			bit0 = swebp__bitread_getbit(br, p[9 + bit1]);
			cat = 2 * bit1 + bit0;
			v = 0;

			for (tab = swebp__cat3456[cat]; *tab; ++tab)
				v += v + swebp__bitread_getbit(br, *tab);

			v += 3 + (8 << cat);
		}
	}

	return v;
}

static simplewebp_i32 swebp__get_coeffs(
	struct swebp__bdec *br,
	const struct swebp__bandprobas *prob[],
	simplewebp_i32 ctx,
	const swebp__quant_t dq,
	simplewebp_i32 n,
	simplewebp_i16 *out
)
{
	const simplewebp_u8 *p = prob[n]->probas[ctx];

	for (; n < 16; n++)
	{
		if (!swebp__bitread_getbit(br, p[0]))
			return n;

		while (!swebp__bitread_getbit(br, p[1]))
		{
			p = prob[++n]->probas[0];
			if (n == 16)
				return 1;
		}

		{
			const swebp__probarray *p_ctx;
			simplewebp_i32 v;

			p_ctx = &prob[n + 1]->probas[0];

			if (!swebp__bitread_getbit(br, p[2]))
			{
				v = 1;
				p = p_ctx[1];
			}
			else
			{
				v = swebp__get_large_value(br, p);
				p = p_ctx[2];
			}

			out[swebp__zigzag[n]] = swebp__bitread_getsigned(br, v) * dq[n > 0];
		}
	}

	return 16;
}

static simplewebp_u32 swebp__nz_code_bits(simplewebp_u32 nz_coeffs, simplewebp_i32 nz, simplewebp_i8 dc_nz)
{
	nz_coeffs <<= 2;
	nz_coeffs |= (nz > 3) ? 3 : (nz > 1) ? 2 : dc_nz;
	return nz_coeffs;
}

static simplewebp_i32 swebp__vp8_decode_macroblock(struct swebp__vp8 *vp8d, struct swebp__bdec *token_br)
{
	struct swebp__mblock *left, *mb;
	struct swebp__mblockdata *block;
	simplewebp_u8 skip;

	left = vp8d->mb_info - 1;
	mb = vp8d->mb_info + vp8d->mb_x;
	block = vp8d->mb_data + vp8d->mb_x;
	skip = vp8d->use_skip_proba ? block->skip : 0;

	if (!skip)
	{
		/* Parse residuals */
		const struct swebp__bandprobas * (*bands)[17], **ac_proba;
		struct swebp__quantmat *q;
		simplewebp_i16 *dst;
		struct swebp__mblock *left_mb;
		simplewebp_u8 tnz, lnz;
		simplewebp_u32 non0_y, non0_uv, out_t_nz, out_l_nz;
		simplewebp_i32 x, y, ch, first;

		bands = vp8d->proba.bands_ptr;
		q = &vp8d->dqm[block->segment];
		dst = block->coeffs;
		left_mb = vp8d->mb_info - 1;
		non0_y = non0_uv = 0;

		memset(dst, 0, 384 * sizeof(simplewebp_i16));
		if (!block->is_i4x4)
		{
			simplewebp_i16 dc[16];
			simplewebp_i32 ctx, nz;

			memset(dc, 0, sizeof(dc));
			ctx = mb->nz_dc + left_mb->nz_dc;
			nz = swebp__get_coeffs(token_br, bands[1], ctx, q->y2_mat, 0, dc);

			mb->nz_dc = left_mb->nz_dc = nz > 0;
			if (nz > 1)
				swebp__transform_wht(dc, dst);
			else
			{
				simplewebp_i32 i, dc0;
				dc0 = (dc[0] + 3) >> 3;
				for (i = 0; i < 16; i++)
					dst[i * 16] = dc0;
			}

			first = 1;
			ac_proba = bands[0];
		}
		else
		{
			first = 0;
			ac_proba = bands[3];
		}

		tnz = mb->nz & 0xf;
		lnz = left_mb->nz & 0xf;

		for (y = 0; y < 4; y++)
		{
			simplewebp_i32 l;
			simplewebp_u32 nz_coeffs;

			l = lnz & 1;
			nz_coeffs = 0;

			for (x = 0; x < 4; x++)
			{
				simplewebp_i32 ctx, nz;

				ctx = l + (tnz & 1);
				nz = swebp__get_coeffs(token_br, ac_proba, ctx, q->y1_mat, first, dst);
				l = nz > first;
				tnz = (tnz >> 1) | (l << 7);
				nz_coeffs = swebp__nz_code_bits(nz_coeffs, nz, dst[0] != 0);
				dst += 16;
			}

			tnz >>= 4;
			lnz = (lnz >> 1) | (l << 7);
			non0_y = (non0_y << 8) | nz_coeffs;
		}

		out_t_nz = tnz;
		out_l_nz = lnz >> 4;

		for (ch = 0; ch < 4; ch += 2)
		{
			simplewebp_u32 nz_coeffs = 0;

			tnz = mb->nz >> (4 + ch);
			lnz = left_mb->nz >> (4 + ch);

			for (y = 0; y < 2; y++)
			{
				simplewebp_i32 l = lnz & 1;

				for (x = 0; x < 2; x++)
				{
					simplewebp_i32 ctx, nz;

					ctx = l + (tnz & 1);
					nz = swebp__get_coeffs(token_br, bands[2], ctx, q->uv_mat, 0, dst);
					l = nz > 0;
					tnz = (tnz >> 1) | (l << 3);
					nz_coeffs = swebp__nz_code_bits(nz_coeffs, nz, dst[0] != 0);
					dst += 16;
				}

				tnz >>= 2;
				lnz = (lnz >> 1) | (l << 5);
			}

			non0_uv |= nz_coeffs << (4 * ch);
			out_t_nz |= tnz << (4 + ch);
			out_l_nz |= (lnz & 0xf0) << ch;
		}

		mb->nz = out_t_nz;
		left_mb->nz = out_l_nz;

		block->nonzero_y = non0_y;
		block->nonzero_uv = non0_uv;
		block->dither = (non0_uv & 0xaaaa) ? 0 : q->dither;

		skip = !(non0_y | non0_uv);
	}
	else
	{
		left->nz = mb->nz = 0;
		if (!block->is_i4x4)
			left->nz_dc = mb->nz_dc = 0;
		
		block->nonzero_y = block->nonzero_uv = block->dither = 0;
	}

	if (vp8d->filter_type > 0)
	{
		struct swebp__finfo *const finfo = vp8d->f_info + vp8d->mb_x;
		*finfo = vp8d->fstrengths[block->segment][block->is_i4x4];
		finfo->inner |= !skip;
	}

	return !token_br->eof;
}

static void swebp__do_transform(simplewebp_u32 bits, const simplewebp_i16 *src, simplewebp_u8 *dst)
{
	switch (bits >> 30)
	{
		case 3:
			swebp__transform(src, dst, 0);
			break;
		case 2:
			swebp__transform_ac3(src, dst);
			break;
		case 1:
			swebp__transform_dc(src, dst);
			break;
		default:
			break;
	}
}

static simplewebp_i32 swebp__check_mode(simplewebp_i32 mb_x, simplewebp_i32 mb_y, simplewebp_i32 mode)
{
	if (mode == 0) {
		if (mb_x == 0)
			return (mb_y == 0) ? 6 : 5;
		else
			return (mb_y == 0) ? 4 : 0;
	}

	return mode;
}

static void swebp__do_transform_uv(simplewebp_u32 bits, const simplewebp_i16 *src, simplewebp_u8 *dst)
{
	if (bits & 0xff)
	{
		if (bits & 0xaa)
			swebp__transform_uv(src, dst);
		else
			swebp__transform_dcuv(src, dst);
	}
}

static simplewebp_i32 swebp__multhi(simplewebp_i32 v, simplewebp_i32 coeff)
{
	return (v * coeff) >> 8;
}

static simplewebp_u8 swebp__yuv2rgb_clip8(simplewebp_i32 v)
{
	return ((v & ~16383) == 0) ? ((simplewebp_u8) (v >> 6)) : (v < 0) ? 0 : 255;
}

static void swebp__yuv2rgb_plain(simplewebp_u8 y, simplewebp_u8 u, simplewebp_u8 v, struct swebp__pixel *rgb)
{
	simplewebp_i32 yhi = swebp__multhi(y, 19077);

	rgb->r = swebp__yuv2rgb_clip8(yhi + swebp__multhi(v, 26149) - 14234);
	rgb->g = swebp__yuv2rgb_clip8(yhi - swebp__multhi(u, 6419) - swebp__multhi(v, 13320) + 8708);
	rgb->b = swebp__yuv2rgb_clip8(yhi + swebp__multhi(u, 33050) - 17685);
}

/* r = top-left, g = top-right, b = bottom-left, a = bottom-right */
static struct swebp__pixel swebp__do_upsample_center(
	const simplewebp_u8 *vtop,
	const simplewebp_u8 *vmid,
	const simplewebp_u8 *vbot,
	size_t xm1,
	size_t x,
	size_t xp1
)
{
	/*
	Consider these layout:
	0  1  2  3  4  5
	y  y  y  y  y  y  0
	 t1    t2    t3
	y  y  y  y  y  y  1

	y  y  y  y  y  y  2
	 u1    u2    u3
	y  y  y  y  y  y  3

	y  y  y  y  y  y  4
	 v1    v2    v3
	y  y  y  y  y  y  5

	t, u, v are the chroma (CbCr/UV).

	We want to place all of them in y, but note that t, u, and v are centered.
	So (note this is XY order, not YX)
	y22 = 9 * u2 + 3 * u1 + 3 * t2 + t1
	y32 = 9 * u2 + 3 * u3 + 3 * t2 + t3
	y23 = 9 * u2 + 3 * u1 + 3 * v2 + v1
	y33 = 9 * u2 + 3 * u3 + 3 * v2 + v3

	Now let's generalize it: assume x, y span from 0 to chroma dimensions:
	y[x * 2 +0, y * 2 +0] = 9 * uv[x, y] + 3 * uv[x-1, y] + 3 * uv[x, y-1] + uv[x-1, y-1]
	y[x * 2 +1, y * 2 +0] = 9 * uv[x, y] + 3 * uv[x+1, y] + 3 * uv[x, y-1] + uv[x+1, y-1]
	y[x * 2 +0, y * 2 +1] = 9 * uv[x, y] + 3 * uv[x-1, y] + 3 * uv[x, y+1] + uv[x-1, y+1]
	y[x * 2 +1, y * 2 +1] = 9 * uv[x, y] + 3 * uv[x+1, y] + 3 * uv[x, y+1] + uv[x+1, y+1]

	For rounding, add + 8 to each, then divide the result by 16, so: (formula_above + 8) / 16
	*/

	struct swebp__pixel out;

	out.r = (simplewebp_u8) ((9u * vmid[x] + 3u * vmid[xm1] + 3u * vtop[x] + vtop[xm1] + 8u) / 16u);
	out.g = (simplewebp_u8) ((9u * vmid[x] + 3u * vmid[xp1] + 3u * vtop[x] + vtop[xp1] + 8u) / 16u);
	out.b = (simplewebp_u8) ((9u * vmid[x] + 3u * vmid[xm1] + 3u * vbot[x] + vbot[xm1] + 8u) / 16u);
	out.a = (simplewebp_u8) ((9u * vmid[x] + 3u * vmid[xp1] + 3u * vbot[x] + vbot[xp1] + 8u) / 16u);
	return out;
}

/* This is bilinear interpolation with center chroma. */
/* See https://stackoverflow.com/a/43784809 */
static void swebp__upsample_chroma(
	const simplewebp_u8 *u,
	const simplewebp_u8 *v,
	struct swebp__chroma *dst,
	size_t w,
	size_t h
)
{
	size_t y, fw;
	fw = w * 2;

	for (y = 0; y < h; y++)
	{
		size_t prev_y, next_y, x;
		const simplewebp_u8 *uline;
		const simplewebp_u8 *vline;
		const simplewebp_u8 *ulineprev;
		const simplewebp_u8 *vlineprev;
		const simplewebp_u8 *ulinenext;
		const simplewebp_u8 *vlinenext;

		prev_y = y == 0 ? 0 : (y - 1);
		next_y = y == (h - 1) ? y : (y + 1);
		ulineprev = u + prev_y * w;
		vlineprev = v + prev_y * w;
		ulinenext = u + next_y * w;
		vlinenext = v + next_y * w;
		uline = u + y * w;
		vline = v + y * w;

		for (x = 0; x < w; x++)
		{
			size_t prev_x, next_x, i00, i10, i01, i11;
			struct swebp__pixel uvalue, vvalue;

			prev_x = x == 0 ? x : (x - 1);
			next_x = x == (w - 1) ? x : (x + 1);
			uvalue = swebp__do_upsample_center(ulineprev, uline, ulinenext, prev_x, x, next_x);
			vvalue = swebp__do_upsample_center(vlineprev, vline, vlinenext, prev_x, x, next_x);
			i00 = (y * 2) * fw + (x * 2);
			i10 = (y * 2) * fw + (x * 2 + 1);
			i01 = (y * 2 + 1) * fw + (x * 2);
			i11 = (y * 2 + 1) * fw + (x * 2 + 1);

			dst[i00].u = uvalue.r;
			dst[i00].v = vvalue.r;
			dst[i10].u = uvalue.g;
			dst[i10].v = vvalue.g;
			dst[i01].u = uvalue.b;
			dst[i01].v = vvalue.b;
			dst[i11].u = uvalue.a;
			dst[i11].v = vvalue.a;
		}
	}
}

static void swebp__yuva2rgba(
	const simplewebp_u8 *yp,
	const struct swebp__chroma *uv,
	const simplewebp_u8 *a,
	size_t w,
	size_t h,
	struct swebp__pixel *rgba
)
{
	size_t y, x, uvw;

	uvw = ((w + 1) / 2) * 2;

	for (y = 0; y < h; y++)
	{
		for (x = 0; x < w; x++)
		{
			swebp__yuv2rgb_plain(yp[y * w + x], uv[y * uvw + x].u, uv[y * uvw + x].v, &rgba[y * w + x]);
			rgba[y * w + x].a = a[y * w + x];
		}
	}
}

static simplewebp_error swebp__vp8_process_row(struct swebp__vp8 *vp8d, struct swebp__yuvdst *destination)
{
	simplewebp_i32 filter_row;

	filter_row = vp8d->filter_type > 0 && vp8d->mb_y >= vp8d->tl_mb_y && vp8d->mb_y <= vp8d->br_mb_y;

	/* Reconstruct row */
	{
		simplewebp_i32 j, mb_x, mb_y;
		simplewebp_u8 *y_dst, *u_dst, *v_dst;

		mb_y = vp8d->mb_y;
		y_dst = vp8d->yuv_b + 40;
		u_dst = vp8d->yuv_b + 584;
		v_dst = vp8d->yuv_b + 600;

		for (j = 0; j < 16; j++)
			y_dst[j * 32 - 1] = 129;
		for (j = 0; j < 8; j++)
		{
			u_dst[j * 32 - 1] = 129;
			v_dst[j * 32 - 1] = 129;
		}

		if (mb_y > 0)
			y_dst[-33] = u_dst[-33] = v_dst[-33] = 129;
		else
		{
			memset(y_dst - 33, 127, 21);
			memset(u_dst - 33, 127, 9);
			memset(v_dst - 33, 127, 9);
		}

		for (mb_x = 0; mb_x < vp8d->mb_w; mb_x++)
		{
			struct swebp__mblockdata *const block = vp8d->mb_data + mb_x;

			if (mb_x > 0)
			{
				for (j = -1; j < 16; j++)
					memcpy(y_dst + j * 32 - 4, y_dst + j * 32 + 12, 4);
				for (j = -1; j < 8; j++)
				{
					memcpy(u_dst + j * 32 - 4, u_dst + j * 32 + 4, 4);
					memcpy(v_dst + j * 32 - 4, v_dst + j * 32 + 4, 4);
				}
			}

			{
				struct swebp__topsmp *top_yuv;
				const simplewebp_i16 *coeffs;
				simplewebp_u32 bits;
				simplewebp_i32 n;

				top_yuv = vp8d->yuv_t + mb_x;
				coeffs = block->coeffs;
				bits = block->nonzero_y;

				if (mb_y > 0)
				{
					memcpy(y_dst - 32, top_yuv[0].y, 16);
					memcpy(u_dst - 32, top_yuv[0].u, 8);
					memcpy(v_dst - 32, top_yuv[0].v, 8);
				}

				if (block->is_i4x4)
				{
					simplewebp_u32 *const top_right = (simplewebp_u32 *) (y_dst - 32 + 16);

					if (mb_y > 0)
					{
						if (mb_x >= vp8d->mb_w - 1)
							memset(top_right, top_yuv[0].y[15], 4);
						else
							memcpy(top_right, top_yuv[1].y, 4);
					}

					top_right[32] = top_right[64] = top_right[96] = top_right[0];

					for (n = 0; n < 16; n++, bits <<= 2)
					{
						/* kScan[n] = (n & 3) * 4 + (n >> 2) * 128 */
						simplewebp_u8 *const dst = y_dst + ((n & 3) * 4 + (n >> 2) * 128);
						swebp__predluma4(block->imodes[n], dst);
						swebp__do_transform(bits, coeffs + n * 16, dst);
					}
				}
				else
				{
					const simplewebp_i32 pred_func = swebp__check_mode(mb_x, mb_y, block->imodes[0]);
					swebp__predluma16(pred_func, y_dst);

					if (bits)
					{
						for (n = 0; n < 16; n++, bits <<= 2)
						{
							simplewebp_u8 *const dst = y_dst + ((n & 3) * 4 + (n >> 2) * 128);
							swebp__do_transform(bits, coeffs + n * 16, dst);
						}
					}
				}

				{
					/* Chroma */
					simplewebp_u32 bits_uv;
					simplewebp_i32 pred_func;

					bits_uv = block->nonzero_uv;
					pred_func = swebp__check_mode(mb_x, mb_y, block->uvmode);

					swebp__predchroma8(pred_func, u_dst);
					swebp__predchroma8(pred_func, v_dst);
					swebp__do_transform_uv(bits_uv >> 0, coeffs + 16 * 16, u_dst);
					swebp__do_transform_uv(bits_uv >> 8, coeffs + 20 * 16, v_dst);
				}

				if (mb_y < vp8d->mb_h - 1)
				{
					memcpy(top_yuv[0].y, y_dst + 15 * 32, 16);
					memcpy(top_yuv[0].u, u_dst + 7 * 32, 8);
					memcpy(top_yuv[0].v, v_dst + 7 * 32, 8);
				}
			}

			/* Transfer reconstructed samples */
			{
				simplewebp_u8 *y_out, *u_out, *v_out;

				y_out = vp8d->cache_y + mb_x * 16;
				u_out = vp8d->cache_u + mb_x * 8;
				v_out = vp8d->cache_v + mb_x * 8;

				for (j = 0; j < 16; j++)
					memcpy(y_out + j * vp8d->cache_y_stride, y_dst + j * 32, 16);
				for (j = 0; j < 8; j++)
				{
					memcpy(u_out + j * vp8d->cache_uv_stride, u_dst + j * 32, 8);
					memcpy(v_out + j * vp8d->cache_uv_stride, v_dst + j * 32, 8);
				}
			}
		}
	}

	/* Finish row */
	{
		simplewebp_i32 extra_y_rows, ysize, uvsize, mb_y;
		simplewebp_u8 *ydst, *udst, *vdst, is_first_row, is_last_row;

		extra_y_rows = swebp__fextrarows[vp8d->filter_type];
		ysize = extra_y_rows * vp8d->cache_y_stride;
		uvsize = (extra_y_rows / 2) * vp8d->cache_uv_stride;
		ydst = vp8d->cache_y - ysize;
		udst = vp8d->cache_u - uvsize;
		vdst = vp8d->cache_v - uvsize;
		mb_y = vp8d->mb_y;
		is_first_row = mb_y == 0;
		is_last_row = mb_y >= vp8d->br_mb_y - 1;

		if (filter_row)
		{
			/* Filter row */
			simplewebp_i32 mb_x;

			for (mb_x = vp8d->tl_mb_x; mb_x < vp8d->br_mb_x; mb_x++)
			{
				simplewebp_i32 y_bps, ilevel, limit;
				struct swebp__finfo *f_info;
				simplewebp_u8 *y_dst;

				f_info = vp8d->f_info + mb_x;
				limit = f_info->limit;

				if (limit > 0)
				{
					ilevel = f_info->ilevel;
					y_bps = vp8d->cache_y_stride;
					y_dst = vp8d->cache_y + mb_x * 16;

					if (vp8d->filter_type == 1)
					{
						/* Simple filter */
						if (mb_x > 0)
							swebp__simple_hfilter16(y_dst, y_bps, limit + 4);
						if (f_info->inner)
							swebp__simple_hfilter16_i(y_dst, y_bps, limit);
						if (mb_y > 0)
							swebp__simple_vfilter16(y_dst, y_bps, limit + 4);
						if (f_info->inner)
							swebp__simple_vfilter16_i(y_dst, y_bps, limit);
					}
					else
					{
						/* Complex filter */
						simplewebp_i32 uv_bps, hev_thresh;
						simplewebp_u8 *u_dst, *v_dst;

						uv_bps = vp8d->cache_uv_stride;
						hev_thresh = f_info->hev_thresh;
						u_dst = vp8d->cache_u + mb_x * 8;
						v_dst = vp8d->cache_v + mb_x * 8;

						if (mb_x > 0)
						{
							swebp__hfilter16(y_dst, y_bps, limit + 4, ilevel, hev_thresh);
							swebp__hfilter8(u_dst, v_dst, uv_bps, limit + 4, ilevel, hev_thresh);
						}
						if (f_info->inner)
						{
							swebp__hfilter16_i(y_dst, y_bps, limit, ilevel, hev_thresh);
							swebp__hfilter8_i(u_dst, v_dst, uv_bps, limit, ilevel, hev_thresh);
						}
						if (mb_y > 0)
						{
							swebp__vfilter16(y_dst, y_bps, limit + 4, ilevel, hev_thresh);
							swebp__vfilter8(u_dst, v_dst, uv_bps, limit + 4, ilevel, hev_thresh);
						}
						if (f_info->inner)
						{
							swebp__vfilter16_i(y_dst, y_bps, limit, ilevel, hev_thresh);
							swebp__vfilter8_i(u_dst, v_dst, uv_bps, limit, ilevel, hev_thresh);
						}
					}
				}
			}
		}

		{
			/* Write YUV */
			simplewebp_u8 *y_out, *u_out, *v_out;
			simplewebp_i32 y_start, y_end;

			y_start = mb_y * 16;
			y_end = (mb_y + 1) * 16;
			if (!is_first_row)
			{
				y_start -= extra_y_rows;
				y_out = ydst;
				u_out = udst;
				v_out = vdst;
			}
			else
			{
				y_out = vp8d->cache_y;
				u_out = vp8d->cache_u;
				v_out = vp8d->cache_v;
			}

			if (!is_last_row)
				y_end -= extra_y_rows;
			if (y_end > vp8d->picture_header.height)
				y_end = vp8d->picture_header.height;

			{
				/* Copy YUV buffer */
				simplewebp_i32 row, iwidth, iwidth2, uv_start, uv_end;

				iwidth = vp8d->picture_header.width;
				iwidth2 = (iwidth + 1) / 2;
				uv_start = y_start / 2;
				uv_end = (y_end + 1) / 2;

				/* Copy Y and A buffer */
				for (row = y_start; row < y_end; row++)
					memcpy(destination->y + row * iwidth, y_out + (row - y_start) * vp8d->cache_y_stride, iwidth);

				/* Copy U and V buffer */
				for (row = uv_start; row < uv_end; row++)
				{
					memcpy(destination->u + row * iwidth2, u_out + (row - uv_start) * vp8d->cache_uv_stride, iwidth2);
					memcpy(destination->v + row * iwidth2, v_out + (row - uv_start) * vp8d->cache_uv_stride, iwidth2);
				}
			}
		}
		
		/* Rotate top samples */
		if (!is_last_row)
		{
			memcpy(vp8d->cache_y - ysize, ydst + 16 * vp8d->cache_y_stride, ysize);
			memcpy(vp8d->cache_u - uvsize, udst + 8 * vp8d->cache_uv_stride, uvsize);
			memcpy(vp8d->cache_v - uvsize, vdst + 8 * vp8d->cache_uv_stride, uvsize);
		}
	}

	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_error swebp__vp8_parse_frame(struct swebp__vp8 *vp8d, struct swebp__yuvdst *destination)
{
	simplewebp_error err;

	for (vp8d->mb_y = 0; vp8d->mb_y < vp8d->br_mb_y; vp8d->mb_y++)
	{
		struct swebp__bdec *token_br = &vp8d->parts[vp8d->mb_y & vp8d->nparts_minus_1];

		if (!swebp__vp8_parse_intra_row(vp8d))
			return SIMPLEWEBP_CORRUPT_ERROR;

		for (vp8d->mb_x = 0; vp8d->mb_x < vp8d->mb_w; vp8d->mb_x++)
		{
			if (!swebp__vp8_decode_macroblock(vp8d, token_br))
				return SIMPLEWEBP_CORRUPT_ERROR;
		}

		/* Prepare for next scanline and reconstruct it */
		swebp__vp8_init_scanline(vp8d);
		err = swebp__vp8_process_row(vp8d, destination);

		if (err != SIMPLEWEBP_NO_ERROR)
			return err;
	}

	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_error swebp__decode_lossy(simplewebp *simplewebp, struct swebp__yuvdst *destination, void *settings)
{
	(void)settings;
	simplewebp_input input;
	size_t vp8size;
	simplewebp_u8 *vp8buffer, *decoder_mem;
	struct swebp__vp8 *vp8d;
	simplewebp_error err;

	vp8d = &simplewebp->decoder.vp8;
	input = simplewebp->vp8_input;

	if (!swebp__seek(0, &input))
		return SIMPLEWEBP_IO_ERROR;

	vp8size = swebp__proxy_size(input.userdata);
	/* Sanity check */
	if (vp8d->frame_header.partition_length > vp8size)
		return SIMPLEWEBP_CORRUPT_ERROR;

	/* Read all VP8 chunk */
	vp8buffer = (simplewebp_u8 *) swebp__alloc(simplewebp, vp8size);
	if (vp8buffer == NULL)
		return SIMPLEWEBP_ALLOC_ERROR;

	if (!swebp__read2(vp8size, vp8buffer, &input))
	{
		swebp__dealloc(simplewebp, vp8buffer);
		return SIMPLEWEBP_CORRUPT_ERROR;
	}

	/* Let's skip already-read stuff at swebp__load_lossy */
	if (!swebp__seek(10, &input))
	{
		swebp__dealloc(simplewebp, vp8buffer);
		return SIMPLEWEBP_CORRUPT_ERROR;
	}

	err = swebp__load_vp8_header(&simplewebp->decoder.vp8, vp8buffer + 10, vp8size - 10);
	if (err != SIMPLEWEBP_NO_ERROR)
	{
		swebp__dealloc(simplewebp, vp8buffer);
		return err;
	}

	/* Enter critical */
	swebp__vp8_enter_critical(&simplewebp->decoder.vp8);
	decoder_mem = swebp__vp8_alloc_memory(vp8d, &simplewebp->allocator);

	if (decoder_mem == NULL)
	{
		swebp__dealloc(simplewebp, vp8buffer);
		return SIMPLEWEBP_ALLOC_ERROR;
	}

	err = swebp__vp8_parse_frame(vp8d, destination);
	if (err != SIMPLEWEBP_NO_ERROR)
	{
		swebp__dealloc(simplewebp, decoder_mem);
		swebp__dealloc(simplewebp, vp8buffer);
		return err;
	}

	/* Exit critical and cleanup */
	swebp__dealloc(simplewebp, decoder_mem);
	swebp__dealloc(simplewebp, vp8buffer);
	vp8d->mem = NULL;
	vp8d->mem_size = 0;
	memset(&vp8d->br, 0, sizeof(struct swebp__bdec));
	vp8d->ready = 0;

	return swebp__alpha_decode(simplewebp, destination->a);
}

struct swebp__vp8l_code_node
{
	simplewebp_u16 child[2];
};

struct swebp__vp8l_code
{
	simplewebp_u16 size;
	/* Constructing root: symbol[0] | (symbol[1] << 8) */
	simplewebp_u8 symbol[2];
	struct swebp__vp8l_code_node *tree;
};

struct swebp__vp8l_group
{
	struct swebp__vp8l_code code[5];
};

const simplewebp_u16 swebp__vp8l_literals_count = 256;
const simplewebp_u16 swebp__vp8l_distances_count = 40;
const simplewebp_u16 swebp__vp8l_litlen_count = 256 + 24;
const size_t swebp__vp8l_offset_count = 120;

static const simplewebp_u8 swebp__vp8l_lencode_order[19] = {
	17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};

static const simplewebp_i8 swebp__vp8l_offsets[120][2] = {
	{ 0, 1}, { 1, 0 }, { 1, 1}, {-1, 1}, { 0, 2}, { 2, 0},
	{ 1, 2}, {-1, 2 }, { 2, 1}, {-2, 1}, { 2, 2}, {-2, 2},
	{ 0, 3}, { 3, 0 }, { 1, 3}, {-1, 3}, { 3, 1}, {-3, 1},
	{ 2, 3}, {-2, 3 }, { 3, 2}, {-3, 2}, { 0, 4}, { 4, 0},
	{ 1, 4}, {-1, 4 }, { 4, 1}, {-4, 1}, { 3, 3}, {-3, 3},
	{ 2, 4}, {-2, 4 }, { 4, 2}, {-4, 2}, { 0, 5}, { 3, 4},
	{-3, 4}, { 4, 3 }, {-4, 3}, { 5, 0}, { 1, 5}, {-1, 5},
	{ 5, 1}, {-5, 1 }, { 2, 5}, {-2, 5}, { 5, 2}, {-5, 2},
	{ 4, 4}, {-4, 4 }, { 3, 5}, {-3, 5}, { 5, 3}, {-5, 3},
	{ 0, 6}, { 6, 0 }, { 1, 6}, {-1, 6}, { 6, 1}, {-6, 1},
	{ 2, 6}, {-2, 6 }, { 6, 2}, {-6, 2}, { 4, 5}, {-4, 5},
	{ 5, 4}, {-5, 4 }, { 3, 6}, {-3, 6}, { 6, 3}, {-6, 3},
	{ 0, 7}, { 7, 0 }, { 1, 7}, {-1, 7}, { 5, 5}, {-5, 5},
	{ 7, 1}, {-7, 1 }, { 4, 6}, {-4, 6}, { 6, 4}, {-6, 4},
	{ 2, 7}, {-2, 7 }, { 7, 2}, {-7, 2}, { 3, 7}, {-3, 7},
	{ 7, 3}, {-7, 3 }, { 5, 6}, {-5, 6}, { 6, 5}, {-6, 5},
	{ 8, 0}, { 4, 7 }, {-4, 7}, { 7, 4}, {-7, 4}, { 8, 1},
	{ 8, 2}, { 6, 6 }, {-6, 6}, { 8, 3}, { 5, 7}, {-5, 7},
	{ 7, 5}, {-7, 5 }, { 8, 4}, { 6, 7}, {-6, 7}, { 7, 6},
	{-7, 6}, { 8, 5 }, { 7, 7}, {-7, 7}, { 8, 6}, { 8, 7},
};

static void swebp__vp8l_insert_code(
	struct swebp__vp8l_code_node *tree,
	simplewebp_u16 *root,
	simplewebp_u16 size,
	simplewebp_u16 symbol,
	simplewebp_u16 word,
	simplewebp_u16 *used,
	simplewebp_u8 length
)
{
	while (length)
	{
		simplewebp_u16 index;
		struct swebp__vp8l_code_node *node;

		index = *root;
		if (index < size)
		{
			*root = (*used)++;
			node = tree + (*root - size);
		}
		else
			node = tree + (index - size);

		root = &node->child[(word >> --length) & 1];
	}

	*root = symbol;
}

static simplewebp_error swebp__vp8l_canonical_code(
	simplewebp *simplewebp,
	simplewebp_u8 *lengths,
	simplewebp_u16 size,
	struct swebp__vp8l_code *code,
	struct swebp__vp8l_code_node *treemem
)
{
	simplewebp_u16 base[16], real_size, current_base, root, used;
	struct swebp__vp8l_code_node *tree;
	size_t i;

	memset(base, 0, sizeof(base));
	real_size = 0;
	current_base = 0;
	root = 0;
	used = size;
	tree = NULL;

	for (i = 0; i < size; i++)
	{
		simplewebp_u8 length = lengths[i];
		if (length > 0)
		{
			base[length - 1]++;
			real_size++;
		}
	}
	real_size--;

	for (i = 0; i < 16; i++)
	{
		simplewebp_u16 c = base[i];
		base[i] = current_base;
		current_base = (current_base + c) << 1;
	}

	if (treemem)
		tree = treemem;
	else
	{
		tree = (struct swebp__vp8l_code_node*) swebp__alloc(
			simplewebp,
			sizeof(struct swebp__vp8l_code_node) * real_size
		);
		if (!tree)
			return SIMPLEWEBP_ALLOC_ERROR;
	}

	for (i = 0; i < real_size; i++)
	{
		tree[i].child[0] = 0;
		tree[i].child[1] = 0;
	}

	for (i = 0; i < size; i++)
	{
		simplewebp_u8 l = lengths[i];
		if (l > 0)
			swebp__vp8l_insert_code(tree, &root, size, (simplewebp_u16) i, base[l - 1]++, &used, l);
	}

	code->size = size;
	code->tree = tree;
	code->symbol[0] = root & 0xFF;
	code->symbol[1] = (root >> 8) & 0xFF;

	return SIMPLEWEBP_NO_ERROR;
}

static simplewebp_u16 swebp__vp8l_read_code(struct swebp__vp8l_bdec *br, struct swebp__vp8l_code *code)
{
	if (code->tree)
	{
		simplewebp_u16 index = code->symbol[0] | (((simplewebp_u16) code->symbol[1]) << 8);

		while (index >= code->size)
			index = code->tree[index - code->size].child[swebp__vp8l_bitread_read(br, 1)];

		return index;
	}
	else
		return code->symbol[code->size < 2 ? 0 : swebp__vp8l_bitread_read(br, 1)];
}

static simplewebp_error swebp__vp8l_decode_code_complex(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	struct swebp__vp8l_code *code,
	simplewebp_u16 size
)
{
	size_t i;
	simplewebp_u8 lencode_lengths[19], lencode_read;
	simplewebp_u8 lengths[256 + 24 + 2048];
	simplewebp_u16 limit, count, p;
	simplewebp_error err;
	struct swebp__vp8l_code_node lencode_treemem[18];
	struct swebp__vp8l_code lc;

	lencode_read = (simplewebp_u8) swebp__vp8l_bitread_read(br, 4) + 4;
	for (i = 0; i < 19; i++)
		lencode_lengths[swebp__vp8l_lencode_order[i]] = i < lencode_read ? swebp__vp8l_bitread_read(br, 3) : 0;

	if (swebp__vp8l_bitread_read(br, 1))
		limit = swebp__vp8l_bitread_read(br, swebp__vp8l_bitread_read(br, 3) * 2 + 2) + 2;
	else
		limit = size;
	
	if (br->eos)
		return SIMPLEWEBP_IO_ERROR;

	err = swebp__vp8l_canonical_code(simplewebp, lencode_lengths, 19, &lc, lencode_treemem);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;

	count = 0;
	p = 8;

	for (i = 0; count < size && i < limit; i++)
	{
		simplewebp_u16 s, c;

		s = swebp__vp8l_read_code(br, &lc);
		c = 0;

		if (br->eos)
		{
			swebp__dealloc(simplewebp, lc.tree);
			return SIMPLEWEBP_IO_ERROR;
		}

		switch (s)
		{
			default:
			case 0:
				lengths[count++] = (simplewebp_u8) s;
				continue;
				/* fall through */
			case 1:
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
			case 8:
			case 9:
			case 10:
			case 11:
			case 12:
			case 13:
			case 14:
			case 15:
				lengths[count++] = (simplewebp_u8) (p = s);
				continue;
			case 16:
				s = 3 + swebp__vp8l_bitread_read(br, 2);
				c = p;
				break;
			case 17:
				s = 3 + swebp__vp8l_bitread_read(br, 3);
				c = 0;
				break;
			case 18:
				s = 11 + swebp__vp8l_bitread_read(br, 7);
				c = 0;
				break;
		}
		
		if (br->eos)
		{
			swebp__dealloc(simplewebp, lc.tree);
			return SIMPLEWEBP_IO_ERROR;
		}

		while (s--)
			lengths[count++] = (simplewebp_u8) c;
	}
	
	err = swebp__vp8l_canonical_code(simplewebp, lengths, count, code, NULL);
	return err;
}

static simplewebp_error swebp__vp8l_decode_code(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	struct swebp__vp8l_code *code,
	simplewebp_u16 size
)
{
	if (swebp__vp8l_bitread_read(br, 1))
	{
		simplewebp_bool two_symbols = (simplewebp_bool) swebp__vp8l_bitread_read(br, 1);
		code->tree = NULL;
		code->size = two_symbols + 1;
		code->symbol[0] = swebp__vp8l_bitread_read(br, 1 + swebp__vp8l_bitread_read(br, 1) * 7);
		code->symbol[1] = two_symbols ? swebp__vp8l_bitread_read(br, 8) : 0;
		return br->eos ? SIMPLEWEBP_IO_ERROR : SIMPLEWEBP_NO_ERROR;
	}
	else
		return swebp__vp8l_decode_code_complex(simplewebp, br, code, size);
}

static simplewebp_error swebp__vp8l_decode_group(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	struct swebp__vp8l_group *group,
	simplewebp_u8 bits
)
{
	size_t i;
	simplewebp_u16 sizes[5];
	sizes[0] = swebp__vp8l_litlen_count + (bits ? 1 << bits : 0);
	sizes[1] = sizes[2] = sizes[3] = swebp__vp8l_literals_count;
	sizes[4] = swebp__vp8l_distances_count;

	for (i = 0; i < 5; i++)
	{
		simplewebp_error err;
		struct swebp__vp8l_code *code;

		code = group->code + i;
		err = swebp__vp8l_decode_code(simplewebp, br, code, sizes[i]);

		if (err != SIMPLEWEBP_NO_ERROR)
		{
			/* Free all tree pointers */
			for (; i != (size_t) -1; i--)
			{
				code = group->code + i;
				swebp__dealloc(simplewebp, code->tree);
				code->tree = NULL;
			}

			return err;
		}
	}

	return SIMPLEWEBP_NO_ERROR;
}

static size_t swebp__hash_color(simplewebp_u8 bits, struct swebp__pixel c)
{
	size_t value = (c.a << 24) | (c.r << 16) | (c.g << 8) | c.b;
	return ((0x1e35a7bd * value) & 0xFFFFFFFF) >> (32 - bits);
}

static void swebp__vp8l_put_cache(simplewebp_u8 bits, struct swebp__pixel *ccache, struct swebp__pixel color)
{
	if (bits)
		ccache[swebp__hash_color(bits, color)] = color;
}

static size_t swebp__vp8l_lendst(struct swebp__vp8l_bdec *br, simplewebp_u16 c)
{
	size_t extra;

	if (c < 4)
		return c;

	extra = (c - 2) >> 1;
	return ((2 + (c & 1)) << extra) + swebp__vp8l_bitread_read(br, (int) extra);
}

static simplewebp_error swebp__decode_vp8l_image(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	simplewebp_bool is_main,
	size_t width,
	size_t height,
	struct swebp__pixel **dest
)
{
	simplewebp_u8 ccache_bits, entropy_bits;
	struct swebp__pixel *color_cache, *entropy, *image;
	struct swebp__vp8l_group *groups;
	size_t group_count, entropy_stride, i;
	simplewebp_error err;

	color_cache = NULL;
	group_count = 1;
	entropy_bits = 0;
	entropy_stride = 0;
	entropy = NULL;
	err = SIMPLEWEBP_NO_ERROR;

	if (*dest == NULL)
	{
		*dest = (struct swebp__pixel *) swebp__alloc(simplewebp, width * height * 4);
		if (*dest == NULL)
			return SIMPLEWEBP_ALLOC_ERROR;
	}
	image = *dest;

	ccache_bits = swebp__vp8l_bitread_read(br, 1) ? swebp__vp8l_bitread_read(br, 4) : 0;
	if (br->eos)
		return SIMPLEWEBP_IO_ERROR;
	if (ccache_bits)
	{
		color_cache = (struct swebp__pixel*) swebp__alloc(simplewebp, (1 << ccache_bits) * 4);
		if (!color_cache)
			return SIMPLEWEBP_ALLOC_ERROR;
	}

	if (is_main)
	{
		if (swebp__vp8l_bitread_read(br, 1))
		{
			size_t block_mask, ew, eh, group;
			struct swebp__pixel *ep;

			entropy_bits = swebp__vp8l_bitread_read(br, 3) + 2;
			block_mask = (1 << entropy_bits) - 1;
			ew = (width + block_mask) >> entropy_bits;
			eh = (height + block_mask) >> entropy_bits;
			ep = NULL;
			entropy_stride = ew;

			err = swebp__decode_vp8l_image(simplewebp, br, 0, ew, eh, &ep);
			if (err != SIMPLEWEBP_NO_ERROR)
			{
				swebp__dealloc(simplewebp, ep);
				swebp__dealloc(simplewebp, color_cache);
				return err;
			}

			entropy = ep;

			for (i = 0; i < ew * eh; i++)
			{
				group = (ep[i].r << 8) | ep[i].g;
				if (group_count <= group)
					group_count = group + 1;
			}
		}

		if (br->eos)
		{
			swebp__dealloc(simplewebp, entropy);
			swebp__dealloc(simplewebp, color_cache);
			return SIMPLEWEBP_IO_ERROR;
		}
	}

	groups = (struct swebp__vp8l_group*) swebp__alloc(simplewebp, sizeof(struct swebp__vp8l_group) * group_count);
	if (!groups)
	{
		swebp__dealloc(simplewebp, entropy);
		swebp__dealloc(simplewebp, color_cache);
		return SIMPLEWEBP_ALLOC_ERROR;
	}
	memset(groups, 0, sizeof(struct swebp__vp8l_group) * group_count);

	for (i = 0; i < group_count; i++)
	{
		err = swebp__vp8l_decode_group(simplewebp, br, groups + i, ccache_bits);
		if (err != SIMPLEWEBP_NO_ERROR)
			break;
	}

	if (err == SIMPLEWEBP_NO_ERROR)
	{
		/* Main image decoding routines. */
		for (i = 0; i < width * height;)
		{
			struct swebp__pixel *pixel;
			struct swebp__vp8l_group *g = groups;
			simplewebp_u16 codeword;
			size_t j;

			pixel = image + i;

			if (entropy)
			{
				size_t ex, ey;
				struct swebp__pixel *ep;

				ex = (i % width) >> entropy_bits;
				ey = (i / width) >> entropy_bits;
				ep = entropy + entropy_stride * ey + ex;
				g += (ep->r << 8) | ep->g;
			}

			codeword = swebp__vp8l_read_code(br, &g->code[0]);

			if (codeword < swebp__vp8l_literals_count)
			{
				struct swebp__pixel color = {0, 0, 0, 0};
				color.r = (simplewebp_u8) swebp__vp8l_read_code(br, &g->code[1]);
				color.g = (simplewebp_u8) codeword;
				color.b = (simplewebp_u8) swebp__vp8l_read_code(br, &g->code[2]);
				color.a = (simplewebp_u8) swebp__vp8l_read_code(br, &g->code[3]);
				*pixel = color;
				i++;

				swebp__vp8l_put_cache(ccache_bits, color_cache, color);
			}
			else if (codeword < swebp__vp8l_litlen_count)
			{
				size_t length, distance;
				ptrdiff_t offset;
				simplewebp_u16 distcode;

				length = swebp__vp8l_lendst(br, codeword - swebp__vp8l_literals_count);
				distcode = swebp__vp8l_read_code(br, &g->code[4]);
				distance = swebp__vp8l_lendst(br, distcode);

				if (distance < swebp__vp8l_offset_count)
					offset = swebp__vp8l_offsets[distance][0] + swebp__vp8l_offsets[distance][1] * width;
				else
					offset = distance - swebp__vp8l_offset_count + 1;
				offset = offset < 1 ? 1 : offset;

				for (j = 0; j <= length; j++)
				{
					*pixel = pixel[-offset];
					swebp__vp8l_put_cache(ccache_bits, color_cache, *pixel);
					i++;
					pixel++;
				}
			}
			else
			{
				*pixel = color_cache[codeword - swebp__vp8l_litlen_count];
				i++;
			}
		}
	}

	/* Loop free memory */
	for (i = 0; i < group_count; i++)
	{
		size_t j;
		/* And the tree */
		for (j = 0; j < 5; j++)
			swebp__dealloc(simplewebp, groups[i].code[j].tree);
	}
	swebp__dealloc(simplewebp, entropy);
	swebp__dealloc(simplewebp, color_cache);
	swebp__dealloc(simplewebp, groups);

	return err;
}

static simplewebp_error swebp__decode_vp8l_transform_data(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	size_t width,
	size_t height,
	simplewebp_u8 *bits_out,
	struct swebp__pixel **dest
)
{
	int bits;
	size_t block_mask, data_w, data_h;

	bits = swebp__vp8l_bitread_read(br, 3) + 2;
	block_mask = (1 << bits) - 1;
	data_w = (width + block_mask) >> bits;
	data_h = (height + block_mask) >> bits;
	*bits_out = (simplewebp_u8) bits;

	return swebp__decode_vp8l_image(simplewebp, br, 0, data_w, data_h, dest);
}

static void swebp__batch_free(simplewebp *simplewebp, void **ptr, size_t count)
{
	size_t i;
	for (i = 0; i < count; i++)
	{
		if (ptr[i])
			swebp__dealloc(simplewebp, ptr[i]);
	}
}

static simplewebp_u8 swebp__vp8l_index_reduction(simplewebp_u32 color_count)
{
	if (color_count > 16)
		return 0;
	else if (color_count > 4)
		return 1;
	else if (color_count > 2)
		return 2;
	else
		return 3;
}

static simplewebp_error swebp__decode_color_index_transform(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	size_t width,
	size_t height,
	simplewebp_u8 *index_size_out,
	struct swebp__pixel **dest
)
{
	(void)width;
	(void)height;
	simplewebp_u32 color_count, i;
	simplewebp_error err;
	struct swebp__pixel *pixel_data;

	color_count = swebp__vp8l_bitread_read(br, 8);
	*index_size_out = (simplewebp_u8) color_count++;
	if (br->eos)
		return SIMPLEWEBP_CORRUPT_ERROR;

	err = swebp__decode_vp8l_image(simplewebp, br, 0, color_count, 1, dest);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;

	pixel_data = *dest;
	/* Reverse subtraction coding */
	for (i = 1; i < color_count; i++)
	{
		struct swebp__pixel *prev = &pixel_data[i - 1];
		pixel_data[i].r += prev->r;
		pixel_data[i].g += prev->g;
		pixel_data[i].b += prev->b;
		pixel_data[i].a += prev->a;
	}

	return SIMPLEWEBP_NO_ERROR;
}

static struct swebp__pixel swebp__average2(struct swebp__pixel a, struct swebp__pixel b)
{
	struct swebp__pixel result;
	result.r = ((simplewebp_u16) a.r + (simplewebp_u16) b.r) / 2;
	result.g = ((simplewebp_u16) a.g + (simplewebp_u16) b.g) / 2;
	result.b = ((simplewebp_u16) a.b + (simplewebp_u16) b.b) / 2;
	result.a = ((simplewebp_u16) a.a + (simplewebp_u16) b.a) / 2;
	return result;
}

static struct swebp__pixel swebp__select(struct swebp__pixel l, struct swebp__pixel t, struct swebp__pixel tl)
{
	int red = (int) l.r + (int) t.r - (int) tl.r;
	int green = (int) l.g + (int) t.g - (int) tl.g;
	int blue = (int) l.b + (int) t.b - (int) tl.b;
	int alpha = (int) l.a + (int) t.a - (int) tl.a;
	int pl = abs(alpha - l.a) + abs(red - l.r) + abs(green - l.g) + abs(blue - l.b);
	int pt = abs(alpha - t.a) + abs(red - t.r) + abs(green - t.g) + abs(blue - t.b);

	return pl < pt ? l : t;
}

static struct swebp__pixel swebp__clamp_add_subtract_full(
	struct swebp__pixel a,
	struct swebp__pixel b,
	struct swebp__pixel c
)
{
	struct swebp__pixel result;
	result.r = swebp__clip((int) a.r + (int) b.r - (int) c.r, 255);
	result.g = swebp__clip((int) a.g + (int) b.g - (int) c.g, 255);
	result.b = swebp__clip((int) a.b + (int) b.b - (int) c.b, 255);
	result.a = swebp__clip((int) a.a + (int) b.a - (int) c.a, 255);
	return result;
}

static struct swebp__pixel swebp__clamp_add_subtract_half(struct swebp__pixel a, struct swebp__pixel b)
{
	struct swebp__pixel result;
	result.r = swebp__clip((int) a.r + ((int) a.r - (int) b.r) / 2, 255);
	result.g = swebp__clip((int) a.g + ((int) a.g - (int) b.g) / 2, 255);
	result.b = swebp__clip((int) a.b + ((int) a.b - (int) b.b) / 2, 255);
	result.a = swebp__clip((int) a.a + ((int) a.a - (int) b.a) / 2, 255);
	return result;
}

static struct swebp__pixel swebp__apply_predictor(
	simplewebp_u8 type,
	struct swebp__pixel *l,
	struct swebp__pixel *tl,
	struct swebp__pixel *t,
	struct swebp__pixel *tr
)
{
	const struct swebp__pixel black = {0, 0, 0, 255};

	switch (type)
	{
		case 0:
		default:
			return black;
		case 1:
			return *l;
		case 2:
			return *t;
		case 3:
			return *tr;
		case 4:
			return *tl;
		case 5:
			return swebp__average2(swebp__average2(*l, *tr), *t);
		case 6:
			return swebp__average2(*l, *tl);
		case 7:
			return swebp__average2(*l, *t);
		case 8:
			return swebp__average2(*tl, *t);
		case 9:
			return swebp__average2(*tr, *t);
		case 10:
			return swebp__average2(swebp__average2(*l, *tl), swebp__average2(*tr, *t));
		case 11:
			return swebp__select(*l, *t, *tl);
		case 12:
			return swebp__clamp_add_subtract_full(*l, *t, *tl);
		case 13:
			return swebp__clamp_add_subtract_half(swebp__average2(*l, *t), *tl);
	}
}

static size_t swebp__subsample_size(size_t size, size_t sampling_bits)
{
	return (size + (((size_t) 1) << sampling_bits) - 1) >> sampling_bits;
}

static void swebp__apply_predictor_transform(
	struct swebp__pixel *rgba,
	size_t width,
	size_t height,
	simplewebp_u8 bits,
	struct swebp__pixel *predictor_data
)
{
	size_t x, y;
	size_t tiles_per_row = swebp__subsample_size(width, bits);

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			struct swebp__pixel result;
			simplewebp_u8 type = 0;

			if (x > 0)
			{
				if (y > 0)
				{
					size_t tile_x = x >> bits;
					size_t tile_y = y >> bits;
					size_t tile_index = tile_y * tiles_per_row + tile_x;
					type = predictor_data[tile_index].g;
				}
				else
					type = 1;
			}
			else
				type = y > 0 ? 2 : 0;

			result = swebp__apply_predictor(type, rgba - 1, rgba - (width + 1), rgba - width, rgba - (width - 1));
			rgba->r += result.r;
			rgba->g += result.g;
			rgba->b += result.b;
			rgba->a += result.a;
			rgba++;
		}
	}
}

static simplewebp_u8 swebp__color_delta(int c1, int c2) {
	int sc1 = c1 >= 128 ? -256 + c1 : c1;
	int sc2 = c2 >= 128 ? -256 + c2 : c2;
	return (simplewebp_u8) ((sc1 * sc2) >> 5);
}

static void swebp__apply_color_transform(
	struct swebp__pixel *rgba,
	size_t width,
	size_t height,
	simplewebp_u8 bits,
	struct swebp__pixel *color_data
)
{
	size_t x, y;
	size_t tiles_per_row = swebp__subsample_size(width, bits);

	for (y = 0; y < height; y++)
	{
		struct swebp__pixel *line = color_data + (y >> bits) * tiles_per_row;

		for (x = 0; x < width; x++)
		{
			struct swebp__pixel *ref = &line[x >> bits];
			rgba->r += swebp__color_delta(ref->b, rgba->g);
			rgba->b += swebp__color_delta(ref->g, rgba->g);
			rgba->b += swebp__color_delta(ref->r, rgba->r);
			rgba++;
		}
	}
}

static void swebp__apply_green_sub_transform(
	struct swebp__pixel *rgba,
	size_t width,
	size_t height
)
{
	size_t x, y;

	for (y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++)
		{
			struct swebp__pixel *col = &rgba[y * width + x];
			col->r += col->g;
			col->b += col->g;
		}
	}
}

static void swebp__apply_index_transform(
	struct swebp__pixel *rgba,
	size_t width,
	size_t height,
	simplewebp_u8 size,
	struct swebp__pixel *index_data
)
{
	size_t x, y;
	simplewebp_u32 color_count;
	simplewebp_u8 bits, mask, mod, rb;
	size_t stride;
	const struct swebp__pixel transparent_black = {0, 0, 0, 0};

	color_count = ((simplewebp_u32) size) + 1;
	bits = swebp__vp8l_index_reduction(color_count);
	stride = swebp__subsample_size(width, bits);
	mask = (1 << (8 >> bits)) - 1;
	mod = (1 << bits) - 1;
	rb = 3 - bits;

	for (y = height - 1; y != (size_t) -1; y--)
	{
		for (x = width - 1; x != (size_t) -1; x--)
		{
			const struct swebp__pixel *l = rgba + stride * y;
			simplewebp_u8 i = (l[x >> bits].g >> ((x & mod) << rb)) & mask;
			rgba[y * width + x] = i < color_count ? index_data[i] : transparent_black;
		}
	}
}

static simplewebp_error swebp__decode_lossless_bitstream_main(
	simplewebp *simplewebp,
	struct swebp__vp8l_bdec *br,
	struct swebp__pixel *rgba,
	simplewebp_bool skipheader
)
{
	size_t actual_width, actual_height;
	size_t full_width;
	simplewebp_u32 width, height;
	simplewebp_u8 filter_list;
	struct swebp__pixel *filter_data[4];
	simplewebp_u8 filter_bits[4];
	simplewebp_bool filter_active[4];
	size_t i;
	simplewebp_error err;

	simplewebp_get_dimensions(simplewebp, &actual_width, &actual_height);

	if (!skipheader)
	{
		simplewebp_u32 version_num;

		/* Validate header */
		if (swebp__vp8l_bitread_read(br, 8) != 0x2F)
			return SIMPLEWEBP_CORRUPT_ERROR;

		/* Validate dimensions */
		full_width = width = swebp__vp8l_bitread_read(br, 14) + 1;
		height = swebp__vp8l_bitread_read(br, 14) + 1;
		if (width != actual_width || height != actual_height)
			return SIMPLEWEBP_CORRUPT_ERROR;

		/* Skip alpha */
		swebp__vp8l_bitread_read(br, 1);
		version_num = swebp__vp8l_bitread_read(br, 3);
		if (version_num != 0)
			return SIMPLEWEBP_UNSUPPORTED_ERROR;
	}
	else
	{
		width = (simplewebp_u32) (full_width = actual_width);
		height = (simplewebp_u32) actual_height;
	}

	/* Read transforms */
	err = SIMPLEWEBP_NO_ERROR;
	filter_list = 0;
	memset(filter_active, 0, sizeof(simplewebp_bool) * 4);
	memset(filter_bits, 0, sizeof(simplewebp_u8) * 4);
	memset(filter_data, 0, sizeof(filter_data));
	for (i = 0; i < sizeof(simplewebp_u8) * 4; i++)
	{
		simplewebp_u8 ttype, transform_bits;
		struct swebp__pixel *filter_out = NULL;

		if (swebp__vp8l_bitread_read(br, 1) == 0)
			break;

		ttype = swebp__vp8l_bitread_read(br, 2);
		if (br->eos)
			return SIMPLEWEBP_CORRUPT_ERROR;

		switch (ttype)
		{
			case 0:
			/* Predictor transform */
			case 1:
			/* Color transform */
				err = swebp__decode_vp8l_transform_data(simplewebp, br, width, height, &transform_bits, &filter_out);
				break;
			case 2:
			/* Green subtraction transform */
				err = SIMPLEWEBP_NO_ERROR;
				transform_bits = 0;
				filter_out = NULL;
				break;
			case 3:
			/* Color index transform */
				err = swebp__decode_color_index_transform(
					simplewebp,
					br,
					width,
					height,
					&transform_bits,
					&filter_out
				);
				width = (simplewebp_u32) swebp__subsample_size(width, swebp__vp8l_index_reduction(transform_bits + 1));
				break;
			default:
			/* Unreachable */
				err = SIMPLEWEBP_CORRUPT_ERROR;
				break;
		}

		if (err != SIMPLEWEBP_NO_ERROR)
		{
			/* Error occured. Rollback. */
			swebp__batch_free(simplewebp, (void **) filter_data, 4);
			return err;
		}

		if (filter_active[ttype])
		{
			/* Multiple same filters defined, likely corrupt */
			swebp__dealloc(simplewebp, filter_out);
			swebp__batch_free(simplewebp, (void **) filter_data, 4);
			return SIMPLEWEBP_CORRUPT_ERROR;
		}

		filter_list = (filter_list) << 2 | ttype;
		filter_bits[i] = transform_bits;
		filter_data[i] = filter_out;
		filter_active[ttype] = 1;
	}

	/* Decode image */
	err = swebp__decode_vp8l_image(simplewebp, br, 1, width, height, &rgba);
	if (err != SIMPLEWEBP_NO_ERROR)
	{
		swebp__batch_free(simplewebp, (void **) filter_data, sizeof(simplewebp_u8) * 4);
		return err;
	}

	/* Apply transform */
	{
		int j = (int)(i - 1);
		for (; j >= 0; j--)
		{
			simplewebp_u8 ttype;
			ttype = filter_list & 3;

			switch (ttype)
			{
				case 0:
					/* Predictor Transform */
					swebp__apply_predictor_transform(rgba, width, height, filter_bits[j], filter_data[j]);
					break;
				case 1:
					/* Color Transform */
					swebp__apply_color_transform(rgba, width, height, filter_bits[j], filter_data[j]);
					break;
				case 2:
					/* Green Transform */
					swebp__apply_green_sub_transform(rgba, width, height);
					break;
				case 3:
					/* Color Index Transform */
					width = (simplewebp_u32) full_width;
					swebp__apply_index_transform(rgba, width, height, filter_bits[j], filter_data[j]);
					break;
			}

			filter_list >>= 2;
		}
	}

	swebp__batch_free(simplewebp, (void **) filter_data, sizeof(simplewebp_u8) * 4);
	return err;
}

/* Also used to decode alpha channel. */
static simplewebp_error swebp__decode_lossless_bitstream(
	simplewebp *simplewebp,
	simplewebp_input *input,
	size_t bitstreamsize,
	struct swebp__pixel *rgba,
	simplewebp_bool skipheader
)
{
	struct swebp__vp8l_bdec br;
	simplewebp_error err;
	simplewebp_u8 *bitstreambuffer = (simplewebp_u8 *) swebp__alloc(simplewebp, bitstreamsize);

	if (!bitstreambuffer)
		return SIMPLEWEBP_ALLOC_ERROR;

	if (!swebp__read2(bitstreamsize, bitstreambuffer, input))
	{
		swebp__dealloc(simplewebp, bitstreambuffer);
		return SIMPLEWEBP_IO_ERROR;
	}

	swebp__vp8l_bitread_init(&br, bitstreambuffer, bitstreamsize);
	err = swebp__decode_lossless_bitstream_main(simplewebp, &br, rgba, skipheader);

	swebp__dealloc(simplewebp, bitstreambuffer);
	return err;
}

static simplewebp_error swebp__decode_lossless(simplewebp *simplewebp, void *buffer)
{
	size_t vp8lsize = swebp__proxy_size(simplewebp->vp8l_input.userdata);
	if (!swebp__seek(0, &simplewebp->vp8l_input))
		return SIMPLEWEBP_IO_ERROR;
	return swebp__decode_lossless_bitstream(
		simplewebp,
		&simplewebp->vp8l_input,
		vp8lsize,
		(struct swebp__pixel *) buffer,
		0
	);
}

simplewebp_error simplewebp_decode_yuva(simplewebp *simplewebp, void *y_buffer, void *u_buffer, void *v_buffer, void *a_buffer, void *settings)
{
	if (simplewebp->webp_type == 0)
	{
		struct swebp__yuvdst destination;
		destination.y = (simplewebp_u8*) y_buffer;
		destination.u = (simplewebp_u8*) u_buffer;
		destination.v = (simplewebp_u8*) v_buffer;
		destination.a = (simplewebp_u8*) a_buffer;
		return swebp__decode_lossy(simplewebp, &destination, settings);
	}

	return SIMPLEWEBP_IS_LOSSLESS_ERROR;
}

simplewebp_error simplewebp_decode(simplewebp *simplewebp, void *buffer, void *settings)
{
	simplewebp_error err = SIMPLEWEBP_NO_ERROR;

	if (simplewebp->webp_type == 0)
	{
		struct swebp__yuvdst dest;
		struct swebp__chroma *upscaled;
		simplewebp_u8 *mem, *orig_mem;
		size_t needed, yw, yh, uvw, uvh;

		/* Calculate allocation size. The allocated memory layout is [y, a, u, v] */
		yw = simplewebp->decoder.vp8.picture_header.width;
		yh = simplewebp->decoder.vp8.picture_header.height;
		uvw = (yw + 1) / 2;
		uvh = (yh + 1) / 2;
		needed = ((yw * yh) + (uvw * uvh)) * 2;
		/* For upsampling the UV */
		needed += uvw * uvh * 4 * sizeof(struct swebp__chroma);

		mem = orig_mem = (simplewebp_u8 *) swebp__alloc(simplewebp, needed);
		if (mem == NULL)
			return SIMPLEWEBP_ALLOC_ERROR;

		dest.y = mem;
		mem += yw * yh;
		dest.a = mem;
		mem += yw * yh;
		dest.u = mem;
		mem += uvw * uvh;
		dest.v = mem;
		mem += uvw * uvh;
		upscaled = (struct swebp__chroma*) mem;

		err = swebp__decode_lossy(simplewebp, &dest, settings);
		if (err != SIMPLEWEBP_NO_ERROR)
		{
			swebp__dealloc(simplewebp, orig_mem);
			return err;
		}

		/* Upsample UV */
		swebp__upsample_chroma(dest.u, dest.v, upscaled, uvw, uvh);
		/* Convert YUVA to RGBA */
		swebp__yuva2rgba(dest.y, upscaled, dest.a, yw, yh, (struct swebp__pixel*) buffer);
		swebp__dealloc(simplewebp, orig_mem);
	}
	else
	{
		err = swebp__decode_lossless(simplewebp, buffer);
		if (err != SIMPLEWEBP_NO_ERROR)
			return err;
		
		if (!simplewebp->decoder.vp8l.has_alpha)
		{
			size_t i, w, h;
			struct swebp__pixel *rgba;

			w = simplewebp->decoder.vp8l.width;
			h = simplewebp->decoder.vp8l.height;

			/* libwebp sets the alpha to 255 if the has_alpha bit is not set. */
			/* Interestingly, WebPShop as of 0.4.3 doesn't seem to respect that flag (it's always RGBA). */
			rgba = (struct swebp__pixel*) buffer;
			for (i = 0; i < w * h; i++)
				rgba[i].a = 255;
		}
	}

	return err;
}

#ifndef SIMPLEWEBP_DISABLE_STDIO

static size_t swebp__stdfile_read(size_t size, void *dest, void *userdata)
{
	FILE *f = (FILE *) userdata;
	return fread(dest, 1, size, f);
}

static simplewebp_bool swebp__stdfile_seek(size_t pos, void *userdata)
{
	FILE *f = (FILE *) userdata;
	return fseek(f, (long) pos, SEEK_SET) == 0;
}

static void swebp__stdfile_close(void *userdata)
{
	FILE *f = (FILE *) userdata;
	fclose(f);
}

static size_t swebp__stdfile_tell(void *userdata)
{
	FILE *f = (FILE *) userdata;
	return ftell(f);
}

simplewebp_error simplewebp_input_from_file(FILE *file, simplewebp_input *out)
{
	out->userdata = file;
	out->read = swebp__stdfile_read;
	out->seek = swebp__stdfile_seek;
	out->tell = swebp__stdfile_tell;
	out->close = swebp__stdfile_close;
	return SIMPLEWEBP_NO_ERROR;
}

simplewebp_error simplewebp_input_from_filename(const char *filename, simplewebp_input *out)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL)
		return SIMPLEWEBP_IO_ERROR;

	return simplewebp_input_from_file(f, out);
}

simplewebp_error simplewebp_load_from_file(FILE *file, const simplewebp_allocator *allocator, simplewebp **out)
{
	simplewebp_input input;
	simplewebp_error err;

	err = simplewebp_input_from_file(file, &input);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;

	err = simplewebp_load(&input, allocator, out);
	if (err != SIMPLEWEBP_NO_ERROR)
		simplewebp_close_input(&input);

	return err;
}

simplewebp_error simplewebp_load_from_filename(const char *filename, const simplewebp_allocator *allocator, simplewebp **out)
{
	simplewebp_input input;
	simplewebp_error err;

	err = simplewebp_input_from_filename(filename, &input);
	if (err != SIMPLEWEBP_NO_ERROR)
		return err;

	err = simplewebp_load(&input, allocator, out);
	if (err != SIMPLEWEBP_NO_ERROR)
		simplewebp_close_input(&input);

	return err;
}
#endif /* SIMPLEWEBP_DISABLE_STDIO */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIMPLEWEBP_IMPLEMENTATION */

/**
 * This software is available under BSD-3-Clause License
 * Some parts of WebP Lossless code is taken from "whale" project by Matej Fencl
 * 
 * Copyright (c) 2010 Google Inc., 2023 Miku AuahDark
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *    following disclaimer in the documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
 *    products derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * Additional IP Rights Grant (Patents)
 * ------------------------------------
 * 
 * "These implementations" means the copyrightable works that implement the WebM codecs distributed by Google as part
 * of the WebM Project.
 * 
 * Google hereby grants to you a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as
 * stated in this section) patent license to make, have made, use, offer to sell, sell, import, transfer, and
 * otherwise run, modify and propagate the contents of these implementations of WebM, where such license applies only
 * to those patent claims, both currently owned by Google and acquired in the future, licensable by Google that are
 * necessarily infringed by these implementations of WebM. This grant does not include claims that would be infringed
 * only as a consequence of further modification of these implementations. If you or your agent or exclusive licensee
 * institute or order or agree to the institution of patent litigation or any other patent enforcement activity
 * against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that any of these
 * implementations of WebM or any code incorporated within any of these implementations of WebM constitute direct or
 * contributory patent infringement, or inducement of patent infringement, then any patent rights granted to you under
 * this License for these implementations of WebM shall terminate as of the date such litigation is filed.
 */
