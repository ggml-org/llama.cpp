package com.arm.aichat.internal.gguf

import android.content.Context
import android.net.Uri
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.arm.aichat.gguf.InvalidFileFormatException
import java.io.File
import java.io.IOException
import java.io.InputStream


/**
 * Utility class to read GGUF model files and extract metadata key-value pairs.
 * This parser reads the header and metadata of a GGUF v3 file (little-endian) and skips tensor data.
 */
internal class GgufMetadataReaderImpl(
    private val skipKeys: Set<String>,
    private val arraySummariseThreshold: Int,
) : GgufMetadataReader {
    companion object {
        private const val ARCH_LLAMA = "llama"
    }

    /** Enum corresponding to GGUF metadata value types (for convenience and array element typing). */
    enum class MetadataType(val code: Int) {
        UINT8(0), INT8(1), UINT16(2), INT16(3),
        UINT32(4), INT32(5), FLOAT32(6), BOOL(7),
        STRING(8), ARRAY(9), UINT64(10), INT64(11), FLOAT64(12);
        companion object {
            private val codeMap = entries.associateBy(MetadataType::code)
            fun fromCode(code: Int): MetadataType = codeMap[code]
                ?: throw IOException("Unknown metadata value type code: $code")
        }
    }

    /** Sealed class hierarchy for metadata values, providing type-safe representations for each GGUF metadata type. */
    sealed class MetadataValue {
        data class UInt8(val value: UByte) : MetadataValue()       // 0:  8-bit unsigned int
        data class Int8(val value: Byte) : MetadataValue()         // 1:  8-bit signed int
        data class UInt16(val value: UShort) : MetadataValue()     // 2:  16-bit unsigned int (little-endian)
        data class Int16(val value: Short) : MetadataValue()       // 3:  16-bit signed int (little-endian)
        data class UInt32(val value: UInt) : MetadataValue()       // 4:  32-bit unsigned int (little-endian)
        data class Int32(val value: Int) : MetadataValue()         // 5:  32-bit signed int (little-endian)
        data class Float32(val value: Float) : MetadataValue()     // 6:  32-bit IEEE754 float
        data class Bool(val value: Boolean) : MetadataValue()      // 7:  Boolean (1-byte, 0=false, 1=true)
        data class StringVal(val value: String) : MetadataValue()  // 8:  UTF-8 string (length-prefixed)
        data class ArrayVal(val elementType: MetadataType, val elements: List<MetadataValue>) : MetadataValue()
        data class UInt64(val value: ULong) : MetadataValue()      // 10: 64-bit unsigned int (little-endian)
        data class Int64(val value: Long) : MetadataValue()        // 11: 64-bit signed int (little-endian)
        data class Float64(val value: Double) : MetadataValue()    // 12: 64-bit IEEE754 double
    }

    /* Convert MetadataValue to plain Kotlin primitives for allMetadata map */
    private fun MetadataValue.toPrimitive(): Any = when (this) {
        is MetadataValue.UInt8     -> value
        is MetadataValue.Int8      -> value
        is MetadataValue.UInt16    -> value
        is MetadataValue.Int16     -> value
        is MetadataValue.UInt32    -> value
        is MetadataValue.Int32     -> value
        is MetadataValue.Float32   -> value
        is MetadataValue.Bool      -> value
        is MetadataValue.StringVal -> value
        is MetadataValue.UInt64    -> value
        is MetadataValue.Int64     -> value
        is MetadataValue.Float64   -> value
        is MetadataValue.ArrayVal  -> elements.map { it.toPrimitive() }
    }

    /**
     * Reads the magic number from the specified file path.
     *
     * @param context Context for obtaining ContentResolver
     * @param uri Uri to the GGUF file provided by ContentProvider
     * @return true if file is valid GGUF, otherwise false
     */
    override suspend fun ensureSourceFileFormat(file: File): Boolean =
        file.inputStream().buffered().use { ensureMagic(it) }

    /**
     * Reads the magic number from the specified file path.
     *
     * @param context Context for obtaining ContentResolver
     * @param uri Uri to the GGUF file provided by ContentProvider
     * @return true if file is valid GGUF, otherwise false
     */
    override suspend fun ensureSourceFileFormat(context: Context, uri: Uri): Boolean =
        context.contentResolver.openInputStream(uri)?.buffered()?.use { ensureMagic(it) } == true

    /** Reads the 4‑byte magic; throws if magic ≠ "GGUF". */
    private fun ensureMagic(input: InputStream): Boolean =
        ByteArray(4).let {
            if (input.read(it) != 4) throw IOException("Not a valid file!")
            it.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46)) // "GGUF"
        }

    /**
     * High‑level entry point: parses a `.gguf` file on disk and returns the fully
     * populated [GgufMetadata] tree.
     *
     * Steps performed internally:
     * 1.  Reads and validates the 8‑byte header (`"GGUF"` magic + version).
     * 2.  Streams through the key‑value section, skipping large blobs if the key
     *     appears in [skipKeys] or if an array exceeds [arraySummariseThreshold].
     * 3.  Converts the resulting raw map into strongly‑typed sub‑structures
     *     (basic info, tokenizer, rope, etc.).
     *
     * The method is STREAMING‑ONLY: tensors are never mapped or loaded into
     * memory, so even multi‑GB model files can be processed in < 50 ms.
     *
     * @param path Absolute or relative filesystem path to a `.gguf` file.
     * @return A [GgufMetadata] instance containing all recognised metadata plus
     *         an `allMetadata` map with any keys that were not given a dedicated
     *         field.
     * @throws IOException if the file is not GGUF, the version is unsupported,
     *         or the metadata block is truncated / corrupt.
     */
    override suspend fun readStructuredMetadata(input: InputStream): GgufMetadata {
        // ── 1. header ──────────────────────────────────────────────────────────
        // throws on mismatch
        val version       = ensureMagicAndVersion(input)
        val tensorCount   = readLittleLong(input)
        val kvCount       = readLittleLong(input)

        // ── 2. metadata map (reuse our raw parser, but we need access to the stream) ──
        val meta = readMetaMap(input, kvCount)    // <String, MetadataValue>

        // ── 3. build structured object ────────────────────────────────────────
        return buildStructured(meta, version, tensorCount, kvCount)
    }

    /** Reads the 4‑byte magic + 4‑byte version; throws if magic ≠ "GGUF". */
    private fun ensureMagicAndVersion(input: InputStream): GgufMetadata.GgufVersion {
        if (!ensureMagic(input)) throw InvalidFileFormatException()
        return GgufMetadata.GgufVersion.fromCode(readLEUInt32(input))
    }

    /**
     * Read an unsigned 32‑bit little‑endian integer.
     *
     * @throws IOException if fewer than four bytes are available.
     */
    private fun readLEUInt32(input: InputStream): Int {
        val b0 = input.read(); val b1 = input.read(); val b2 = input.read(); val b3 = input.read()
        if (b3 == -1) throw IOException("Unexpected EOF while reading UInt32")
        return (b3 and 0xFF shl 24) or
            (b2 and 0xFF shl 16) or
            (b1 and 0xFF shl  8) or
            (b0 and 0xFF)
    }

    /**
     * Low‑level helper that reads the entire “key-value” section from the current
     * stream position.
     *
     * @param input  Open stream positioned JUST AFTER the header.
     * @param kvCnt  Number of key‑value pairs (taken from the header).
     * @return       Mutable map with one [MetadataValue] for every key that is NOT skipped.
     *
     * The function honours [skipKeys] and [arraySummariseThreshold] by invoking
     * [skipValue] or [parseValue] accordingly.
     */
    private fun readMetaMap(input: InputStream, kvCnt: Long): Map<String, MetadataValue> =
        mutableMapOf<String, MetadataValue>().apply {
             repeat(kvCnt.toInt()) {
                 val key = readString(input)
                 val valueT = MetadataType.fromCode(littleEndianBytesToInt(input.readNBytesExact(4)))
                 if (key in skipKeys) {
                     skipValue(input, valueT)
                 } else {
                     this[key] = parseValue(input, valueT)
                 }
             }
         }

    /**
     * Converts a flat [Map]<[String], [MetadataValue]> into the strongly‑typed
     * [GgufMetadata] tree used by the rest of the app.
     *
     * Only the keys listed in the spec are copied into dedicated data classes;
     * everything else is preserved in `GgufMetadata.allMetadata`.
     *
     * @param m            Raw key/value map.
     * @param version      GGUF file‑format version (enum).
     * @param tensorCnt    Number of tensors (from the header).
     * @param kvCnt        Total metadata pair count (from the header).
     */
    private fun buildStructured(
        m: Map<String, MetadataValue>,
        version: GgufMetadata.GgufVersion,
        tensorCnt: Long,
        kvCnt: Long
    ): GgufMetadata {
        // ---------- helpers ----------
        fun String.str()  = (m[this] as? MetadataValue.StringVal)?.value
        fun String.bool() = (m[this] as? MetadataValue.Bool)?.value
        fun String.i32()  = (m[this] as? MetadataValue.Int32)?.value
        fun String.u32()  = (m[this] as? MetadataValue.UInt32)?.value?.toInt()
        fun String.f32()  = (m[this] as? MetadataValue.Float32)?.value
        fun String.f64()  = (m[this] as? MetadataValue.Float64)?.value?.toFloat()
        fun String.strList(): List<String>? =
            (m[this] as? MetadataValue.ArrayVal)
                ?.elements
                ?.mapNotNull { (it as? MetadataValue.StringVal)?.value }

        val arch = "general.architecture".str() ?: ARCH_LLAMA

        // -------------- populate sections ----------------
        val basic = GgufMetadata.BasicInfo(
            uuid      = "general.uuid".str(),
            name      = "general.basename".str(),
            nameLabel = "general.name".str(),
            sizeLabel = "general.size_label".str()
        )

        val author = GgufMetadata.AuthorInfo(
            organization = "general.organization".str(),
            author       = "general.author".str(),
            doi          = "general.doi".str(),
            url          = "general.url".str(),
            repoUrl      = "general.repo_url".str(),
            license      = "general.license".str(),
            licenseLink  = "general.license.link".str()
        ).takeUnless {
            organization == null && author == null && doi == null &&
                url == null && repoUrl == null && license == null && licenseLink == null
        }

        val additional = GgufMetadata.AdditionalInfo(
            type        = "general.type".str(),
            description = "general.description".str(),
            tags        = "general.tags".strList(),
            languages   = "general.languages".strList()
        ).takeUnless {
            type == null && description == null && tags == null && languages == null
        }

        val architectureInfo = GgufMetadata.ArchitectureInfo(
            architecture        = arch,
            fileType            = "general.file_type".u32(),
            vocabSize           = "$arch.vocab_size".u32(),
            finetune            = "general.finetune".str(),
            quantizationVersion = "general.quantization_version".u32()
        ).takeUnless { fileType == null && vocabSize == null && finetune == null && quantizationVersion == null }

        val baseModels = buildList {
            val n = "general.base_model.count".u32() ?: 0
            for (i in 0 until n) {
                fun k(s: String) = "general.base_model.$i.$s"
                add(
                    GgufMetadata.BaseModelInfo(
                        name         = k("name").str(),
                        author       = k("author").str(),
                        version      = k("version").str(),
                        organization = k("organization").str(),
                        url          = k("url").str(),
                        doi          = k("doi").str(),
                        uuid         = k("uuid").str(),
                        repoUrl      = k("repo_url").str(),
                    )
                )
            }
        }.takeIf { it.isNotEmpty() }

        val tokenizer = GgufMetadata.TokenizerInfo(
            model            = "tokenizer.ggml.model".str(),
            bosTokenId       = "tokenizer.ggml.bos_token_id".u32(),
            eosTokenId       = "tokenizer.ggml.eos_token_id".u32(),
            unknownTokenId   = "tokenizer.ggml.unknown_token_id".u32(),
            paddingTokenId   = "tokenizer.ggml.padding_token_id".u32(),
            addBosToken      = "tokenizer.ggml.add_bos_token".bool(),
            addEosToken      = "tokenizer.ggml.add_eos_token".bool(),
            chatTemplate     = "tokenizer.chat_template".str()
        ).takeUnless { model == null && bosTokenId == null && eosTokenId == null &&
            unknownTokenId == null && paddingTokenId == null &&
            addBosToken == null && addEosToken == null && chatTemplate == null
        }

        val dimensions = GgufMetadata.DimensionsInfo(
            contextLength    = "$arch.context_length".u32(),
            embeddingSize    = "$arch.embedding_length".u32(),
            blockCount       = "$arch.block_count".u32(),
            feedForwardSize  = "$arch.feed_forward_length".u32()
        ).takeUnless { contextLength == null && embeddingSize == null && blockCount == null && feedForwardSize == null }

        val attention = GgufMetadata.AttentionInfo(
            headCount             = "$arch.attention.head_count".u32(),
            headCountKv           = "$arch.attention.head_count_kv".u32(),
            keyLength             = "$arch.attention.key_length".u32(),
            valueLength           = "$arch.attention.value_length".u32(),
            layerNormEpsilon      = "$arch.attention.layer_norm_epsilon".f32(),
            layerNormRmsEpsilon   = "$arch.attention.layer_norm_rms_epsilon".f32(),
        ).takeUnless { headCount == null && headCountKv == null && keyLength == null && valueLength == null &&
            layerNormEpsilon == null && layerNormRmsEpsilon == null
        }

        val rope = GgufMetadata.RopeInfo(
            frequencyBase          = "$arch.rope.freq_base".f32(),
            dimensionCount         = "$arch.rope.dimension_count".u32(),
            scalingType            = "$arch.rope.scaling.type".str(),
            scalingFactor          = "$arch.rope.scaling.factor".f32(),
            attnFactor             = "$arch.rope.scaling.attn_factor".f32(),
            originalContextLength  = "$arch.rope.scaling.original_context_length".u32(),
            finetuned              = "$arch.rope.scaling.finetuned".bool()
        ).takeUnless { frequencyBase == null && dimensionCount == null &&
            scalingType == null && scalingFactor == null && attnFactor == null &&
            originalContextLength == null && finetuned == null
        }

        val experts = GgufMetadata.ExpertsInfo(
            count      = "$arch.expert_count".u32(),
            usedCount  = "$arch.expert_used_count".u32()
        ).takeUnless { count == null && usedCount == null }

        return GgufMetadata(
            version = version,
            tensorCount = tensorCnt,
            kvCount = kvCnt,
            basic = basic,
            author = author,
            additional = additional,
            architecture = architectureInfo,
            baseModels = baseModels,
            tokenizer = tokenizer,
            dimensions = dimensions,
            attention = attention,
            rope = rope,
            experts = experts
        )
    }

    /**
     * Recursively parses a metadata value of the given type from the input stream.
     * @param input The input stream positioned at the start of the value.
     * @param type The metadata value type to parse.
     */
    private fun parseValue(input: InputStream, type: MetadataType): MetadataValue = when (type) {
        MetadataType.UINT8 -> {
            // 1-byte unsigned integer
            val byteVal = input.read()
            if (byteVal == -1) throw IOException("Unexpected EOF while reading uint8 value.")
            MetadataValue.UInt8(byteVal.toUByte())
        }
        MetadataType.INT8 -> {
            // 1-byte signed integer
            val byteVal = input.read()
            if (byteVal == -1) throw IOException("Unexpected EOF while reading int8 value.")
            MetadataValue.Int8(byteVal.toByte())
        }
        MetadataType.UINT16 -> {
            // 2-byte unsigned integer (little-endian)
            val bytes = ByteArray(2)
            if (input.read(bytes) != 2) throw IOException("Unexpected EOF while reading uint16 value.")
            // Combine two bytes (little-endian) into an unsigned 16-bit value
            val u16 = ((bytes[1].toInt() and 0xFF) shl 8) or (bytes[0].toInt() and 0xFF)
            MetadataValue.UInt16(u16.toUShort())
        }
        MetadataType.INT16 -> {
            // 2-byte signed integer (little-endian)
            val bytes = ByteArray(2)
            if (input.read(bytes) != 2) throw IOException("Unexpected EOF while reading int16 value.")
            // Combine to 16-bit and interpret as signed
            val i16 = ((bytes[1].toInt() and 0xFF) shl 8) or (bytes[0].toInt() and 0xFF)
            MetadataValue.Int16(i16.toShort())
        }
        MetadataType.UINT32 -> {
            // 4-byte unsigned integer (little-endian)
            val bytes = ByteArray(4)
            if (input.read(bytes) != 4) throw IOException("Unexpected EOF while reading uint32 value.")
            // Combine four bytes into a 32-bit value (as Long to avoid overflow), then convert to UInt
            val u32 = (bytes[3].toLong() and 0xFFL shl 24) or
                (bytes[2].toLong() and 0xFFL shl 16) or
                (bytes[1].toLong() and 0xFFL shl 8) or
                (bytes[0].toLong() and 0xFFL)
            MetadataValue.UInt32(u32.toUInt())
        }
        MetadataType.INT32 -> {
            // 4-byte signed integer (little-endian)
            val bytes = ByteArray(4)
            if (input.read(bytes) != 4) throw IOException("Unexpected EOF while reading int32 value.")
            // Combine four bytes into a 32-bit signed int
            val i32 = (bytes[3].toInt() and 0xFF shl 24) or
                (bytes[2].toInt() and 0xFF shl 16) or
                (bytes[1].toInt() and 0xFF shl 8) or
                (bytes[0].toInt() and 0xFF)
            MetadataValue.Int32(i32)
        }
        MetadataType.FLOAT32 -> {
            // 4-byte IEEE 754 float (little-endian)
            val bytes = ByteArray(4)
            if (input.read(bytes) != 4) throw IOException("Unexpected EOF while reading float32 value.")
            // Assemble 4 bytes into a 32-bit int bit-pattern, then convert to Float
            val bits = (bytes[3].toInt() and 0xFF shl 24) or
                (bytes[2].toInt() and 0xFF shl 16) or
                (bytes[1].toInt() and 0xFF shl 8) or
                (bytes[0].toInt() and 0xFF)
            val floatVal = Float.fromBits(bits)
            MetadataValue.Float32(floatVal)
        }
        MetadataType.BOOL -> {
            // 1-byte boolean (0 = false, 1 = true)
            val byteVal = input.read()
            if (byteVal == -1) throw IOException("Unexpected EOF while reading boolean value.")
            if (byteVal != 0 && byteVal != 1) {
                throw IOException("Invalid boolean value: $byteVal (must be 0 or 1).")
            }
            MetadataValue.Bool(byteVal != 0)
        }
        MetadataType.STRING -> {
            // UTF-8 string (length-prefixed with 8-byte length)
            val str = readString(input)
            MetadataValue.StringVal(str)
        }
        MetadataType.ARRAY -> {
            val elemType = MetadataType.fromCode(littleEndianBytesToInt(input.readNBytesExact(4)))
            val len      = readLittleLong(input)
            val count    = len.toInt()

            if (arraySummariseThreshold >= 0 && count > arraySummariseThreshold) {
                // fast‑forward without allocation
                repeat(count) { skipValue(input, elemType) }
                MetadataValue.StringVal("Array($elemType, $count items) /* summarised */")
            } else {
                val list = ArrayList<MetadataValue>(count)
                repeat(count) { list += parseValue(input, elemType) }
                MetadataValue.ArrayVal(elemType, list)
            }
        }
        MetadataType.UINT64 -> {
            // 8-byte unsigned integer (little-endian)
            val bytes = ByteArray(8)
            if (input.read(bytes) != 8) throw IOException("Unexpected EOF while reading uint64 value.")
            // Combine 8 bytes into an unsigned 64-bit (ULong). Use ULong for full 0 to 2^64-1 range.
            val u64 = (bytes[7].toULong() and 0xFFuL shl 56) or
                (bytes[6].toULong() and 0xFFuL shl 48) or
                (bytes[5].toULong() and 0xFFuL shl 40) or
                (bytes[4].toULong() and 0xFFuL shl 32) or
                (bytes[3].toULong() and 0xFFuL shl 24) or
                (bytes[2].toULong() and 0xFFuL shl 16) or
                (bytes[1].toULong() and 0xFFuL shl 8) or
                (bytes[0].toULong() and 0xFFuL)
            MetadataValue.UInt64(u64)
        }
        MetadataType.INT64 -> {
            // 8-byte signed integer (little-endian)
            val bytes = ByteArray(8)
            if (input.read(bytes) != 8) throw IOException("Unexpected EOF while reading int64 value.")
            // Combine 8 bytes into a signed 64-bit value (Long)
            val i64 = (bytes[7].toLong() and 0xFFL shl 56) or
                (bytes[6].toLong() and 0xFFL shl 48) or
                (bytes[5].toLong() and 0xFFL shl 40) or
                (bytes[4].toLong() and 0xFFL shl 32) or
                (bytes[3].toLong() and 0xFFL shl 24) or
                (bytes[2].toLong() and 0xFFL shl 16) or
                (bytes[1].toLong() and 0xFFL shl 8) or
                (bytes[0].toLong() and 0xFFL)
            MetadataValue.Int64(i64)
        }
        MetadataType.FLOAT64 -> {
            // 8-byte IEEE 754 double (little-endian)
            val bytes = ByteArray(8)
            if (input.read(bytes) != 8) throw IOException("Unexpected EOF while reading float64 value.")
            // Assemble 8 bytes into a 64-bit bit-pattern, then convert to Double
            val bits = (bytes[7].toLong() and 0xFFL shl 56) or
                (bytes[6].toLong() and 0xFFL shl 48) or
                (bytes[5].toLong() and 0xFFL shl 40) or
                (bytes[4].toLong() and 0xFFL shl 32) or
                (bytes[3].toLong() and 0xFFL shl 24) or
                (bytes[2].toLong() and 0xFFL shl 16) or
                (bytes[1].toLong() and 0xFFL shl 8) or
                (bytes[0].toLong() and 0xFFL)
            val doubleVal = Double.fromBits(bits)
            MetadataValue.Float64(doubleVal)
        }
    }


    private fun <T> T?.takeUnless(check: T.() -> Boolean): T? =
        this?.takeIf { !it.check() }

    /** Helper: Skip a value in the stream without storing it (still maintains pointer). */
    private fun skipValue(input: InputStream, type: MetadataType) {
        when (type) {
            MetadataType.UINT8, MetadataType.INT8, MetadataType.BOOL -> input.skipFully(1)
            MetadataType.UINT16, MetadataType.INT16                  -> input.skipFully(2)
            MetadataType.UINT32, MetadataType.INT32, MetadataType.FLOAT32 -> input.skipFully(4)
            MetadataType.UINT64, MetadataType.INT64, MetadataType.FLOAT64 -> input.skipFully(8)
            MetadataType.STRING -> {
                val len = readLittleLong(input); input.skipFully(len)
            }
            MetadataType.ARRAY -> {
                val elemType = MetadataType.fromCode(littleEndianBytesToInt(input.readNBytesExact(4)))
                val len      = readLittleLong(input)
                repeat(len.toInt()) { skipValue(input, elemType) }   // recursive skip
            }
        }
    }

    /** Helper: Read an 8-byte little-endian unsigned value and return it as a signed Long (assuming it fits in 63 bits). */
    private fun readLittleLong(input: InputStream): Long {
        val bytes = ByteArray(8)
        input.readFully(bytes)

        // Combine 8 bytes into a 64-bit value (Little Endian).
        // Note: If the value exceeds Long.MAX_VALUE (bit 63 is 1), this will produce a negative Long (two's complement).
        // In our context (lengths/counts), such extremely large values are not expected.
        return (bytes[7].toLong() and 0xFFL shl 56) or
            (bytes[6].toLong() and 0xFFL shl 48) or
            (bytes[5].toLong() and 0xFFL shl 40) or
            (bytes[4].toLong() and 0xFFL shl 32) or
            (bytes[3].toLong() and 0xFFL shl 24) or
            (bytes[2].toLong() and 0xFFL shl 16) or
            (bytes[1].toLong() and 0xFFL shl 8) or
            (bytes[0].toLong() and 0xFFL)
    }

    /** Helper: Read a GGUF string from the stream (8-byte length followed by UTF-8 bytes). */
    private fun readString(input: InputStream): String =
        // Read 8-byte little-endian length (number of bytes in the string).
        readLittleLong(input).let { len ->
            if (len < 0 || len > Int.MAX_VALUE) throw IOException("String too long: $len")

            // Read the UTF-8 bytes of the given length.
            ByteArray(len.toInt()).let {
                if (it.isNotEmpty()) input.readFully(it)
                String(it, Charsets.UTF_8)
            }
        }

    /** Helper: Convert a 4-byte little-endian byte array to a 32-bit integer. */
    private fun littleEndianBytesToInt(bytes: ByteArray): Int =
        // Note: assumes bytes length is 4.
        (bytes[3].toInt() and 0xFF shl 24) or
            (bytes[2].toInt() and 0xFF shl 16) or
            (bytes[1].toInt() and 0xFF shl 8) or
            (bytes[0].toInt() and 0xFF)

    /**
     * Robust skip that works the same on JDK 11 and Android’s desugared runtime.
     *
     * @param n  Number of bytes to advance in the stream.
     * @throws IOException on premature EOF.
     */
    private fun InputStream.skipFully(n: Long) {
        var remaining = n
        val scratch = ByteArray(8192)                 // read‑and‑toss buffer
        while (remaining > 0) {
            val skipped = skip(remaining)
            when {
                skipped > 0      -> remaining -= skipped               // normal fast path
                skipped == 0L    -> {
                    // fallback: read and discard
                    val read = read(scratch, 0, minOf(remaining, scratch.size.toLong()).toInt())
                    if (read == -1) throw IOException("EOF while skipping $n bytes")
                    remaining -= read
                }
                else             -> throw IOException("Skip returned negative value")
            }
        }
    }

    /**
     * Extension that keeps reading until the requested number of bytes are filled.
     * Falls back to `read()` when `skip()` returns 0, which happens on some Android
     * streams.
     *
     * @param buf  Destination buffer.
     * @param len  Number of bytes to fill (defaults to `buf.size`).
     * @throws IOException on premature EOF.
     */
    private fun InputStream.readFully(buf: ByteArray, len: Int = buf.size) {
        var off = 0
        while (off < len) {
            val n = read(buf, off, len - off)
            if (n == -1) throw IOException("EOF after $off of $len bytes")
            off += n
        }
    }

    /**
     * Read EXACTLY `n` bytes or throw – never returns a partially‑filled array.
     * This is used for small fixed‑length reads (e.g. 4‑byte type codes).
     *
     * @throws IOException on premature EOF.
     */
    private fun InputStream.readNBytesExact(n: Int) = ByteArray(n).also {
        if (read(it) != n) throw IOException("Unexpected EOF")
    }
}
