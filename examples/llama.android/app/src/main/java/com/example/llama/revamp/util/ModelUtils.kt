package com.example.llama.revamp.util

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.util.Locale

/**
 * Gets the file name from a content URI
 */
fun getFileNameFromUri(context: Context, uri: Uri): String? =
    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME).let { nameIndex ->
                if (nameIndex != -1) cursor.getString(nameIndex) else null
            }
        } else {
            null
        }
    } ?: uri.lastPathSegment

/**
 * Gets the file size from a content URI
 */
fun getFileSizeFromUri(context: Context, uri: Uri): Long? =
    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
        if (cursor.moveToFirst()) {
            cursor.getColumnIndex(OpenableColumns.SIZE).let { sizeIndex ->
                if (sizeIndex != -1) cursor.getLong(sizeIndex) else null
            }
        } else {
            null
        }
    }

/**
 * Try to extract parameters by looking for patterns like 7B, 13B, etc.
 */
fun extractParametersFromFilename(filename: String): String? =
    Regex("([0-9]+(\\.[0-9]+)?)[bB]").find(filename)?.value?.uppercase()

/**
 * Try to extract quantization by looking for patterns like Q4_0, Q5_K_M, etc.
 */
fun extractQuantizationFromFilename(filename: String) =
    listOf(
        Regex("[qQ][0-9]_[0-9]"),
        Regex("[qQ][0-9]_[kK]_[mM]"),
        Regex("[qQ][0-9]_[kK]"),
        Regex("[qQ][0-9][fF](16|32)")
    ).firstNotNullOfOrNull {
        it.find(filename)?.value?.uppercase()
    }

/**
 * Try to extract model type (Llama, Mistral, etc.)
 *
 * TODO-han.yin: Replace with GGUF parsing, also to be moved into the util object
 */
fun extractModelTypeFromFilename(filename: String): String? {
    val lowerFilename = filename.lowercase()
    return listOf("llama", "mistral", "phi", "qwen", "falcon", "mpt")
        .firstNotNullOfOrNull { type ->
            if (lowerFilename.contains(type)) {
                type.replaceFirstChar {
                    if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString()
                }
            } else { null }
        }
}
