package com.example.llama.revamp.util

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import kotlinx.coroutines.yield
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.nio.channels.ReadableByteChannel
import java.nio.channels.WritableByteChannel
import java.util.Locale

/**
 * Convert bytes into human readable sizes
 */
fun formatFileByteSize(sizeInBytes: Long) = when {
    sizeInBytes >= 1_000_000_000 -> {
        val sizeInGb = sizeInBytes / 1_000_000_000.0
        String.format(Locale.getDefault(), "%.1f GB", sizeInGb)
    }
    sizeInBytes >= 1_000_000 -> {
        val sizeInMb = sizeInBytes / 1_000_000.0
        String.format(Locale.getDefault(), "%.0f MB", sizeInMb)
    }
    else -> {
        val sizeInKb = sizeInBytes / 1_000.0
        String.format(Locale.getDefault(), "%.0f KB", sizeInKb)
    }
}

/**
 * Formats numbers to human-readable form (K, M)
 */
fun formatContextLength(contextLength: Int): String {
    return when {
        contextLength >= 1_000_000 -> String.format(Locale.getDefault(), "%.1fM", contextLength / 1_000_000.0)
        contextLength >= 1_000 -> String.format(Locale.getDefault(), "%.0fK", contextLength / 1_000.0)
        else -> contextLength.toString()
    }
}

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


suspend fun copyWithChannels(
    input: InputStream,
    output: OutputStream,
    totalSize: Long,
    bufferSize: Int,
    yieldSize: Int,
    onProgress: (suspend (Float) -> Unit)?
) {
    val inChannel: ReadableByteChannel = Channels.newChannel(input)
    val outChannel: WritableByteChannel = Channels.newChannel(output)

    val buffer = ByteBuffer.allocateDirect(bufferSize)
    var totalBytesRead = 0L

    while (inChannel.read(buffer) != -1) {
        buffer.flip()
        while (buffer.hasRemaining()) {
            outChannel.write(buffer)
        }
        totalBytesRead += buffer.position()
        buffer.clear()

        // Report progress
        onProgress?.invoke(totalBytesRead.toFloat() / totalSize)

        if (totalBytesRead % (yieldSize) == 0L) {
            yield()
        }
    }

    outChannel.close()
    inChannel.close()
}

suspend fun copyWithBuffer(
    input: InputStream,
    output: OutputStream,
    totalSize: Long,
    bufferSize: Int,
    yieldSize: Int,
    onProgress: (suspend (Float) -> Unit)?
) {
    val bufferedInput = BufferedInputStream(input, bufferSize)
    val bufferedOutput = BufferedOutputStream(output, bufferSize)
    val buffer = ByteArray(bufferSize)

    var bytesRead: Int
    var totalBytesRead = 0L

    while (input.read(buffer).also { bytesRead = it } != -1) {
        output.write(buffer, 0, bytesRead)
        totalBytesRead += bytesRead

        // Report progress
        onProgress?.invoke(totalBytesRead.toFloat() / totalSize)

        // Yield less frequently with larger buffers
        if (totalBytesRead % (yieldSize) == 0L) { // Every 64MB
            yield()
        }
    }

    bufferedOutput.close()
    bufferedInput.close()
}
