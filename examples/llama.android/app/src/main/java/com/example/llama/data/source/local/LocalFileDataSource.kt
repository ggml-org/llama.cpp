package com.example.llama.data.source.local

import android.content.Context
import android.net.Uri
import android.util.Log
import com.example.llama.data.source.local.LocalFileDataSource.FileInfo
import com.example.llama.util.copyWithBuffer
import com.example.llama.util.copyWithChannels
import com.example.llama.util.getFileNameFromUri
import com.example.llama.util.getFileSizeFromUri
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton

interface LocalFileDataSource {
    /**
     * Copy local file from [sourceUri] into [destinationFile]
     */
    suspend fun copyFile(
        sourceUri: Uri,
        destinationFile: File,
        fileSize: Long,
        onProgress: (suspend (Float) -> Unit)? = null
    ): Result<File>

    /**
     * Obtain the file name and size from given [uri]
     */
    suspend fun getFileInfo(uri: Uri): FileInfo?

    /**
     * Clean up incomplete file due to unfinished import
     */
    suspend fun cleanupPartialFile(file: File): Boolean

    data class FileInfo(val name: String, val size: Long)
}

@Singleton
class LocalFileDataSourceImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : LocalFileDataSource {

    override suspend fun copyFile(
        sourceUri: Uri,
        destinationFile: File,
        fileSize: Long,
        onProgress: (suspend (Float) -> Unit)?
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.contentResolver.openInputStream(sourceUri)
                ?: return@withContext Result.failure(IOException("Failed to open input stream"))
            val outputStream = FileOutputStream(destinationFile)

            if (fileSize > LARGE_MODEL_THRESHOLD_SIZE) {
                // Use NIO channels for large models
                Log.i(TAG, "Copying ${destinationFile.name} (size: $fileSize) via NIO...")
                copyWithChannels(
                    input = inputStream,
                    output = outputStream,
                    totalSize = fileSize,
                    bufferSize = NIO_BUFFER_SIZE,
                    yieldSize = NIO_YIELD_SIZE,
                    onProgress = onProgress
                )
            } else {
                // Default copy with buffer for small models
                Log.i(TAG, "Copying ${destinationFile.name} (size: $fileSize) via buffer...")
                copyWithBuffer(
                    input = inputStream,
                    output = outputStream,
                    totalSize = fileSize,
                    bufferSize = DEFAULT_BUFFER_SIZE,
                    yieldSize = DEFAULT_YIELD_SIZE,
                    onProgress = onProgress
                )
            }

            Result.success(destinationFile)
        } catch (e: Exception) {
            if (destinationFile.exists()) {
                destinationFile.delete()
            }
            Result.failure(e)
        }
    }

    override suspend fun getFileInfo(uri: Uri): FileInfo? {
        val name = getFileNameFromUri(context, uri)
        val size = getFileSizeFromUri(context, uri)
        return if (name != null && size != null) {
            FileInfo(name, size)
        } else null
    }

    override suspend fun cleanupPartialFile(file: File): Boolean =
        try {
            if (file.exists()) file.delete() else false
        } catch (e: Exception) {
            Log.e(TAG, "Failed to delete file: ${e.message}")
            false
        }

    companion object {
        private val TAG = LocalFileDataSourceImpl::class.java.simpleName

        private const val LARGE_MODEL_THRESHOLD_SIZE = 1024 * 1024 * 1024
        private const val NIO_BUFFER_SIZE = 32 * 1024 * 1024
        private const val NIO_YIELD_SIZE = 128 * 1024 * 1024
        private const val DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
        private const val DEFAULT_YIELD_SIZE = 16 * 1024 * 1024
    }
}
