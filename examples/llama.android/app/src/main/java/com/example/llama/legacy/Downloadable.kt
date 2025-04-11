package com.example.llama.legacy

import android.app.DownloadManager
import android.net.Uri
import android.util.Log
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.core.database.getLongOrNull
import androidx.core.net.toUri
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

data class Downloadable(val name: String, val source: Uri, val destination: File) {
    companion object {
        @JvmStatic
        private val tag: String? = this::class.qualifiedName

        sealed interface State
        data object Ready: State
        data class Downloading(val id: Long): State
        data class Downloaded(val downloadable: Downloadable): State
        data class Error(val message: String): State

        @JvmStatic
        @Composable
        fun Button(viewModel: LegacyViewModel, dm: DownloadManager, item: Downloadable) {
            var status: State by remember {
                mutableStateOf(
                    if (item.destination.exists()) Downloaded(item)
                    else Ready
                )
            }
            var progress by remember { mutableDoubleStateOf(0.0) }

            val coroutineScope = rememberCoroutineScope()

            suspend fun waitForDownload(result: Downloading, item: Downloadable): State {
                while (true) {
                    val cursor = dm.query(DownloadManager.Query().setFilterById(result.id))

                    if (cursor == null) {
                        Log.e(tag, "dm.query() returned null")
                        return Error("dm.query() returned null")
                    }

                    if (!cursor.moveToFirst() || cursor.count < 1) {
                        cursor.close()
                        Log.i(tag, "cursor.moveToFirst() returned false or cursor.count < 1, download canceled?")
                        return Ready
                    }

                    val pix = cursor.getColumnIndex(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)
                    val tix = cursor.getColumnIndex(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)
                    val sofar = cursor.getLongOrNull(pix) ?: 0
                    val total = cursor.getLongOrNull(tix) ?: 1
                    cursor.close()

                    if (sofar == total) {
                        return Downloaded(item)
                    }

                    progress = (sofar * 1.0) / total

                    delay(1000L)
                }
            }

            fun onClick() {
                when (val s = status) {
                    is Downloaded -> {
                        viewModel.load(item.destination.path)
                    }

                    is Downloading -> {
                        coroutineScope.launch {
                            status = waitForDownload(s, item)
                        }
                    }

                    else -> {
                        item.destination.delete()

                        val request = DownloadManager.Request(item.source).apply {
                            setTitle("Downloading model")
                            setDescription("Downloading model: ${item.name}")
                            setAllowedNetworkTypes(DownloadManager.Request.NETWORK_WIFI)
                            setDestinationUri(item.destination.toUri())
                        }

                        viewModel.log("Saving ${item.name} to ${item.destination.path}")
                        Log.i(tag, "Saving ${item.name} to ${item.destination.path}")

                        val id = dm.enqueue(request)
                        status = Downloading(id)
                        onClick()
                    }
                }
            }

            Button(onClick = { onClick() }, enabled = status !is Downloading) {
                when (status) {
                    is Downloading -> Text(text = "Downloading ${(progress * 100).toInt()}%")
                    is Downloaded -> Text("Load ${item.name}")
                    is Ready -> Text("Download ${item.name}")
                    is Error -> Text("Download ${item.name}")
                }
            }
        }

    }
}
