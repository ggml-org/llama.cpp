package com.example.llama

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder

/**
 * Arka planda model yanıtı üretilirken süreci canlı tutar ve bildirim gösterir.
 * Basit Binder pattern ile MainActivity'ye bağlanır.
 */
class KovaForegroundService : Service() {

    companion object {
        const val NOTIFICATION_ID  = 1001
        const val CHANNEL_ID       = "kova_generation"
        const val CHANNEL_NAME     = "Kova - Yanıt Üretimi"
    }

    inner class LocalBinder : Binder() {
        fun getService(): KovaForegroundService = this@KovaForegroundService
    }

    private val binder = LocalBinder()
    private lateinit var notificationManager: NotificationManager

    override fun onCreate() {
        super.onCreate()
        notificationManager = getSystemService(NotificationManager::class.java)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int =
        START_NOT_STICKY

    // ── Bildirim kanalı ───────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            CHANNEL_NAME,
            NotificationManager.IMPORTANCE_LOW  // Ses/titreşim yok, sessiz
        ).apply {
            description = "Model yanıt üretirken gösterilir"
            setShowBadge(false)
        }
        notificationManager.createNotificationChannel(channel)
    }

    // ── Bildirim oluşturucular ────────────────────────────────────────────────────

    private fun buildGeneratingNotification(preview: String = ""): Notification {
        val openAppIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val content = if (preview.isNotEmpty())
            preview.take(80) + if (preview.length > 80) "…" else ""
        else
            "Yanıt üretiliyor…"

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Kova")
            .setContentText(content)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(openAppIntent)
            .setOngoing(true)          // Kullanıcı kapatamaz
            .setOnlyAlertOnce(true)    // Sadece ilk gösterimde ses
            .build()
    }

    private fun buildCompletionNotification(preview: String): Notification {
        val openAppIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
            },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Kova — Yanıt hazır")
            .setContentText(preview.take(100) + if (preview.length > 100) "…" else "")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(openAppIntent)
            .setAutoCancel(true)
            .build()
    }

    // ── MainActivity tarafından çağrılan metodlar ─────────────────────────────────

    /** Üretim başladığında çağır — foreground'a geçer, bildirim gösterir */
    fun onGenerationStarted() {
        startForeground(NOTIFICATION_ID, buildGeneratingNotification())
    }

    /** Token geldiğinde çağır — bildirimi günceller (her tokenda değil, aralıklı çağır) */
    fun onTokenUpdate(preview: String) {
        notificationManager.notify(
            NOTIFICATION_ID,
            buildGeneratingNotification(preview)
        )
    }

    /** Üretim bittiğinde çağır */
    fun onGenerationFinished(fullResponse: String, appInForeground: Boolean) {
        if (appInForeground) {
            // Uygulama açıksa bildirimi sessizce kaldır
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            // Uygulama arka plandaysa tamamlanma bildirimi göster
            stopForeground(STOP_FOREGROUND_DETACH)
            notificationManager.notify(
                NOTIFICATION_ID,
                buildCompletionNotification(fullResponse)
            )
        }
        stopSelf()
    }

    /** Üretim iptal edildiğinde çağır */
    fun onGenerationCancelled() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }
}
