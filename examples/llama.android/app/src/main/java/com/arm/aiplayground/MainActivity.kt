package com.arm.aiplayground

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import com.arm.aiplayground.ui.AppContent
import com.arm.aiplayground.ui.theme.LlamaTheme
import com.arm.aiplayground.ui.theme.isDarkTheme
import com.arm.aiplayground.ui.theme.md_theme_dark_scrim
import com.arm.aiplayground.ui.theme.md_theme_light_scrim
import com.arm.aiplayground.viewmodel.SettingsViewModel
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val settingsViewModel: SettingsViewModel = hiltViewModel()
            val colorThemeMode by settingsViewModel.colorThemeMode.collectAsState()
            val darkThemeMode by settingsViewModel.darkThemeMode.collectAsState()

            val isDarkTheme = isDarkTheme(darkThemeMode)
            LlamaTheme(colorThemeMode = colorThemeMode, isDarkTheme = isDarkTheme) {
                DisposableEffect(darkThemeMode) {
                    enableEdgeToEdge(
                        statusBarStyle = SystemBarStyle.auto(
                            android.graphics.Color.TRANSPARENT,
                            android.graphics.Color.TRANSPARENT,
                        ) { isDarkTheme },
                        navigationBarStyle = SystemBarStyle.auto(
                            md_theme_light_scrim.value.toInt(),
                            md_theme_dark_scrim.value.toInt(),
                        ) { isDarkTheme },
                    )
                    onDispose {}
                }

                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppContent(settingsViewModel)
                }
            }
        }
    }
}
