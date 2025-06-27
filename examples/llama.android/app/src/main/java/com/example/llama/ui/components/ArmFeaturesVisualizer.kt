package com.example.llama.ui.components

import android.content.Intent
import android.llama.cpp.ArmFeature
import android.llama.cpp.ArmFeaturesMapper
import android.llama.cpp.LLamaTier
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MultiChoiceSegmentedButtonRow
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.core.net.toUri
import kotlin.math.sqrt

/**
 * ARM Features visualization using segmented buttons.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ArmFeaturesVisualizer(
    detectedTier: LLamaTier?,
    modifier: Modifier = Modifier,
    onFeatureClick: ((ArmFeature) -> Unit)? = null
) {
    val featuresData = remember(detectedTier) {
        ArmFeaturesMapper.getFeatureDisplayData(detectedTier)
    }

    Column(
        modifier = modifier.fillMaxWidth()
    ) {
        // Segmented Button Row for Features
        MultiChoiceSegmentedButtonRow(
            modifier = Modifier.fillMaxWidth()
        ) {
            featuresData.forEachIndexed { index, item ->
                val weight = sqrt(item.feature.displayName.length.toFloat())

                SegmentedButton(
                    modifier = Modifier.weight(weight),
                    shape = SegmentedButtonDefaults.itemShape(
                        index = index,
                        count = featuresData.size
                    ),
                    icon = {},
                    onCheckedChange = { onFeatureClick?.invoke(item.feature) },
                    checked = item.isSupported,
                ) {
                    Text(
                        text = item.feature.displayName,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = if (item.isSupported) {
                            FontWeight.Medium
                        } else {
                            FontWeight.Light
                        },
                        color = if (item.isSupported) {
                            MaterialTheme.colorScheme.onSurface
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        }
                    )
                }
            }
        }
    }
}

/**
 * Alternative version with clickable features that open ARM documentation.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ArmFeaturesVisualizerClickable(
    detectedTier: LLamaTier?,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    ArmFeaturesVisualizer(
        detectedTier = detectedTier,
        modifier = modifier,
        onFeatureClick = { feature ->
            // Open ARM documentation in browser
            val intent = Intent(Intent.ACTION_VIEW, feature.armDocUrl.toUri())
            context.startActivity(intent)
        }
    )
}
