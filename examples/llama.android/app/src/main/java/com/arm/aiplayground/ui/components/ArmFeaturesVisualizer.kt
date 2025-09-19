package com.arm.aiplayground.ui.components

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.text.TextAutoSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MultiChoiceSegmentedButtonRow
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import com.arm.aichat.ArmFeature
import com.arm.aichat.ArmFeaturesMapper.DisplayItem
import kotlin.math.sqrt

/**
 * ARM Features visualization using segmented buttons.
 */
@Composable
fun ArmFeaturesVisualizer(
    supportedFeatures: List<DisplayItem>,
    autoSizeMinScaling: Float = 0.5f,
    onFeatureClick: ((ArmFeature) -> Unit)? = null
) {
    // Segmented Button Row for Features
    MultiChoiceSegmentedButtonRow(
        modifier = Modifier.fillMaxWidth()
    ) {
        supportedFeatures.forEachIndexed { index, item ->
            val weight = sqrt(item.feature.displayName.length.toFloat())

            SegmentedButton(
                modifier = Modifier.weight(weight),
                shape = SegmentedButtonDefaults.itemShape(
                    index = index,
                    count = supportedFeatures.size
                ),
                icon = {},
                onCheckedChange = { onFeatureClick?.invoke(item.feature) },
                checked = item.isSupported,
            ) {
                Text(
                    text = item.feature.displayName,
                    autoSize = TextAutoSize.StepBased(
                        minFontSize = MaterialTheme.typography.labelSmall.fontSize * autoSizeMinScaling,
                        maxFontSize = MaterialTheme.typography.labelSmall.fontSize
                    ),
                    style = MaterialTheme.typography.labelSmall,
                    maxLines = 1,
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
