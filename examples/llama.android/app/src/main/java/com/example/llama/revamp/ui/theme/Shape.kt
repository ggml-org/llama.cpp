package com.example.llama.revamp.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

val Shapes = Shapes(
    small = RoundedCornerShape(4.dp),
    medium = RoundedCornerShape(8.dp),
    large = RoundedCornerShape(16.dp)
)

// Additional custom shapes for specific components
val CardShape = RoundedCornerShape(12.dp)
val ButtonShape = RoundedCornerShape(8.dp)
val InputFieldShape = RoundedCornerShape(8.dp)
