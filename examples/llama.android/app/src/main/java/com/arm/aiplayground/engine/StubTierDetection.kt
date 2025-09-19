package com.arm.aiplayground.engine

import com.arm.aichat.LLamaTier
import com.arm.aichat.TierDetection
import android.util.Log

/**
 * A stub [TierDetection] for agile development & testing
 */
object StubTierDetection : TierDetection {
    private val tag = StubTierDetection::class.java.simpleName

    override fun getDetectedTier(): LLamaTier? = LLamaTier.T3

    override fun clearCache() {
        Log.d(tag, "Cache cleared")
    }
}
