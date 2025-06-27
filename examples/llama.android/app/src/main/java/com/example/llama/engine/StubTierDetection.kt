package com.example.llama.engine

import android.llama.cpp.LLamaTier
import android.llama.cpp.TierDetection
import android.util.Log

/**
 * A stub [TierDetection] for agile development & testing
 */
object StubTierDetection : TierDetection {
    private val tag = StubTierDetection::class.java.simpleName

    override val detectedTier: LLamaTier?
        get() = LLamaTier.T2

    override fun clearCache() {
        Log.d(tag, "Cache cleared")
    }
}
