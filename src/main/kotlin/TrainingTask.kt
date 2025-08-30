package org.cuttlefish

import kotlinx.serialization.Serializable

@Deprecated("Its like not the vibe")
@Serializable
enum class TrainingTask {
    ODD_EVEN,
    COUNT_ONES
    // Add more tasks here as you implement them, e.g., IS_PRIME, DIVISIBLE_BY_N
}