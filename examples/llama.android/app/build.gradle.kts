plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.symbol.processing)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.jetbrains.kotlin.compose.compiler)
    alias(libs.plugins.jetbrains.kotlin.serialization)
    alias(libs.plugins.hilt)
}

android {
    namespace = "com.arm.aiplayground"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.arm.aiplayground"

        minSdk = 33
        targetSdk = 36

        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        jvmToolchain(17)

        compileOptions {
            targetCompatibility = JavaVersion.VERSION_17
        }
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
}

dependencies {
    // Platform & Bundles
    implementation(platform(libs.compose.bom))
    implementation(libs.bundles.androidx)
    ksp(libs.androidx.room.compiler)
    implementation(libs.bundles.compose)
    implementation(libs.bundles.kotlinx)
    ksp(libs.hilt.android.compiler)
    implementation(libs.bundles.hilt)
    implementation(libs.bundles.retrofit)

    // Subproject
    implementation(project(":lib"))

    debugImplementation(libs.bundles.debug)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.bundles.testing)
}
