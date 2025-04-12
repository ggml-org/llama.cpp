plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.symbol.processing)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.jetbrains.kotlin.compose.compiler)
}

android {
    namespace = "com.example.llama"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.llama"

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
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
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
    implementation(libs.bundles.coroutines)

    // Individual dependencies
    implementation(libs.accompanist.systemuicontroller)

    // Subproject
    implementation(project(":llama"))

    debugImplementation(libs.bundles.debug)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.compose.bom))
    androidTestImplementation(libs.bundles.testing)
}
