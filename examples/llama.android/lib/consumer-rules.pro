-keep class com.arm.aichat.* { *; }
-keep class com.arm.aichat.gguf.* { *; }

-keepclasseswithmembernames class * {
    native <methods>;
}

-keep class kotlin.Metadata { *; }
