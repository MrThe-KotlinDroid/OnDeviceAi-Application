#include <jni.h>
#include <string>
#include <vector>
#include "sentencepiece_processor.h"

// Global instance of the Google SentencePiece processor
sentencepiece::SentencePieceProcessor processor;

extern "C" JNIEXPORT jboolean

JNICALL
Java_com_example_android_1app_TokenizerEngine_load(JNIEnv *env, jobject thiz, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    const auto status = processor.Load(path);
    env->ReleaseStringUTFChars(model_path, path);

    return status.ok() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jintArray

JNICALL
Java_com_example_android_1app_TokenizerEngine_encode(JNIEnv *env, jobject thiz, jstring text) {
    const char *str = env->GetStringUTFChars(text, nullptr);
    std::vector<int> ids;

    // Call Google's C++ Tokenizer
    processor.Encode(str, &ids);
    env->ReleaseStringUTFChars(text, str);

    // Convert C++ vector back to Kotlin IntArray
    jintArray result = env->NewIntArray(ids.size());
    env->SetIntArrayRegion(result, 0, ids.size(), ids.data());
    return result;
}

extern "C" JNIEXPORT jstring

JNICALL
Java_com_example_android_1app_TokenizerEngine_decode(JNIEnv *env, jobject thiz, jintArray ids) {
    jsize len = env->GetArrayLength(ids);
    jint *body = env->GetIntArrayElements(ids, nullptr);

    std::vector<int> id_vector(body, body + len);
    env->ReleaseIntArrayElements(ids, body, JNI_ABORT);

    std::string text;
    // Call Google's C++ Detokenizer
    processor.Decode(id_vector, &text);

    // Convert C++ string back to Kotlin String
    return env->NewStringUTF(text.c_str());
}