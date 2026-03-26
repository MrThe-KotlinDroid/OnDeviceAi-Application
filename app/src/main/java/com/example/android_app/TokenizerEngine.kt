package com.example.android_app

class TokenizerEngine {

    // This loads the compiled C++ library into memory
    init {
        System.loadLibrary("tokenizer_bridge")
    }

    // The three functions demanded by the Protocol
    external fun load(modelPath: String): Boolean
    external fun encode(text: String): IntArray
    external fun decode(ids: IntArray): String
}