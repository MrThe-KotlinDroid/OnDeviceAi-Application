package com.example.android_app

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

class InferenceManager(private val context: Context) {
    private val ortEnv = OrtEnvironment.getEnvironment()
    private var encoderSession: OrtSession? = null
    private var decoderSession: OrtSession? = null

    private fun extractModel(fileName: String): String {
        val file = File(context.filesDir, fileName)
        if (!file.exists()) {
            context.assets.open(fileName).use { type ->
                FileOutputStream(file).use { output ->
                    type.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    private fun createDirectLongBuffer(array: LongArray): LongBuffer {
        val buffer = ByteBuffer.allocateDirect(array.size * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
        buffer.put(array)
        buffer.rewind()
        return buffer
    }

    fun initEncoder() {
        try {
            val path = extractModel("encoder_model_int8.onnx")
            encoderSession = ortEnv.createSession(path, OrtSession.SessionOptions())
        } catch (e: Exception) {
            Log.e("NEW_TAG", "Error initializing encoder: ${e.message}")
        }
    }

    fun initDecoder() {
        try {
            val path = extractModel("decoder_model_merged_int8.onnx")
            decoderSession = ortEnv.createSession(path, OrtSession.SessionOptions())
        } catch (e: Exception) {
            Log.e("NEW_TAG", "Error initializing decoder: ${e.message}")
        }
    }

    fun createEmptyCache(): MutableMap<String, OnnxTensor> {
        val cache = mutableMapOf<String, OnnxTensor>()
        // The shape MUST remain 0 for all dimensions initially
        val shape = longArrayOf(1, 8, 0, 64)

        for (i in 0..5) {
            val decKBuffer = ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val decVBuffer = ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val encKBuffer = ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer()
            val encVBuffer = ByteBuffer.allocateDirect(0).order(ByteOrder.nativeOrder()).asFloatBuffer()

            cache["past_key_values.$i.decoder.key"] = OnnxTensor.createTensor(ortEnv, decKBuffer, shape)
            cache["past_key_values.$i.decoder.value"] = OnnxTensor.createTensor(ortEnv, decVBuffer, shape)
            cache["past_key_values.$i.encoder.key"] = OnnxTensor.createTensor(ortEnv, encKBuffer, shape)
            cache["past_key_values.$i.encoder.value"] = OnnxTensor.createTensor(ortEnv, encVBuffer, shape)
        }
        return cache
    }

    fun runEncoderSanityCheck() {
        val session = encoderSession ?: return
        try {
            val shape = longArrayOf(1, 3)
            val inputIdsTensor = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(longArrayOf(0L, 1L, 2L)), shape)
            val attentionMaskTensor = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(longArrayOf(1L, 1L, 1L)), shape)

            val inputs = mapOf("input_ids" to inputIdsTensor, "attention_mask" to attentionMaskTensor)
            val startTime = System.nanoTime()
            session.run(inputs).use { result ->
                val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0
                val info = result.get(0).info as TensorInfo
                Log.d("NEW_TAG", "Encoder Output Shape: ${info.shape.contentToString()}")
                Log.d("NEW_TAG", "Inference Latency: ${String.format("%.2f", latencyMs)} ms")
            }
        } catch (e: Exception) {
            Log.e("NEW_TAG", "Error in encoder sanity check: ${e.message}")
        }
    }

    fun runDecoderSanityCheck() {
        val session = decoderSession ?: return
        try {
            val cache = createEmptyCache()
            val inputs = mutableMapOf<String, OnnxTensor>()
            inputs.putAll(cache)

            inputs["input_ids"] = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(longArrayOf(0L)), longArrayOf(1, 1))
            val dummyEncBuffer = ByteBuffer.allocateDirect(1 * 3 * 512 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            inputs["encoder_hidden_states"] = OnnxTensor.createTensor(ortEnv, dummyEncBuffer, longArrayOf(1, 3, 512))
            inputs["encoder_attention_mask"] = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(longArrayOf(1L, 1L, 1L)), longArrayOf(1, 3))

            val useCacheBuffer = ByteBuffer.allocateDirect(1).order(ByteOrder.nativeOrder())
            useCacheBuffer.put(0.toByte()).rewind()
            inputs["use_cache_branch"] = OnnxTensor.createTensor(ortEnv, useCacheBuffer, longArrayOf(1), ai.onnxruntime.OnnxJavaType.BOOL)

            session.run(inputs).use {
                Log.d("NEW_TAG", "Decoder initialized and KV Cache prepared.")
            }
        } catch (e: Exception) {
            Log.e("NEW_TAG", "Error in decoder sanity check: ${e.message}")
        }
    }

    fun generateSummary(inputTokenIds: LongArray): List<Long> {
        val generatedTokens = mutableListOf<Long>(0L)
        val inputShape = longArrayOf(1, inputTokenIds.size.toLong())

        val inputIdsTensor = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(inputTokenIds), inputShape)
        val attentionMaskArray = LongArray(inputTokenIds.size) { 1L }
        val attentionMaskTensor = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(attentionMaskArray), inputShape)

        val encoderInputs = mapOf("input_ids" to inputIdsTensor, "attention_mask" to attentionMaskTensor)
        val encoderOutput = encoderSession?.run(encoderInputs)
        val encoderResultTensor = encoderOutput?.get(0) as? OnnxTensor ?: return emptyList()

        // Detach C++ Memory
        val rawBuffer = encoderResultTensor.floatBuffer
        val detachedBuffer = ByteBuffer.allocateDirect(rawBuffer.capacity() * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
        rawBuffer.rewind()
        detachedBuffer.put(rawBuffer)
        detachedBuffer.rewind()

        val encoderHiddenStates = OnnxTensor.createTensor(ortEnv, detachedBuffer, longArrayOf(1, inputTokenIds.size.toLong(), 512))

        var currentCache = createEmptyCache()

        // The ultimate fix: Hold onto all previous steps to prevent GC from killing the Encoder cache
        val allResults = mutableListOf<OrtSession.Result>()
        var isFirstStep = true

        try {
            for (step in 0 until 50) {
                val inputs = mutableMapOf<String, OnnxTensor>()
                inputs["encoder_hidden_states"] = encoderHiddenStates

                val encoderAttentionMask = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(attentionMaskArray), inputShape)
                inputs["encoder_attention_mask"] = encoderAttentionMask
                inputs.putAll(currentCache)

                val useCacheVal = !isFirstStep
                val decoderInputIds = if (isFirstStep) generatedTokens.toLongArray() else longArrayOf(generatedTokens.last())
                val decInputIdsTensor = OnnxTensor.createTensor(ortEnv, createDirectLongBuffer(decoderInputIds), longArrayOf(1, decoderInputIds.size.toLong()))
                inputs["input_ids"] = decInputIdsTensor

                val useCacheBuffer = ByteBuffer.allocateDirect(1).order(ByteOrder.nativeOrder())
                useCacheBuffer.put((if (useCacheVal) 1 else 0).toByte()).rewind()
                val useCacheTensor = OnnxTensor.createTensor(ortEnv, useCacheBuffer, longArrayOf(1), ai.onnxruntime.OnnxJavaType.BOOL)
                inputs["use_cache_branch"] = useCacheTensor

                val decoderResult = decoderSession?.run(inputs) ?: break

                var logitsData: FloatBuffer? = null
                var seqLen = 0
                var vocabSize = 0
                for (entry in decoderResult) {
                    if (entry.key == "logits") {
                        val t = entry.value as OnnxTensor
                        logitsData = t.floatBuffer
                        seqLen = t.info.shape[1].toInt()
                        vocabSize = t.info.shape[2].toInt()
                        break
                    }
                }

                if (logitsData == null) break

                val lastTokenOffset = (seqLen - 1) * vocabSize
                var maxLogit = -Float.MAX_VALUE
                var nextTokenId = 0L
                for (i in 0 until vocabSize) {
                    val logit = logitsData.get(lastTokenOffset + i)
                    if (logit > maxLogit) {
                        maxLogit = logit
                        nextTokenId = i.toLong()
                    }
                }

                generatedTokens.add(nextTokenId)

                val newCache = mutableMapOf<String, OnnxTensor>()
                for (entry in decoderResult) {
                    if (entry.key.startsWith("present")) {
                        val newKey = entry.key.replace("present", "past_key_values")
                        
                        // CRITICAL FIX: Only update the encoder cache on the first step.
                        // On subsequent steps, inherit the valid static encoder cache from the previous loop.
                        if (isFirstStep || newKey.contains("decoder")) {
                            newCache[newKey] = entry.value as OnnxTensor
                        } else {
                            newCache[newKey] = currentCache[newKey]!!
                        }
                    }
                }

                decInputIdsTensor.close()
                useCacheTensor.close()
                encoderAttentionMask.close()

                if (isFirstStep) {
                    currentCache.values.forEach { it.close() }
                    isFirstStep = false
                }

                currentCache = newCache
                allResults.add(decoderResult)

                if (nextTokenId == 1L) break
            }
        } catch (e: Exception) {
            Log.e("NEW_TAG", "Error during generation step: ${e.message}", e)
        } finally {
            allResults.forEach { it.close() }
            encoderOutput?.close()
            encoderHiddenStates.close()
            inputIdsTensor.close()
            attentionMaskTensor.close()
        }

        return generatedTokens
    }
}