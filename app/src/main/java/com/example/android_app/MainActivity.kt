package com.example.android_app

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.android_app.ui.theme.Android_appTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.Locale

data class InferenceResult(
    val summary: String,
    val latencyMs: Long,
    val tps: Double,
    val memoryMb: Double
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val inferenceManager = InferenceManager(this)
        inferenceManager.initEncoder()
        inferenceManager.initDecoder()

        setContent {
            Android_appTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    SummarizerUI(inferenceManager)
                }
            }
        }
    }
}

@Composable
fun SummarizerUI(inferenceManager: InferenceManager) {
    var userInput by remember { mutableStateOf("") }
    var summaryResult by remember { mutableStateOf("") }
    var latency by remember { mutableLongStateOf(0L) }
    var tps by remember { mutableDoubleStateOf(0.0) }
    var memory by remember { mutableDoubleStateOf(0.0) }
    
    var isLoading by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val context = androidx.compose.ui.platform.LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        TextField(
            value = userInput,
            onValueChange = { userInput = it },
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp),
            placeholder = { Text("Enter text to summarize...") },
            label = { Text("Input Text") }
        )

        Button(
            onClick = {
                if (userInput.isNotBlank()) {
                    scope.launch {
                        isLoading = true
                        val result = runInference(context, inferenceManager, userInput)
                        summaryResult = result.summary
                        latency = result.latencyMs
                        tps = result.tps
                        memory = result.memoryMb
                        isLoading = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isLoading
        ) {
            Text("Generate Summary")
        }

        if (isLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxSize()
            ) {
                Text(
                    text = "Result:",
                    style = MaterialTheme.typography.titleMedium
                )
                
                if (summaryResult.isNotEmpty()) {
                    Text(
                        text = "Latency: $latency ms | TPS: ${String.format(Locale.US, "%.2f", tps)}",
                        style = MaterialTheme.typography.bodySmall
                    )
                    Text(
                        text = "Memory Delta: ${String.format(Locale.US, "%.2f", memory)} MB",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = if (summaryResult.isEmpty() && !isLoading) "Summary will appear here..." else summaryResult,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

fun extractTokenizerModel(context: Context): String {
    val modelFile = File(context.filesDir, "spiece.model")
    if (!modelFile.exists()) {
        context.assets.open("spiece.model").use { input ->
            FileOutputStream(modelFile).use { output ->
                input.copyTo(output)
            }
        }
    }
    return modelFile.absolutePath
}

suspend fun runInference(context: Context, inferenceManager: InferenceManager, userInput: String): InferenceResult {
    return withContext(Dispatchers.Default) {
        val modelPath = extractTokenizerModel(context)
        val tokenizer = TokenizerEngine()

        if (tokenizer.load(modelPath)) {
            val text = "summarize: $userInput"
            val tokens = tokenizer.encode(text).toMutableList()
            tokens.add(1) // Append EOS token
            Log.d("NEW_TAG", "Generated IDs: ${tokens.joinToString(", ")}")

            val benchmark = inferenceManager.generateSummary(tokens.map { it.toLong() }.toLongArray())
            Log.d("NEW_TAG", "AI Output IDs: ${benchmark.generatedIds}")

            val intArray = benchmark.generatedIds.filter { it > 0L }.map { it.toInt() }.toIntArray()
            val englishText = tokenizer.decode(intArray).replace(" ", " ").trim()
            
            InferenceResult(englishText, benchmark.latencyMs, benchmark.tps, benchmark.memoryUsedMb)
        } else {
            InferenceResult("Failed to load tokenizer model.", 0L, 0.0, 0.0)
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SummarizerPreview() {
    Android_appTheme {
        Surface {
            // Mock inference manager for preview if needed, or just a shell
        }
    }
}
