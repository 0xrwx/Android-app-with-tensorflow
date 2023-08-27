package com.example.coursework

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import com.example.coursework.ui.theme.CourseworkTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

import android.content.ContentResolver
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp

import com.example.coursework.ml.ModelUnquantTwo
import java.io.InputStream

const val imageSize = 224

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            CourseworkTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ImageImportScreen()
                }
            }
        }
    }
}

@Composable
fun ImageImportScreen() {
    var selectedImageBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val context = LocalContext.current

    val getContent = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val contentResolver: ContentResolver = context.contentResolver
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            inputStream?.use {
                selectedImageBitmap = BitmapFactory.decodeStream(it)
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Button(onClick = { getContent.launch("image/*") }) {
            Text("Select Image from Gallery")
        }

        Spacer(modifier = Modifier.height(16.dp))

        selectedImageBitmap?.let { bitmap ->
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Selected Image",
                modifier = Modifier.size(200.dp)
            )
            ImageClassifier(bitmap)
        }
    }
}

fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
    // Assuming the model takes a 224x224 RGB image
    val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
    byteBuffer.order(ByteOrder.nativeOrder())
    val intValues = IntArray(imageSize * imageSize)
    bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    var pixel = 0
    for (i in 0 until imageSize) {
        for (j in 0 until imageSize) {
            val value = intValues[pixel++]
            byteBuffer.putFloat((Color.red(value) - 127.5f) / 127.5f)
            byteBuffer.putFloat((Color.green(value) - 127.5f) / 127.5f)
            byteBuffer.putFloat((Color.blue(value) - 127.5f) / 127.5f)
        }
    }
    return byteBuffer
}

fun classifyImage(context: Context, bitmap: Bitmap): String {
    // val bitmap: Bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.cherry_4)
    // Resize to match the model's expected input size
    val resizedBitmap: Bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
    val byteBuffer:  ByteBuffer = bitmapToByteBuffer(resizedBitmap)

    val model = ModelUnquantTwo.newInstance(context)
    val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
    inputFeature0.loadBuffer(byteBuffer)

    val outputs = model.process(inputFeature0)
    val results = outputs.outputFeature0AsTensorBuffer.floatArray

    model.close()

//    return if (results[0] > 0.5) "Rose ${(results[0] * 100).toInt()}%"
//    else "Barbados Cherry ${(results[1] * 100).toInt()}%"
    return "results[0]: ${results[0]}, results[1]: ${results[1]}"
}

@Composable
fun ImageClassifier(bitmap: Bitmap) {
    val context = LocalContext.current
    // val bitmap: Bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.cherry_4)
    val classificationResult = classifyImage(context, bitmap)

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text(text = "It's an $classificationResult")
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    CourseworkTheme {
        ImageImportScreen()
    }
}