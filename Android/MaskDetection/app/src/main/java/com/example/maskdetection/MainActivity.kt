package com.example.maskdetection


import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.MenuItem
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.androidnetworking.AndroidNetworking
import com.androidnetworking.common.Priority
import com.androidnetworking.error.ANError
import com.androidnetworking.interceptors.HttpLoggingInterceptor
import com.androidnetworking.interfaces.StringRequestListener
import com.google.android.gms.vision.Frame
import com.google.android.gms.vision.face.FaceDetector
import com.otaliastudios.cameraview.CameraView
import okhttp3.*
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.text.SimpleDateFormat
import java.util.*


class MainActivity : AppCompatActivity() {
    private lateinit var model: Interpreter
    private lateinit var modelFile: MappedByteBuffer
    lateinit var cameraView: CameraView
    lateinit var overlayView: OverlayView
    lateinit var toolbar: Toolbar
    val compatList = CompatibilityList()
    private lateinit var labels :List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        toolbar = findViewById(R.id.toolbar)

        setSupportActionBar(toolbar)
//        supportActionBar?.setHomeButtonEnabled(true)


        cameraView = findViewById(R.id.cameraView)
        overlayView = findViewById(R.id.overlayView)

        cameraView.setLifecycleOwner(this)

        // load model
        modelFile = FileUtil.loadMappedFile(this, "model_v2.tflite")
        labels = FileUtil.loadLabels(this, "labels_v2.txt")

        model = Interpreter(modelFile, Interpreter.Options()
            .apply {
                if (compatList.isDelegateSupportedOnThisDevice) {
                    // if the device has a supported GPU, add the GPU delegate
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    // if the GPU is not supported, run on 4 threads
                    this.setNumThreads(4)
                }
            })

        // Create a FaceDetector
        val faceDetector = FaceDetector.Builder(this).setTrackingEnabled(true)
            .build()
        if (!faceDetector.isOperational) {
            AlertDialog.Builder(this)
                .setMessage("Could not set up the face detector!")
                .show()
        }

        cameraView.addFrameProcessor { frame ->

            val matrix = Matrix()
            matrix.setRotate(frame.rotationToUser.toFloat())

            if (frame.dataClass === ByteArray::class.java) {

                Log.e("TAG", "onCreate: frame callback")
                val out = ByteArrayOutputStream()
                val yuvImage = YuvImage(
                    frame.getData(),
                    ImageFormat.NV21,
                    frame.size.width,
                    frame.size.height,
                    null
                )

                yuvImage.compressToJpeg(
                    Rect(0, 0, frame.size.width, frame.size.height), 100, out
                )

                val imageBytes = out.toByteArray()
                var bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

                bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                bitmap =
                    Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                bitmap =
                    Bitmap.createScaledBitmap(bitmap, overlayView.width, overlayView.height, true)

                overlayView.boundingBox = processBitmap(bitmap, faceDetector)
                overlayView.invalidate()
            } else {
                Toast.makeText(this, "Camera Data not Supported", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            android.R.id.home -> onBackPressed()
        }
        return super.onOptionsItemSelected(item)
    }


    var jsonArray = JSONArray()
    private fun processBitmap(bitmap: Bitmap, faceDetector: FaceDetector): MutableList<Box> {
        val boundingBoxList = mutableListOf<Box>()

        // Detect the faces
        val frame = Frame.Builder().setBitmap(bitmap).build()
        val faces = faceDetector.detect(frame)


        // Mark out the identified face
        for (i in 0 until faces.size()) {
            val thisFace = faces.valueAt(i)
            val left = thisFace.position.x
            val top = thisFace.position.y
            val right = left + thisFace.width
            val bottom = top + thisFace.height
            val bitmapCropped = Bitmap.createBitmap(
                bitmap,
                left.toInt(),
                top.toInt(),
                if (right.toInt() > bitmap.width) {
                    bitmap.width - left.toInt()
                } else {
                    thisFace.width.toInt()
                },
                if (bottom.toInt() > bitmap.height) {
                    bitmap.height - top.toInt()
                } else {
                    thisFace.height.toInt()
                }
            )


            val label = predict(bitmapCropped)


            var predictionn = ""

            val label1 = label["Mask"] ?: 0F
            val label2 = label["No Mask"] ?: 0F
            val label3 = label["Covered Mouth Chin"] ?: 0F
            val label4 = label["Covered Nose Mouth"] ?: 0F

            val map = mapOf(
                LABEL.Mask to label1,
                LABEL.No_Mask to label2,
                LABEL.Covered_Mouth_Chin to label3,
                LABEL.Covered_Nose_Mouth to label4
            )

            val maxValue = map.maxOf { it.value }
            val labelType = map.filterValues { it == maxValue }.keys.elementAt(0)


            predictionn = when (labelType) {
                LABEL.Mask -> "With Mask : " + String.format("%.1f", label1 * 100) + "%"
                LABEL.No_Mask -> "Without Mask : " + String.format("%.1f", label2 * 100) + "%"
                LABEL.Covered_Mouth_Chin -> "Covered Mouth Chin : " + String.format(
                    "%.1f",
                    label2 * 100
                ) + "%"
                LABEL.Covered_Nose_Mouth -> "Covered Nose Mouth : " + String.format(
                    "%.1f",
                    label2 * 100
                ) + "%"
            }


            val json = JSONObject().apply {
                put("timestamp", System.currentTimeMillis().dateWithFormat("dd-MM-yyyy_HH-mm-ss"))
                put("status", labelType.name.replace("_", " "))
                put("total_number", faces.size())
            }
            jsonArray.put(json)
            Log.e("TAG", "processBitmap: size >> ${jsonArray.length()}")
            if (jsonArray.length() >= 10) {
                dumpDataLog(jsonArray)
                jsonArray = JSONArray()
            }

//            if (with > without){
//                predictionn = "With Mask : " + String.format("%.1f", with*100) + "%"
//            } else {
//                predictionn = "Without Mask : " + String.format("%.1f", without*100) + "%"
//            }
            boundingBoxList.add(Box(RectF(left, top, right, bottom), predictionn, labelType))
        }
        return boundingBoxList
    }

    private fun dumpDataLog(jsonArray: JSONArray) {
        val json = JSONObject().apply {
            put("array", JSONArray(jsonArray.toString()))
        }

        Log.e("TAG", "dumpDataLog: data >> $json")

        val client = OkHttpClient().newBuilder()
            .build()
        val mediaType = MediaType.parse("application/json")
        val body = RequestBody.create(
            mediaType,
            json.toString()
        )

        val request: Request = Request.Builder()
            .url("https://gsheet-data.herokuapp.com/post_json")
            .method("POST", body)
            .addHeader("Content-Type", "application/json")
            .build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                e.printStackTrace()
            }

            override fun onResponse(call: Call, response: Response) {
                Log.e("call", "onResponse: ${response.body()?.string()}")
            }
        })

        /*AndroidNetworking.post("https://gsheet-data.herokuapp.com/post_json")
            .addApplicationJsonBody(json)
            .setTag("test")
            .addHeaders("Content-Type", "application/json")
            .setPriority(Priority.HIGH)
            .setOkHttpClient(
                OkHttpClient.Builder()
                    .addInterceptor(HttpLoggingInterceptor().setLevel(HttpLoggingInterceptor.Level.BODY))
                    .build()
            )
            .build()
            .getAsString(object : StringRequestListener {
                override fun onResponse(response: String?) {
                    Log.e("TAG", "response >> $response")
                }

                override fun onError(anError: ANError?) {
                    Log.e("TAG", "onError: ${anError?.errorCode}")
                    anError?.printStackTrace()
                }
            })*/
    }


    private fun predict(input: Bitmap): MutableMap<String, Float> {

        // data type
        val imageDataType = model.getInputTensor(0).dataType()
        val inputShape = model.getInputTensor(0).shape()

        val outputDataType = model.getOutputTensor(0).dataType()
        val outputShape = model.getOutputTensor(0).shape()

        var inputImageBuffer = TensorImage(imageDataType)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)

        // preprocess
        val cropSize = kotlin.math.min(input.width, input.height)
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(NormalizeOp(127.5f, 127.5f))
            .build()

        // load image
        inputImageBuffer.load(input)
        inputImageBuffer = imageProcessor.process(inputImageBuffer)

        // run model
        model.run(inputImageBuffer.buffer, outputBuffer.buffer.rewind())

        // get output
        val labelOutput = TensorLabel(labels, outputBuffer)

        val label = labelOutput.mapWithFloatValue
        return label
    }
}

private fun Long.dateWithFormat(formatString: String): String {
    val date = Date()
    date.time = this
    return SimpleDateFormat(formatString, Locale.US).format(date)
}
