import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:http/http.dart' as http;
import 'package:camera/camera.dart';
import 'package:face_mask_detector/main.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:intl/intl.dart';

class Home extends StatefulWidget {
  const Home({Key? key}) : super(key: key);

  @override
  _HomeState createState() => _HomeState();
}

class _HomeState extends State<Home> {
  late CameraImage imgCamera;
  late CameraController cameraController;
  bool isWorking = false;
  // List recognitionsList;

  String result = "";

  late CameraDescription frontCamera;

  late FaceDetectorOptions options;

  late FaceDetector faceDetector;

  List<OverlayBox>? recognitionsList = [];
  late Size screen;
  List<Map> detectedList = [];

  int _cameraIndex = -1;

  initCamera() {
    options = FaceDetectorOptions(
        enableTracking: true,
        minFaceSize: 0.3,
        performanceMode: FaceDetectorMode.accurate);
    faceDetector = FaceDetector(options: options);

    frontCamera = cameras!.firstWhere(
        (element) => element.lensDirection == CameraLensDirection.front);
    _cameraIndex = cameras!.indexOf(frontCamera);
    cameraController = CameraController(frontCamera, ResolutionPreset.high,
        enableAudio: false);

    cameraController.initialize().then((value) {
      if (!mounted) {
        return;
      }

      setState(() {
        cameraController.startImageStream((imageFromStream) => {
              if (!isWorking)
                {
                  isWorking = true,
                  imgCamera = imageFromStream,
                  runModelOnFrame(),
                }
            });
      });
    });
  }

  loadmodel() async {
    await Tflite.loadModel(
      model: "assets/model_v2.tflite",
      labels: "assets/labels_v2.txt",
    );
  }

  runModelOnFrame() async {
    // ignore: unnecessary_null_comparison
    if (imgCamera != null) {
      var inputImage = await _processCameraImage(imgCamera);

      print("input image >> " + inputImage.inputImageData!.size.toString());

      //detect faces
      final List<Face> faces = await faceDetector.processImage(inputImage);

      print("faces size >>>>   :::::::::   " + faces.length.toString());

      if (faces.isEmpty) {
        setState(() {
          result = "";
        });
        isWorking = false;
        return;
      }

      // var bitmap = Bitmap.fromHeadless(
      //     imgCamera.width, imgCamera.height, inputImage.bytes!);

      for (Face face in faces) {
        final Rect rect = face.boundingBox;
        var overlayBox = OverlayBox();
        overlayBox.rect = face.boundingBox;
        overlayBox.description = "";

        if (faces.isNotEmpty) {
          var recognitions = await Tflite.runModelOnFrame(
            bytesList: imgCamera.planes.map((plane) {
              return plane.bytes;
            }).toList(),
            imageHeight: imgCamera.height,
            imageWidth: imgCamera.width,
            imageMean: 127.5,
            imageStd: 127.5,
            rotation: 90,
            threshold: 0.7,
            asynch: true,
          );

          DetectJson detectJson = DetectJson();
          detectJson.status = recognitions!.first["label"];
          detectJson.total_number = faces.length;
          detectJson.timestamp =
              DateFormat("dd-MM-yyyy_HH-mm-ss").format(DateTime.now());

          detectedList.add(detectJson.tojsonData());

          if (detectedList.length >= 100) {
            var body = json.encode({"array": detectedList});
            postData(body).then(
                (value) => {print("data after response \n\n\n " + value.body)});
            detectedList = [];
          }

          result = "";
          print("recognitions >> " + recognitions.toString());

          if (recognitions.isNotEmpty) {
            recognitions.sort(sortComparator);
            var first = recognitions.first;

            result =
                first["label"] + " : " + first["confidence"].toString() + "\n";
            result += "faces count : " + faces.length.toString();
          }

          // recognitions.forEach((response) {
          //   result += response["label"] +
          //       " : " +
          //       response["confidence"].toString() +
          //       "\n";
          //   result += "faces count : " + faces.length.toString();
          // });

          setState(() {
            result;
          });
        }
      }
      isWorking = false;
    }
  }

  Future<http.Response> postData(String body) {
    print("post call >>>> " + body);
    return http.post(
      Uri.parse('https://gsheet-data.herokuapp.com/post_json'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: body,
    );
  }

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    initCamera();
    loadmodel();
  }

  @override
  Widget build(BuildContext context) {
    screen = MediaQuery.of(context).size;
    return MaterialApp(
      home: SafeArea(
        child: Scaffold(
          body: Stack(
            children: [
              Positioned.fill(
                child: (!cameraController.value.isInitialized)
                    ? Container()
                    : CameraPreview(cameraController),
              ),
              Align(
                alignment: Alignment.topCenter,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Text(
                    result,
                    style: const TextStyle(
                      backgroundColor: Colors.black54,
                      fontSize: 20,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<InputImage> _processCameraImage(CameraImage image) async {
    final WriteBuffer allBytes = WriteBuffer();
    for (final Plane plane in image.planes) {
      allBytes.putUint8List(plane.bytes);
    }
    final bytes = allBytes.done().buffer.asUint8List();

    final Size imageSize =
        Size(image.width.toDouble(), image.height.toDouble());

    print("image size for input image data >> " + imageSize.toString());

    final camera = cameras![_cameraIndex];
    final imageRotation =
        // InputImageRotationValue.fromRawValue(camera.sensorOrientation) ??
        InputImageRotation.rotation0deg;

    final inputImageFormat =
        InputImageFormatValue.fromRawValue(image.format.raw) ??
            InputImageFormat.nv21;

    final planeData = image.planes.map(
      (Plane plane) {
        return InputImagePlaneMetadata(
          bytesPerRow: plane.bytesPerRow,
          height: plane.height,
          width: plane.width,
        );
      },
    ).toList();

    final inputImageData = InputImageData(
      size: imageSize,
      imageRotation: imageRotation,
      inputImageFormat: inputImageFormat,
      planeData: planeData,
    );

    final inputImage =
        InputImage.fromBytes(bytes: bytes, inputImageData: inputImageData);

    print("inputImage size >>>> " + imageSize.toString());

    return inputImage;
  }

  int sortComparator(a, b) {
    return double.parse(b["confidence"].toString())
        .compareTo(double.parse(a["confidence"].toString()));
  }
}

class DetectJson {
  String timestamp = "";
  String status = "";
  int total_number = 0;

  Map<String, dynamic> tojsonData() {
    var map = <String, dynamic>{};
    map["timestamp"] = timestamp;
    map["status"] = status;
    map["total_number"] = total_number;

    return map;
  }
}

class OverlayBox {
  late Rect rect;
  late String description;
  late LABEL label;
}

enum LABEL { Mask, No_Mask, Covered_Mouth_Chin, Covered_Nose_Mouth }
