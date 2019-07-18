# Android Object Detection - OpenCV and YOLO

<p align="center">
  <img src="https://github.com/Ickarus/AndroidObjectDetector-OpenCV/blob/master/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png">
</p>
<p align="center">
 Real time object detection Android application using OpenCV 4.1 and YOLO. 
</p>
<p align="center">
Author: Matteo Medioli
</p>
<p align="center">
YOLO: https://pjreddie.com/darknet/yolo/
</p>

## 1.Import OpenCV 4.1 Module in AndroidStudio 3.4.1
1) Dowload OpenCV SDK from https://sourceforge.net/projects/opencvlibrary/files/4.1.0/opencv-4.1.0-android-sdk.zip/download
2) Clone this project.
3) Open Android Studio and import this project.
4) Build project.
5) From AndroidStudio top-menù select **New -> Import Module** and select your path to OpenCV sdk folder (i.e */where_opencv_saved/OpenCV-android-sdk/sdk*) and rename module as OpenCV.
6) After load OpenCV module, re-build project.

## 2.Add OpenCV dependecies to your application
After OpenCV module import:
1) From AndroidStudio top-menù select **File -> Project Structure**
2) Navigate to Dependencies and click on **app**. On the right panel there's a plus button **+** for add Dependency. Click on it and choose **Module Dependecy**.
3) Select **OpenCV** module loaded before.
4) Click Ok and Apply changes.
5) Build project.

## 3.How detection works: CameraActivity.java
This activity is the core of application and it implements *org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2*.
It has 2 main private instance variable: a **net** (*org.opencv.dnn.Net*) and a **cameraView** (*org.opencv.android.CameraBridgeViewBase*).
Basically has three main features:

#### a) Load Network
Load convolutional net from *\*.cfg* and *\*.weights* files and read *labels name* (COCO Dataset) in assets folder when calls onCameraViewStarted() using  **Dnn.readNetFromDarknet(String path_cfg, String path_weights)**.  
**NOTE**: this repo doesn't contain weights file. You have to download it from YOLO site.


#### b) Detection from camera preview
Iteratively generate a frame from CameraBridgeViewBase preview and analize it as an image. Real time detection and the frames flow generation is managed by **onCameraFrame(CvCameraViewFrame inputFrame)**.
Preview frame is translate in a Mat matrix and set as input for **Dnn.blobFromImage(frame, scaleFactor, frame_size, mean, true, false)** to preprocess frames. Note that **frame_size** is 416x416 for YOLO Model (you can find input dimension in *\*.cfg* file). We can change the size by adding or subtracting by a factor of 32. Reducing the framesize increases the performance but worsens the accuracy.
The detection phase is implemented by **net.forward(List\<Mat> results, List\<String> outNames)** that runs forward pass to compute output of layer with name *outName*. In *results* the method writes all detections in preview frame as Mat objects.
Theese Mat instances contain all information such as positions and labels of detected objects.


#### c) Draw bounding-box and labelling
Performing **Non Maximum Suppression** by YOLO, in **List\<Mat> results** are stored all coordinates of optimal bounding boxes
(the first 4 numbers are [*center_x, center_y, width, height*], followed by all class probabilities). *classId* is the corresponding index for label of detection in COCO Dataset list *className*.

## TO DO
- [ ] Full screen JavaCameraView portrait mode
- [ ] Speed up JavaCameraView
- [ ] Add model chioce



