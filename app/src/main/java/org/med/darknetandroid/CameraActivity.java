package org.med.darknetandroid;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Camera;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.imgproc.Imgproc;


public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "CameraActivity";
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE  = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    private static List<String> classNames;
    private static List<Scalar> colors=new ArrayList<>();
    private Net net;
    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean permissionGranted = false;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!permissionGranted) {
            checkPermissions();
        }
        mOpenCvCameraView = findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        //mOpenCvCameraView.setMaxFrameSize(640,480);
        mOpenCvCameraView.setCvCameraViewListener(this);
        classNames = readLabels("labels.txt", this);
        for(int i=0; i<classNames.size(); i++)
            colors.add(randomColor());
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        String modelConfiguration = getAssetsFile("yolov2-tiny.cfg", this);
        String modelWeights = getAssetsFile("yolov2-tiny.weights", this);
        net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame){


        // Rotate mRgba 90 degrees
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        //Imgproc.GaussianBlur(frame,frame, new Size(5,5), 0);

        Size frame_size = new Size(416, 416);
        Scalar mean = new Scalar(127.5);

        Mat blob = Dnn.blobFromImage(frame, 0.00392, frame_size, mean, true, false);
        net.setInput(blob);

        List<Mat> result = new ArrayList<>();
        List<String> outBlobNames = net.getUnconnectedOutLayersNames();

        net.forward(result, outBlobNames);
        Log.d("DETECT", "netForward success!");
        Log.d("DETECT", "Frame Size: "+frame.width()+"X"+frame.height());
        float confThreshold = 0.25f;
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        List<Scalar> clrs = new ArrayList<>();
        Log.d("DETECT", "LIST MAT SIZE: "+result.size());
        for (int i = 0; i < result.size(); ++i)
        {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities
            Mat level = result.get(i);
            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold)
                {

                    int centerX = (int)(row.get(0,0)[0] * frame.cols());
                    int centerY = (int)(row.get(0,1)[0] * frame.rows());
                    int width   = (int)(row.get(0,2)[0] * frame.cols());
                    int height  = (int)(row.get(0,3)[0] * frame.rows());
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.add((int)classIdPoint.x);
                    confs.add(confidence);
                    DecimalFormat df = new DecimalFormat("#.##");
                    Log.d("ID DETECTED", ""+clsIds.toString());
                    Log.d("ID ADDED", ""+clsIds.get(clsIds.size()-1));
                    Log.d("ID[0] ADDED", ""+clsIds.get(0));
                    labels.add(classNames.get(clsIds.get(clsIds.size()-1))+ ": "+df.format(confs.get(i)));
                    clrs.add(colors.get(clsIds.get(clsIds.size()-1)));
                    rects.add(new Rect(left, top, width, height));

                    Log.d("LABELS DETECT", ""+labels.toString());
                }
            }
        }

        // Apply non-maximum suppression procedure.
        float nmsThresh = 0.5f;
        MatOfFloat confidences = new MatOfFloat();
        confidences.fromList(confs);
        Rect[] boxesArray = rects.toArray(new Rect[0]);
        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); //We draw the bounding boxes for objects here//
        Log.d("DETECT", "BEFORE BOX!");
        // Draw result boxes:
        if(!indices.empty())
        {
            int [] ind = indices.toArray();
            for (int i = 0; i < ind.length; ++i)
            {
                int idx = ind[i];
                Rect box = boxesArray[idx];
                Imgproc.rectangle(frame, box.tl(), box.br(), clrs.get(idx), 3, 2);
                Imgproc.putText (frame, labels.get(idx), new Point(box.x, box.y-5), Imgproc.FONT_HERSHEY_SIMPLEX , 1, new Scalar(0, 0, 0),4);
                Imgproc.putText (frame, labels.get(idx), new Point(box.x, box.y-5), Imgproc.FONT_HERSHEY_SIMPLEX , 1, new Scalar(255, 255, 255),2);
                Log.d("DETECT", labels.get(idx));
                //saveFrame(frame);
            }
        }

        return frame;
    }

    private boolean checkPermissions() {

        int permissionCheck = ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA);

        if (permissionCheck != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
            return false;
        } else {
            return true;
        }

    }


    private static String getAssetsFile(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }

    private List<String> readLabels (String file, Context context)
    {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream;
        List<String> labelsArray = new ArrayList<>();
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            Scanner fileScanner = new Scanner(new File(outFile.getAbsolutePath())).useDelimiter("\n");
            String label;
            while (fileScanner.hasNext()) {
                label = fileScanner.next();
                labelsArray.add(label);
            }
            fileScanner.close();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to read labels!");
        }
        return labelsArray;
    }

    private Scalar randomColor() {
        Random random = new Random();
        int r = random.nextInt(255);
        int g = random.nextInt(255);
        int b = random.nextInt(255);
        return new Scalar(r,g,b);
    }

    private void saveFrame(Mat frame) {
        try {
            ; // see other conf types
            Bitmap bmp = Bitmap.createBitmap(frame.width(), frame.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame, bmp);
            String path = Environment.getExternalStorageDirectory().toString();
            OutputStream fOut = null;
            File file = new File(path, "foto.jpg"); // the File to save , append increasing numeric counter to prevent files from getting overwritten.
            fOut = new FileOutputStream(file);
            bmp.compress(Bitmap.CompressFormat.JPEG, 85, fOut); // saving the Bitmap to a file compressed as a JPEG with 85% compression rate
            fOut.flush(); // Not really required
            fOut.close(); // do not forget to close the stream
            MediaStore.Images.Media.insertImage(getContentResolver(), file.getAbsolutePath(), file.getName(), file.getName());
            //Toast.makeText(getApplicationContext(), "Detect object, screen captured",Toast.LENGTH_SHORT).show();
        } catch (IOException e) {
            Log.e("ERROR", "" + e);
        }
    }



    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
}
