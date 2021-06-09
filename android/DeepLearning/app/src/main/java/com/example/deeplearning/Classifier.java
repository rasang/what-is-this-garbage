package com.example.deeplearning;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Classifier {
    private static Context mContext;
    private Interpreter mInterpreter;
    private static Classifier mInstance;

    public static Classifier newInstance(Context context){
        mContext = context;
        if(mInstance == null){
            mInstance = new Classifier();
        }
        return mInstance;
    }

    // 获得实例
    Interpreter get(){
        try {
            if(mInterpreter == null){
                mInterpreter = new Interpreter(loadModel(mContext));
            }
        } catch (IOException e){
            e.printStackTrace();
        }
        return mInterpreter;
    }

    // 加载代码，来自网上
    private MappedByteBuffer loadModel(Context context) throws IOException{
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("resnet_transfer.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static float[][][][] getScaledMatrix(Bitmap bitmap, int[] ddims) {
        //新建一个1*256*256*3的四维数组
        float[][][][] inFloat = new float[ddims[0]][ddims[1]][ddims[2]][ddims[3]];
        //新建一个一维数组，长度是图片像素点的数量
        int[] pixels = new int[ddims[1] * ddims[2]];
        //把原图缩放成我们需要的图片大小
        Bitmap bm = Bitmap.createScaledBitmap(bitmap, ddims[1], ddims[2], false);
        //把图片的每个像素点的值放到我们前面新建的一维数组中
        bm.getPixels(pixels, 0, bm.getWidth(), 0, 0, ddims[1], ddims[2]);
        int pixel = 0;
        //for循环，把每个像素点的值转换成RBG的值，存放到我们的目标数组中
        for (int i = 0; i < ddims[1]; ++i) {
            for (int j = 0; j < ddims[2]; ++j) {
                final int val = pixels[pixel++];
                float red = ((val >> 16) & 0xFF);
                float green = ((val >> 8) & 0xFF);
                float blue = (val & 0xFF);
                float[] arr = {red, green, blue};
                inFloat[0][i][j] = arr;
            }
        }
        if (bm.isRecycled()) {
            bm.recycle();
        }
        return inFloat;
    }
}
