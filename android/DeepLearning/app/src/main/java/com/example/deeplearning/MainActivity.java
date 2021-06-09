package com.example.deeplearning;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends Activity {

    // 相机的requestCode
    private static final int REQUESTCODE_CAMERA = 1002;
    // 存储图片用
    private Bitmap picture;
    // 用来设置首页是否可见，一般来说只有第一次进入可见
    private LinearLayout linearLayout;
    // 用来显示要识别的图片
    private ImageView imageView;
    // 显示识别结果
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // 获得组件实例
        this.linearLayout = findViewById(R.id.main);
        this.imageView = findViewById(R.id.background);
        this.textView = findViewById(R.id.result);
    }

    /* 从相机中获取照片 */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // 判断结果是否有数据
        if (resultCode != RESULT_OK) {
            return;
        }
        Log.d("test", String.valueOf(requestCode));
        switch (requestCode) {
            // 处理返回数据
            case REQUESTCODE_CAMERA:
                // 先获得bitmap对象
                Bundle bundle = data.getExtras();
                Bitmap bitmap = (Bitmap) bundle.get("data");
                this.picture = bitmap;
                // 将首页背景设置消失
                this.linearLayout.setVisibility(View.INVISIBLE);
                // 设置imageView为拍摄的图片
                this.imageView.setBackground(new BitmapDrawable(getResources(), bitmap));
                // 将所得图片按照一定比例缩小
                Bitmap scaleBitmap = Bitmap.createScaledBitmap(bitmap, 200, 200, false);
                int[] dims = {1, 200, 200, 3};
                // 转化成float数组用于输入
                float[][][][] image = Classifier.getScaledMatrix(scaleBitmap, dims);
                // 设置result数组用来存结果
                float[][] result = {{0, 0, 0 , 0, 0, 0}};
                // 识别
                Classifier.newInstance(this).get().run(image, result);
                // 将结果拼接输出
                int  maxIndex = -1;
                float max = 0;
                String resultText = "";
                for(int i=0;i<result[0].length;i++){
                    if (i == 0)
                            resultText = resultText + "纸板：" + String.format("%.2f", result[0][i]) + "\n";
                    else if (i==1)
                            resultText = resultText + "玻璃：" + String.format("%.2f", result[0][i]) + "\n";
                    else if (i==2)
                            resultText = resultText + "金属：" + String.format("%.2f", result[0][i]) + "\n";
                    else if (i==3)
                            resultText = resultText + "纸张：" + String.format("%.2f", result[0][i]) + "\n";
                    else if (i==4)
                            resultText = resultText + "塑料：" + String.format("%.2f", result[0][i]) + "\n";
                    else if (i==5)
                            resultText = resultText + "其他：" + String.format("%.2f", result[0][i]) + "\n";
                }
                this.textView.setText(resultText);
        }
    }

    //启动Activity拍摄照片
    public void takePicture(View view) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
            //如果有了相机的权限有调用相机
            startActivityForResult(new Intent(MainActivity.this, CropActivity.class), REQUESTCODE_CAMERA);

        }else{
            //否则去请求相机权限
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},100);
        }
    }
}
