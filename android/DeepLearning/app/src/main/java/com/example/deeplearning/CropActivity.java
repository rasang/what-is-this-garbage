package com.example.deeplearning;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.provider.MediaStore;
import android.view.View;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import com.theartofdev.edmodo.cropper.CropImageView;

public class CropActivity extends Activity {

    //相机请求码
    private static final int REQUESTCODE_CAMERA = 1002;
    private Bitmap picture;
    private CropImageView cropImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_crop);
        this.cropImageView = findViewById(R.id.cropImageView);
        // 设置纵横比
        this.cropImageView.setAspectRatio(1, 1);
        // 拍摄照片
        getPictureFromCamera();
    }

    // 调用相机拍摄照片
    private void getPictureFromCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUESTCODE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) {
            return;
        }
        switch (requestCode) {
            case REQUESTCODE_CAMERA:
                Bundle bundle = data.getExtras();
                Bitmap bitmap = (Bitmap) bundle.get("data");
                this.picture = bitmap;
                // 将所得Bitmap设置到第三方用来裁剪图片的库
                cropImageView.setImageBitmap(picture);
                break;
        }
    }

    // OK直接识别
    public void ok(View view){
        Intent intent = new Intent();
        Bitmap resultPicture = this.cropImageView.getCroppedImage();
        intent.putExtra("data", resultPicture);
        setResult(RESULT_OK, intent);
        finish();
    }

    // 取消识别
    public void cancel(View view){
        Intent intent = new Intent();
        setResult(RESULT_CANCELED, intent);
        finish();
    }
}
