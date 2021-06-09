"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import numpy as np
import glob
import cv2
import argparse
import tensorflow as tf

print("TF version:{}".format(tf.__version__))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def converer_keras_to_tflite_v1(keras_path, outputs_layer=None, out_tflite=None):
    """
    :param keras_path: keras *.h5 files
    :param outputs_layer
    :param out_tflite: output *.tflite file
    :return:
    """
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    model_keras = tf.keras.models.load_model(keras_path)
    print(model_keras.summary())
    # 从keras模型中提取fc1层, 需先保存成新keras模型, 再转换成tflite
    model_embedding = tf.keras.models.Model(inputs=model_keras.input,
                                            outputs=model_keras.get_layer(outputs_layer).output)
    print(model_embedding.summary())
    keras_file = os.path.join(model_dir, "{}_{}.h5".format(model_name, outputs_layer))
    tf.keras.models.Model.save(model_embedding, keras_file)

    # converter = tf.lite.TocoConverter.from_keras_model_file(keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)  # tf1.3
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)  # tf2.0
    tflite_model = converter.convert()

    if not out_tflite:
        out_tflite = os.path.join(model_dir, "{}_{}.tflite".format(model_name, outputs_layer))
    open(out_tflite, "wb").write(tflite_model)
    print("successfully convert to tflite done")
    print("save model at: {}".format(out_tflite))


def converer_keras_to_tflite_v2(keras_path, outputs_layer=None, out_tflite=None, optimize=False, quantization=False):
    """
    :param keras_path: keras *.h5 files
    :param outputs_layer: default last layer
    :param out_tflite: output *.tflite file
    :param optimize
    :return:
    """
    if not os.path.exists(keras_path):
        raise Exception("Error:{}".format(keras_path))
    model_dir = os.path.dirname(keras_path)
    model_name = os.path.basename(keras_path)[:-len(".h5")]
    # 加载keras模型, 结构打印
    # model = tf.keras.models.load_model(keras_path)
    model = tf.keras.models.load_model(model_path, custom_objects={'tf': tf}, compile=False)

    print(model.summary())
    if outputs_layer:
        # 从keras模型中提取层,转换成tflite
        model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(outputs_layer).output)
        # outputs = [model.output["bbox"],model.output["scores"]]
        # model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
        print(model.summary())
    # converter = tf.lite.TocoConverter.from_keras_model_file(keras_file)
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)  # tf1.3
    converter = tf.lite.TFLiteConverter.from_keras_model(model)  # tf2.0
    prefix = [model_name, outputs_layer]
    # converter.allow_custom_ops = False
    # converter.experimental_new_converter = True
    """"
    https://tensorflow.google.cn/lite/guide/ops_select
    我们优先推荐使用 TFLITE_BUILTINS 转换模型，然后是同时使用 TFLITE_BUILTINS,SELECT_TF_OPS ，
    最后是只使用 SELECT_TF_OPS。同时使用两个选项（也就是 TFLITE_BUILTINS,SELECT_TF_OPS）
    会用 TensorFlow Lite 内置的运算符去转换支持的运算符。
    有些 TensorFlow 运算符 TensorFlow Lite 只支持部分用法，这时可以使用 SELECT_TF_OPS 选项来避免这种局限性。
    """
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
    #                                        tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if optimize:
        print("weight quantization")
        # Enforce full integer quantization for all ops and use int input/output
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        prefix += ["optimize"]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "int8":
        converter.representative_dataset = representative_dataset_gen
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8  # or tf.uint8
        # converter.inference_output_type = tf.int8  # or tf.uint8
        converter.target_spec.supported_types = [tf.int8]
    elif quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]

    prefix += [quantization]
    if not out_tflite:
        prefix = [str(n) for n in prefix if n]
        prefix = "_".join(prefix)
        out_tflite = os.path.join(model_dir, "{}.tflite".format(prefix))
    tflite_model = converter.convert()
    open(out_tflite, "wb").write(tflite_model)
    print("successfully convert to tflite done")
    print("save model at: {}".format(out_tflite))


def representative_dataset_gen():
    """
    # 生成代表性数据集
    :return:
    """
    image_dir = '/home/dm/panjinquan3/FaceDetector/tf-yolov3-detection/data/finger_images/'
    input_size = [320, 320]
    imgSet = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for img_path in imgSet:
        orig_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image_tensor = cv2.resize(rgb_image, dsize=tuple(input_size))
        image_tensor = np.asarray(image_tensor / 255.0, dtype=np.float32)
        image_tensor = image_tensor[np.newaxis, :]
        yield [image_tensor]


def parse_args():
    # weights_path = "../yolov3-micro.h5"
    weights_path = "../yolov3-micro_freeze_head.h5"
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--model_path", help="model_path", default=weights_path, type=str)
    parser.add_argument("--outputs_layer", help="outputs_layer", default=None, type=str)
    parser.add_argument("-o", "--out_tflite", help="out tflite model path", default=None, type=str)
    parser.add_argument("-opt", "--optimize", help="optimize model", default=True, type=bool)
    parser.add_argument("-q", "--quantization", help="quantization model", default="float16", type=str)
    return parser.parse_args()


def get_model(model_path):
    if not model_path:
        model_path = os.path.join(os.getcwd(), "*.h5")
        model_list = glob.glob(model_path)
    else:
        model_list = [model_path]
    return model_list


def unsupport_tflite_op():
    """
    tf.shape,tf.Size
    error: 'tf.Size' op is neither a custom op nor a flex op
    error: 'tf.Softmax' op is neither a custom op nor a flex op
    ===========================================================
    tf.Softmax-->tf.nn.softmax
    """
    pass


if __name__ == '__main__':

    args = parse_args()
    # outputs_layer = "fc1"
    outputs_layer = args.outputs_layer
    model_list = get_model("resnet_transfer_trained_model.h5")
    out_tflite = args.out_tflite
    optimize = args.optimize
    quantization = args.quantization
    for model_path in model_list:
        converer_keras_to_tflite_v2(model_path, outputs_layer, out_tflite=out_tflite, optimize=optimize,quantization=quantization)