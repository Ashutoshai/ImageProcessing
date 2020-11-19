Helmet detection using google Mediapipe
**helmetdetection**
In this assignment I used tensorflow object detection Api
Steps to Achieve the Functionality
1.Creating Input data .txt files through bounding boxes from images.
2.Converting to train and text.
3.Creating train and test tf records.
4.Added helmet label.
5.Started the training of data.
6.Convert the model file to tflite by using following command
7.Change the mediapipe graph.pbtxt file and add label into label.txt file and place the tflite model into model directory of mediapipe.
8.in the graph.pbtxt file change the below configuration
Modelname,numboxes,stride,classes,aspect_ratio,x_scale,y_scale,h_scale,w_scale.
9.Also used android studio to run the tflite model.
10.Since I donot have gpus trained on less iamges on cpu.

Observations:
1.Automl and Teachable Machine is not comaptible with Mediapipe
2.Tensorflow 2.x is not comaptible with google media cpu mode of execution.
3.Faced Environment issues with yolov3 with mediapipe.

Sample Commands

**tfrecord creation**
python generate_tfrecord.py --csv_input="C:\Users\pashutosh\models\research\train.csv"  --output_path="C:\Users\pashutosh\models\research\train.record" --image_dir="C:/Users/pashutosh/models/research/train/images/"
**train the tf record**
python object_detection/legacy/train.py --logtostderr --train_dir=C:/Users/pashutosh/models/research/trained_model/  --pipeline_config_path=C:/Users/pashutosh/models/research/pretrained_model/ssd_mobilenet_v2_quantized/pipeline.config

python object_detection/export_tflite_ssd_graph.py --pipeline_config_path=C:/Users/pashutosh/models/research/pretrained_model/ssd_mobilenet_v2_quantized/pipeline.config --trained_checkpoint_prefix=C:/Users/pashutosh/models/research/trained_model/model.ckpt-586 --output_directory=C:/Users/pashutosh/models/research/trained_model/tflite/ --add_postprocessing_op=true

**convert to tflite mode in GPU**
tflite_convert --graph_def_file=tflite/tflite_graph.pb --output_file=tflite/detect.tflite --output_format=TFLITE --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite-Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=127 --change_concat_input_ranges=false --allow_custom_ops

**convert to tflite mode in CPU**
C:\Users\pashutosh\mediapipe_repo\mediapipe\bazel.exe run graph_transforms:summarize_graph --in_graph=C:/Users/pashutosh/models/research/trained_model/tflite/tflite_graph.pb


tflite_convert --graph_def_file=tflite/tflite_graph.pb --output_file=tflite/model.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=FLOAT --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions


In the below Youtube link we have uploaded the video in which masked images are there.


