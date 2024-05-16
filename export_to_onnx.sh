export RESULTS_FOLDER=/path/to/nnUNet/results
python export_nnunet_to_onnx.py --task 1 --folds 0 --output_path model.onnx --input_shape 1 1 128 128 128
