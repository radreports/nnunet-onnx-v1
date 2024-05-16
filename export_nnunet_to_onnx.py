import os
import torch
import argparse
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.paths import network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

def load_model(task_id, folds, model_name="3d_fullres"):
    task_name = convert_id_to_task_name(task_id)
    model_folder_name = os.path.join(network_training_output_dir,"nnUNet", model_name, f"{task_name}", "nnUNetTrainerV2__nnUNetPlansv2.1")
    
    # Check if the model folder exists
    if not os.path.exists(model_folder_name):
        raise FileNotFoundError(f"Model folder {model_folder_name} does not exist.")
    
    # Check if all fold directories exist
    fold_dirs = [os.path.join(model_folder_name, f"fold_{fold}") for fold in folds]
    for fold_dir in fold_dirs:
        if not os.path.isdir(fold_dir):
            raise FileNotFoundError(f"Fold directory {fold_dir} does not exist.")

    print(f"Loading model from: {model_folder_name}")
    trainer, params = load_model_and_checkpoint_files(model_folder_name, folds)
    model = trainer.network
    return model

def convert_to_onnx(model, input_shape, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model has been converted to ONNX and saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True, help="Task ID")
    parser.add_argument('--folds', type=int, nargs='+', default=[0], help="Fold numbers of the model")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument('--input_shape', type=int, nargs='+', required=True, help="Shape of the model input")
    parser.add_argument('--model_name', type=str, default="3d_fullres", help="Model name (e.g., 2d, 3d_fullres, 3d_lowres)")
    args = parser.parse_args()

    # Ensure the RESULTS_FOLDER environment variable is set
    if 'RESULTS_FOLDER' not in os.environ:
        raise EnvironmentError("Please set the RESULTS_FOLDER environment variable.")

    network_training_output_dir = os.environ['RESULTS_FOLDER']

    model = load_model(args.task, args.folds, args.model_name)
    convert_to_onnx(model, args.input_shape, args.output_path)
