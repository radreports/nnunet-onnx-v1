import onnx
import onnxruntime as ort
import numpy as np
import nibabel as nib
import argparse

def load_nii_gz_image(image_path):
    # Load the image using nibabel
    img = nib.load(image_path)
    data = img.get_fdata()
    # Convert data to float32 and add batch and channel dimensions
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=0)  # Add channel dimension
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    return data, img.affine

def save_nii_gz_image(data, affine, output_path):
    # Remove batch and channel dimensions
    data = np.squeeze(data)
    # Save the output data as .nii.gz
    output_img = nib.Nifti1Image(data, affine)
    nib.save(output_img, output_path)

def test_onnx_model(onnx_model_path, input_image_path, output_image_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Create a runtime session
    ort_session = ort.InferenceSession(onnx_model_path)

    # Load and preprocess the input image
    input_data, affine = load_nii_gz_image(input_image_path)

    # Run the model
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)

    # Save the output
    save_nii_gz_image(ort_outs[0], affine, output_image_path)
    print(f"ONNX model output saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--onnx_model_path', type=str, required=True, help="Path to the ONNX model")
    parser.add_argument('-i', '--input_image_path', type=str, required=True, help="Path to the input .nii.gz image")
    parser.add_argument('-o', '--output_image_path', type=str, required=True, help="Path to save the output .nii.gz image")
    args = parser.parse_args()

    test_onnx_model(args.onnx_model_path, args.input_image_path, args.output_image_path)
