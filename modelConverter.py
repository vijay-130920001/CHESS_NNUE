import torch

def convert_pth_to_txt(pth_path, txt_path):
    try:
        # Load the state dictionary
        state_dict = torch.load(pth_path, map_location=torch.device('cpu')) # Load on CPU to avoid device issues

        param_output = ""
        param_count = 0

        for name, param in state_dict.items():
            param_count += param.numel()
            param_output += f"Layer: {name}\n"  # Add layer name for clarity
            for value in param.flatten().tolist():
                param_output += str(value) + "\n"
            param_output += "\n" # Add extra newline for better readability

        print(f"Converted {param_count} weights.")

        with open(txt_path, "w") as weight_file:
            weight_file.write(param_output)

        print(f"Weights saved to {txt_path}")

    except FileNotFoundError:
        print(f"Error: .pth file not found at {pth_path}")
    except RuntimeError as e: # Catch other potential errors during loading
        print(f"Error loading .pth file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# Example usage:
pth_file = "vijay2a_MAE_small_model.pth"  # Replace with the actual path to your .pth file
txt_file = "vijay2_small_model__wt.txt"  # Replace with desired output path

convert_pth_to_txt(pth_file, txt_file)
