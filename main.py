import os
import sys
from PIL import Image
import warnings
from transparent_background import Remover
import numpy as np
# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*tensorboard.*")
warnings.filterwarnings("ignore", message=".*xFormers not available.*")

try:
    remover = Remover(jit=True)  # Attempt to use GPU
except Exception as e:
    remover = Remover(jit=False)
    
def process_image(input_path, output_path):
    """
    Processes an input image and saves the result to the output path.

    Parameters:
    - input_path (str): Path to the input image file. Bapak
    - output_path (str): Path to save the processed output. Mamak
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_image = Image.open(input_path)

    output = remover.process(input_image, type='rgba')
    if isinstance(output, np.ndarray):
        output = Image.fromarray(output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.save(output_path, format='PNG')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_image(input_path, output_path)
