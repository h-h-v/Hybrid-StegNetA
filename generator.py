import os
import sys
from PIL import Image
import torch
from torchvision import datasets, transforms
import numpy as np
import shutil
import random

# --- CONFIGURATION ---
BASE_DIR = 'Stegware_Project_Data'
DATA_DIR = os.path.join(BASE_DIR, 'data')
PAYLOAD_FILE_PATH = 'simulated_malware.py' 
# ---------------------

# --- Utility Functions (Same as before) ---

def read_payload_binary(filepath):
    """Reads the entire content of the file as a binary string."""
    try:
        with open(filepath, 'rb') as f:
            byte_content = f.read()
            binary_data = ''.join(format(byte, '08b') for byte in byte_content)
            delimiter = '1' * 16 + '0' * 16 
            return binary_data + delimiter
    except FileNotFoundError:
        print(f"Error: Payload file not found at {filepath}.")
        print("Please ensure 'simulated_malware.py' is in the current directory.")
        return None

def create_dirs():
    """Creates the structured folders in your local project directory."""
    print("Creating project directories...")
    os.makedirs(os.path.join(DATA_DIR, 'final_clean'), exist_ok=True)
    stegware_dir = os.path.join(DATA_DIR, 'stegware')
    os.makedirs(stegware_dir, exist_ok=True)
    # Create four distinct stegware family folders
    os.makedirs(os.path.join(stegware_dir, 'lsb1_stego'), exist_ok=True) 
    os.makedirs(os.path.join(stegware_dir, 'lsb3_stego'), exist_ok=True) 
    os.makedirs(os.path.join(stegware_dir, 'ppm_stego'), exist_ok=True)
    os.makedirs(os.path.join(stegware_dir, 'parity_stego'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'temp_cifar_raw'), exist_ok=True)
    print("Directories created successfully!")

def download_cifar10():
    """Downloads CIFAR-10 and saves images to a temporary raw folder."""
    print("Downloading CIFAR-10 dataset...")
    raw_path = os.path.join(DATA_DIR, 'temp_cifar_raw')
    train_dataset = datasets.CIFAR10(
        root=raw_path,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    image_files = []
    for i, (img_tensor, _) in enumerate(train_dataset):
        img = transforms.ToPILImage()(img_tensor)
        filename = f'cifar_{i:05d}.png'
        img.save(os.path.join(raw_path, filename))
        image_files.append(filename)
    print(f"Saved {len(train_dataset)} raw images for processing.")
    return image_files

# --- NEW EMBEDDING FUNCTIONS ---

# Technique 1: LSB-1 (Least Significant Bit)
def embed_lsb1(image_path, output_path, binary_message):
    if not binary_message: return
    try: img = Image.open(image_path).convert('RGB')
    except Exception: return

    pixels = list(img.getdata())
    capacity = len(pixels) * 3
    if len(binary_message) > capacity: binary_message = binary_message[:capacity]

    binary_index = 0
    new_pixels = []
    for pixel in pixels:
        new_pixel = list(pixel)
        for i in range(3):
            if binary_index < len(binary_message):
                bit = int(binary_message[binary_index])
                new_pixel[i] = (new_pixel[i] & 0xFE) | bit # Mask 0xFE for LSB-1
                binary_index += 1
        new_pixels.append(tuple(new_pixel))
        if binary_index >= len(binary_message): break

    stego_img = Image.new('RGB', img.size)
    stego_img.putdata(new_pixels)
    stego_img.save(output_path)

# Technique 2: LSB-3 (Modifies the 3rd Least Significant Bit)
def embed_lsb3(image_path, output_path, binary_message):
    if not binary_message: return
    try: img = Image.open(image_path).convert('RGB')
    except Exception: return

    pixels = list(img.getdata())
    capacity = len(pixels) * 3
    if len(binary_message) > capacity: binary_message = binary_message[:capacity]

    mask = 0xFF - (1 << 2) # Mask 0xFB for LSB-3 (clears the 3rd bit)
    
    binary_index = 0
    new_pixels = []
    for pixel in pixels:
        new_pixel = list(pixel)
        for i in range(3):
            if binary_index < len(binary_message):
                bit = int(binary_message[binary_index])
                # Clear the 3rd LSB
                channel_value = new_pixel[i] & mask
                # Set the 3rd LSB
                new_pixel[i] = channel_value | (bit << 2)
                binary_index += 1
        new_pixels.append(tuple(new_pixel))
        if binary_index >= len(binary_message): break

    stego_img = Image.new('RGB', img.size)
    stego_img.putdata(new_pixels)
    stego_img.save(output_path)


# Technique 3: Pixel Pair Matching (PPM - Simplified)
# Embeds 1 bit per pixel using a simple PPM strategy (only checking the LSB)
def embed_ppm(image_path, output_path, binary_message):
    if not binary_message: return
    try: img = Image.open(image_path).convert('L') # Convert to grayscale for simplicity
    except Exception: return

    width, height = img.size
    pixels = list(img.getdata())
    
    # Capacity is 1 bit per pixel
    capacity = len(pixels) 
    if len(binary_message) > capacity: binary_message = binary_message[:capacity]
    
    binary_index = 0
    new_pixels = []
    
    for p in pixels:
        if binary_index < len(binary_message):
            m = int(binary_message[binary_index])
            p_lsb = p & 0x01
            
            # Simple PPM logic: change the pixel value only if the LSB doesn't match the message bit
            if p_lsb != m:
                if p < 255:
                    p += 1 # Increment to change LSB from 0 to 1, or 1 to 0
                else: # Handle edge case where p is 255
                    p -= 1 
            new_pixels.append(p)
            binary_index += 1
        else:
            new_pixels.append(p)

    stego_img = Image.new('L', img.size) # Save as L (Grayscale)
    stego_img.putdata(new_pixels)
    stego_img.save(output_path)

# Technique 4: Parity Encoding (Hides 1 bit per 4-pixel block)
def embed_parity(image_path, output_path, binary_message):
    if not binary_message: return
    try: img = Image.open(image_path).convert('L') # Convert to grayscale for simplicity
    except Exception: return

    width, height = img.size
    pixels = list(img.getdata())
    
    BLOCK_SIZE = 4
    # Capacity is 1 bit per block
    capacity = len(pixels) // BLOCK_SIZE
    if len(binary_message) > capacity: binary_message = binary_message[:capacity]

    binary_index = 0
    new_pixels = list(pixels) # Modifiable copy of the pixel list
    
    for i in range(0, len(pixels) - BLOCK_SIZE + 1, BLOCK_SIZE):
        if binary_index >= len(binary_message): break

        block = pixels[i:i + BLOCK_SIZE]
        m = int(binary_message[binary_index])
        
        # Calculate the current parity (XOR of LSBs of the 4 pixels)
        current_parity = (block[0] & 1) ^ (block[1] & 1) ^ (block[2] & 1) ^ (block[3] & 1)
        
        # If current parity doesn't match the message bit, change the LSB of the last pixel
        if current_parity != m:
            last_pixel = new_pixels[i + BLOCK_SIZE - 1]
            new_pixels[i + BLOCK_SIZE - 1] = last_pixel ^ 1 # XOR with 1 flips the LSB

        binary_index += 1

    stego_img = Image.new('L', img.size)
    stego_img.putdata(new_pixels)
    stego_img.save(output_path)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    create_dirs()
    
    all_files = download_cifar10()
    random.shuffle(all_files)

    # 50% Clean (25000) and 50% Stegware (25000)
    TOTAL_IMAGES = len(all_files)
    CLEAN_COUNT = 25000 
    STEGO_FAMILY_COUNT = 25000 // 4 # 6250 per family

    # Slice the randomized files into 5 groups
    clean_files = all_files[:CLEAN_COUNT]
    
    lsb1_files = all_files[CLEAN_COUNT : CLEAN_COUNT + STEGO_FAMILY_COUNT]
    lsb3_files = all_files[CLEAN_COUNT + STEGO_FAMILY_COUNT : CLEAN_COUNT + 2 * STEGO_FAMILY_COUNT]
    ppm_files = all_files[CLEAN_COUNT + 2 * STEGO_FAMILY_COUNT : CLEAN_COUNT + 3 * STEGO_FAMILY_COUNT]
    parity_files = all_files[CLEAN_COUNT + 3 * STEGO_FAMILY_COUNT : ]


    # Read the payload
    binary_payload = read_payload_binary(PAYLOAD_FILE_PATH)
    if not binary_payload:
        print("Cannot proceed without the payload file.")
        sys.exit(1)

    temp_raw_path = os.path.join(DATA_DIR, 'temp_cifar_raw')
    final_clean_path = os.path.join(DATA_DIR, 'final_clean')

    print("\n--- Generating Stegware and Finalizing Clean Set ---")
    
    # A. Finalize Clean Set
    for filename in clean_files:
        shutil.move(os.path.join(temp_raw_path, filename), os.path.join(final_clean_path, filename))
    print(f"1. Final Clean Set established: {len(clean_files)} images.")

    # B. Generate Stego Families
    families = [
        ('lsb1_stego', lsb1_files, embed_lsb1),
        ('lsb3_stego', lsb3_files, embed_lsb3),
        ('ppm_stego', ppm_files, embed_ppm),
        ('parity_stego', parity_files, embed_parity)
    ]

    total_stego_count = 0
    for name, files, embed_func in families:
        stego_path = os.path.join(DATA_DIR, 'stegware', name)
        processed_count = 0
        for filename in files:
            raw_path = os.path.join(temp_raw_path, filename)
            stego_output_path = os.path.join(stego_path, filename)
            
            # Call the specialized embedding function
            embed_func(raw_path, stego_output_path, binary_payload)
            processed_count += 1
        total_stego_count += processed_count
        print(f"2. {name} Family generated: {processed_count} images.")

    # Clean up temporary files
    shutil.rmtree(temp_raw_path)
    
    print("\nPhase 1 completed successfully.")
    print(f"Total Clean images: {len(clean_files)}")
    print(f"Total Stegware images: {total_stego_count}")
    print(f"Total Final Dataset Size: {len(clean_files) + total_stego_count}")
