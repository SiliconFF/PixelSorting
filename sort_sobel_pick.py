from PIL import Image
import numpy as np
import random
import cv2
from enum import Enum

class ColorMode(Enum):
    BRIGHTNESS = "Brightness"
    R = "Red"
    G = "Green"
    B = "Blue"


#Parameters
target_segment_length = 150
min_segment_length = 10
#max_segment_length = 1200
horizontal = False
vertical = True
color_mode = ColorMode.BRIGHTNESS


#function to generate edge mask:
def get_edge_mask(frame):
    # Process at full resolution
    (H, W) = frame.shape[:2]
    #frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.1)
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    
    # Perform Canny edge detection
    if horizontal:
        edges = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=7) #horizonal edges only
    else:
        edges = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=7) #vertical edges only
    
    # Enhance edge sharpness with unsharp masking
    gaussian = cv2.GaussianBlur(edges, (3, 3), 1)
    edges_sharpened = cv2.addWeighted(edges, 1.5, gaussian, -0.5, 0)
    output_image = Image.fromarray(edges_sharpened)
    #output_image.save('mask.png')
    return edges_sharpened



# Sorting function (sort by brightness)
# Define constants once
R_WEIGHT, G_WEIGHT, B_WEIGHT = 0.299, 0.587, 0.114
weights = np.array([0.299, 0.587, 0.114])
def get_brightness(pixel):
    return R_WEIGHT * pixel[0] + G_WEIGHT * pixel[1] + B_WEIGHT * pixel[2]

def get_red(pixel):
    return pixel[0]

def get_green(pixel):
    return pixel[1]

def get_blue(pixel):
    return pixel[2]


# Load image and mask
image = Image.open("person.jpg").convert("RGB")
mask = get_edge_mask(np.array(image))

# Resize mask to match image dimensions if necessary
#mask = mask.resize(image.size, Image.Resampling.LANCZOS)
image_array = np.array(image)
mask_array = np.array(mask)

# Verify dimensions
print("Image shape:", image_array.shape)
print("Mask shape:", mask_array.shape)

# Check if dimensions match
if image_array.shape[:2] != mask_array.shape:
    raise ValueError("Image and mask dimensions do not match!")

height, width, _ = image_array.shape

# Create output array
output_array = image_array.copy()


if horizontal:
    # Process each row
    for y in range(height):
        x = 0
        while x < width:
            # Skip non-sortable areas (mask <= 1)
            while x < width and mask_array[y, x] <= 10:
                x += 1
            if x == width:
                break
            start_x = x
            
            # Find the end of the sortable segment (mask > 1)
            while x < width and mask_array[y, x] > 10:
                x += 1
            end_x = x
            
            if end_x-start_x<min_segment_length and (x+target_segment_length)<width:
                end_x += random.randint(min_segment_length,target_segment_length)
            x=end_x
            # Extract pixels in the segment
            sortable_pixels = image_array[y, start_x:end_x]
            sortable_indices = list(range(start_x, end_x))
            
            if sortable_pixels.size > 0:
                # Sort pixels by brightness
                if color_mode == ColorMode.BRIGHTNESS:
                    pixel_info = np.dot(sortable_pixels, weights)
                elif color_mode == ColorMode.R:
                    pixel_info = [get_red(pixel) for pixel in sortable_pixels]
                elif color_mode == ColorMode.G:
                    pixel_info = [get_green(pixel) for pixel in sortable_pixels]
                elif color_mode == ColorMode.B:
                    pixel_info = [get_blue(pixel) for pixel in sortable_pixels]
                
                sorted_indices = np.argsort(pixel_info)
                sorted_pixels = sortable_pixels[sorted_indices]
                
                # Place sorted pixels back into the segment
                for i, x_pos in enumerate(sortable_indices):
                    output_array[y, x_pos] = sorted_pixels[i]
    # Save the result
    output_image = Image.fromarray(output_array)
    output_image.save("sorted_image_rows.jpg")

if vertical:
    # Process each column
    for x in range(width):
        y = 0
        while y < height:
            # Skip non-sortable areas (mask <= 1)
            while y < height and mask_array[y, x] > 10:
                y += 1
            if y == height:
                break
            start_y = y
            
            # Find the end of the sortable segment (mask > 1)
            while y < height and mask_array[y, x] <= 10:
                y += 1
            end_y = y
            
            if end_y-start_y<min_segment_length and (y+target_segment_length)<height:
                end_y += random.randint(min_segment_length,target_segment_length)
            y=end_y
            # Extract pixels in the segment
            sortable_pixels = image_array[start_y:end_y, x]
            sortable_indices = list(range(start_y, end_y))
            
            if sortable_pixels.size > 0:
                # Sort pixels by brightness
                if color_mode == ColorMode.BRIGHTNESS:
                    pixel_info = np.dot(sortable_pixels, weights)
                elif color_mode == ColorMode.R:
                    pixel_info = [get_red(pixel) for pixel in sortable_pixels]
                elif color_mode == ColorMode.G:
                    pixel_info = [get_green(pixel) for pixel in sortable_pixels]
                elif color_mode == ColorMode.B:
                    pixel_info = [get_blue(pixel) for pixel in sortable_pixels]
                sorted_indices = np.argsort(pixel_info)#[::-1]
                sorted_pixels = sortable_pixels[sorted_indices]
                
                # Place sorted pixels back into the segment
                for i, y_pos in enumerate(sortable_indices):
                    output_array[y_pos, x] = sorted_pixels[i]

    # Save the result
    output_image = Image.fromarray(output_array)
    output_image.save("sorted_image_columns.jpg")
