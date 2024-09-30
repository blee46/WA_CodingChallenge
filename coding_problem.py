import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load image
image = cv2.imread('red.png')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for red cones (adjust if needed)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Threshold the image to isolate red cones
mask = cv2.inRange(hsv, lower_red, upper_red)

# Apply morphological operations to remove noise (optional)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours of the cones
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to remove small or irrelevant contours
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

# Extract cone centers
cone_centers = []
for cnt in filtered_contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cone_centers.append((cX, cY))

# Separate points into left and right clusters based on x-coordinate
cone_centers = np.array(cone_centers)
mid_x = image.shape[1] // 2
left_cones = cone_centers[cone_centers[:, 0] < mid_x]
right_cones = cone_centers[cone_centers[:, 0] > mid_x]

# Fit lines using polyfit (adjust degree if needed)
if len(left_cones) > 1:
    left_fit = np.polyfit(left_cones[:, 1], left_cones[:, 0], 1)
if len(right_cones) > 1:
    right_fit = np.polyfit(right_cones[:, 1], right_cones[:, 0], 1)

# Generate points for lines
y_vals = np.linspace(0, image.shape[0] - 1, 100)
if len(left_cones) > 1:
    left_x_vals = left_fit[0] * y_vals + left_fit[1]
if len(right_cones) > 1:
    right_x_vals = right_fit[0] * y_vals + right_fit[1]

# Draw lines on the image
if len(left_cones) > 1:
    pts = np.array([np.transpose(np.vstack([left_x_vals, y_vals]))], dtype=np.int32)
    cv2.polylines(image, pts, isClosed=False, color=(255, 0, 0), thickness=2)
if len(right_cones) > 1:
    pts = np.array([np.transpose(np.vstack([right_x_vals, y_vals]))], dtype=np.int32)
    cv2.polylines(image, pts, isClosed=False, color=(255, 0, 0), thickness=2)

# Save output
cv2.imwrite("answer.png", image)

# Display the result
# Load the modified image
modified_image = cv2.imread("answer.png")

# Convert from BGR to RGB for display
modified_image_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(modified_image_rgb)
plt.axis('off')  # Turn off axis labels and ticks
plt.title("Modified Image")
plt.show()