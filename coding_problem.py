import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the new image
image_path = 'red.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply binary thresholding to isolate cones (using manual thresholding for now)
_, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours of the cones
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours (optional step for visualization)
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Now we extract the centroid of each contour to use them as points for clustering
cone_centers = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cone_centers.append((cX, cY))

# Separate points into left and right clusters based on their x-values (simplified for this path)
cone_centers = np.array(cone_centers)
mid_x = image.shape[1] // 2

left_cones = cone_centers[cone_centers[:, 0] < mid_x]
right_cones = cone_centers[cone_centers[:, 0] > mid_x]

# Fit lines for both left and right cones using linear regression (polyfit)
if len(left_cones) > 1:
    left_fit = np.polyfit(left_cones[:, 1], left_cones[:, 0], 1)  # x = m*y + c
if len(right_cones) > 1:
    right_fit = np.polyfit(right_cones[:, 1], right_cones[:, 0], 1)

# Generate points for the lines
y_vals = np.arange(0, image.shape[0])

if len(left_cones) > 1:
    left_x_vals = left_fit[0] * y_vals + left_fit[1]
if len(right_cones) > 1:
    right_x_vals = right_fit[0] * y_vals + right_fit[1]

# Draw the lines on the image
output_img = image.copy()
if len(left_cones) > 1:
    for i in range(len(y_vals)-1):
        cv2.line(output_img, (int(left_x_vals[i]), y_vals[i]), (int(left_x_vals[i+1]), y_vals[i+1]), (255, 0, 0), 2)

if len(right_cones) > 1:
    for i in range(len(y_vals)-1):
        cv2.line(output_img, (int(right_x_vals[i]), y_vals[i]), (int(right_x_vals[i+1]), y_vals[i+1]), (255, 0, 0), 2)

# Save the result
output_path = "answer.png"
cv2.imwrite(output_path, output_img)

# Show the result
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

output_path
