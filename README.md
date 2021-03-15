## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---

[//]: # (Image References)

[image1]: ./output_images/chess_undistort.jpg "Undistorted"
[image2]: ./output_images/road_undistort.jpg "Road Transformed"
[image3]: ./output_images/road_final_binary.jpg "Binary Example"
[image4]: ./output_images/road_warp.jpg "Warp Example"
[image5]: ./output_images/road_poly.jpg "Fit Visual"
[image6]: ./output_images/road_lane.jpg "Fit Visual"
[image7]: ./output_images/final_lane_result.jpg "Output"
[video1]: .output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `main.py`).

Using the chessboard image, we used the findChessboardCorners function to find the position in the image using the findChessboardCorners function, and the camera calibration and distortion coefficients were obtained using the calibrateCamera function, and the distortion was corrected using the undistort function. 
This is the part designated as the find chess part in the main.py file.

The result is like this:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

A calibration session was performed using the obtained camera parameter coefficients in the actual image.
This is the part designated as the undistort part in the main.py file.

The result is like this:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Through edge detection with sobel and color split in hsl color space and then binarization using certain threshold value, the image has only black or white color with highlighten lanes.
This is the part designated as the sobel part and hsl part in the main.py file.

The result is like this:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I designated four coordinates from the near part to the far part including the lane area, and then warp by designating the four coordinates of the rectangular shape to be converted.
This is the part designated as the warp part in the main.py file.

The result is like this:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I set a small square roi of the image to get the histogram for each part and find the line.
This is the part designated as the poly fit part and search from prior part in the main.py file.

The result is like this:

![alt text][image5]

Then, the line found through them was marked again like this:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the difference between the pixels on the nearest left and right lines and the pixels on the next nearest left and right lines to calculate the curve of the lane, which is shown in the calculation curve section of the main.py

The mean of the pixels in the nearest left and right lines is taken to calculate how far away the vehicle is from the center, which is shown in the calculate how long distance from middle section of the main.py

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

As a result, Line that was obtained was drawn in the existing image by reverse perceptive.

The result is like this:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
and video result:

[video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There was a problem in that line detection was difficult in areas where obstacles such as vehicles appeared, or in shaded areas or areas where sunlight shines differently.
Considering these things, it is necessary to consider the characteristics of the actual line better and to construct an algorithm that can detect the line well even when the brightness changes.

