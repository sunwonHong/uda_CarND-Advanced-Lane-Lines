import numpy as np
import sys
import cv2
import copy
import matplotlib.image as npimg
import matplotlib.pyplot as plt

class backup_par:
    def __init__(self):
        self.leftpts_x = None
        self.rightpts_x = None
        self.leftpts_y = None
        self.rightpts_y = None

bu_par = backup_par()

def find_sobel(img, orient='x', ks= 3, min_=30, max_=100):
    
    temp_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(temp_gray, cv2.CV_64F, 1, 0, ks))
    if orient == 'y':
        sobel = np.absolute(cv2.Sobel(temp_gray, cv2.CV_64F, 0, 1, ks))
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    result = np.zeros_like(scaled_sobel)
    result[(scaled_sobel >= min_) & (scaled_sobel <= max_)] = 1
    return result

def find_mag(img, ks=3, tr=(0, 255)):
    temp_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(temp_gray, cv2.CV_64F, 1, 0, ksize=ks)
    sobel_y = cv2.Sobel(temp_gray, cv2.CV_64F, 0, 1, ksize=ks)
    magimg = np.sqrt(sobel_x**2 + sobel_y**2)
    magimg = (magimg/np.max(magimg)/255).astype(np.uint8) 
    result = np.zeros_like(magimg)
    result[(magimg >= tr[0]) & (magimg <= tr[1])] = 1
    return result

def find_dir(img, ks=3, tr=(0, np.pi/2)):
    temp_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(temp_gray, cv2.CV_64F, 1, 0, ksize=ks)
    sobel_y = cv2.Sobel(temp_gray, cv2.CV_64F, 0, 1, ksize=ks)
    dirimg = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    result =  np.zeros_like(dirimg)
    result[(dirimg >= tr[0]) & (dirimg <= tr[1])] = 1
    return result

def find_pix(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    out_img = np.dstack((img, img, img))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_normal = np.argmax(histogram[:midpoint])
    rightx_normal = np.argmax(histogram[midpoint:]) + midpoint

    wd_height = np.int(img.shape[0]//9)
    nonzeroimg = img.nonzero()
    nonzeroimg_y = np.array(nonzeroimg[0])
    nonzeroimg_x = np.array(nonzeroimg[1])
    leftx_now = leftx_normal
    rightx_now = rightx_normal

    left_lane_ind = []
    right_lane_ind = []

    for wd in range(9):
        wd_y_low = img.shape[0] - (wd+1)*wd_height
        wd_y_high = img.shape[0] - wd*wd_height
        wd_xleft_low = leftx_now - 100
        wd_xleft_high = leftx_now + 100
        wd_xright_low = rightx_now - 100
        wd_xright_high = rightx_now + 100
        
        cv2.rectangle(out_img,(wd_xleft_low,wd_y_low),
        (wd_xleft_high,wd_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(wd_xright_low,wd_y_low),
        (wd_xright_high,wd_y_high),(0,255,0), 2) 
        
        good_left_ind = ((nonzeroimg_y >= wd_y_low) & (nonzeroimg_y < wd_y_high) & 
        (nonzeroimg_x >= wd_xleft_low) &  (nonzeroimg_x < wd_xleft_high)).nonzero()[0]
        good_right_ind = ((nonzeroimg_y >= wd_y_low) & (nonzeroimg_y < wd_y_high) & 
        (nonzeroimg_x >= wd_xright_low) &  (nonzeroimg_x < wd_xright_high)).nonzero()[0]
        
        left_lane_ind.append(good_left_ind)
        right_lane_ind.append(good_right_ind)
        
        if len(good_left_ind) > 50:
            leftx_now = np.int(np.mean(nonzeroimg_x[good_left_ind]))
        if len(good_right_ind) > 50:        
            rightx_now = np.int(np.mean(nonzeroimg_x[good_right_ind]))

    try:
        left_lane_ind = np.concatenate(left_lane_ind)
        right_lane_ind = np.concatenate(right_lane_ind)
    except ValueError:
        pass

    leftx = nonzeroimg_x[left_lane_ind]
    lefty = nonzeroimg_y[left_lane_ind] 
    rightx = nonzeroimg_x[right_lane_ind]
    righty = nonzeroimg_y[right_lane_ind]

    return leftx, lefty, rightx, righty, out_img

def find_polynomial(img):
    leftx, lefty, rightx, righty, out_img = find_pix(img)

    if len(leftx) > 0:
        bu_par.leftpts_x = leftx
    if len(lefty) > 0:
        bu_par.leftpts_y = lefty
    if len(rightx) > 0:
        bu_par.rightpts_x = rightx
    if len(righty) > 0:
        bu_par.rightpts_y = righty

    left_fit = np.polyfit(bu_par.leftpts_y, bu_par.leftpts_x, 2)
    right_fit = np.polyfit(bu_par.rightpts_y, bu_par.rightpts_x, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img

def find_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def find_near_poly(binary_warped):
    margin = 100

    nonzeroimg = binary_warped.nonzero()
    nonzeroimg_y = np.array(nonzeroimg[0])
    nonzeroimg_x = np.array(nonzeroimg[1])
    
    left_lane_ind = ((nonzeroimg_x > (left_fit[0]*(nonzeroimg_y**2) + left_fit[1]*nonzeroimg_y + 
                    left_fit[2] - 100)) & (nonzeroimg_x < (left_fit[0]*(nonzeroimg_y**2) + 
                    left_fit[1]*nonzeroimg_y + left_fit[2] + 100)))
    right_lane_ind = ((nonzeroimg_x > (right_fit[0]*(nonzeroimg_y**2) + right_fit[1]*nonzeroimg_y + 
                    right_fit[2] - 100)) & (nonzeroimg_x < (right_fit[0]*(nonzeroimg_y**2) + 
                    right_fit[1]*nonzeroimg_y + right_fit[2] + 100)))
    
    leftx = nonzeroimg_x[left_lane_ind]
    lefty = nonzeroimg_y[left_lane_ind] 
    rightx = nonzeroimg_x[right_lane_ind]
    righty = nonzeroimg_y[right_lane_ind]

    if len(leftx) > 0:
        bu_par.leftpts_x = leftx
    if len(lefty) > 0:
        bu_par.leftpts_y = lefty
    if len(rightx) > 0:
        bu_par.rightpts_x = rightx
    if len(righty) > 0:
        bu_par.rightpts_y = righty
	
    left_fitx, right_fitx, ploty = find_poly(binary_warped.shape, bu_par.leftpts_x, bu_par.leftpts_y, bu_par.rightpts_x, bu_par.rightpts_y)
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    wd_img = np.zeros_like(out_img)
    out_img[nonzeroimg_y[left_lane_ind], nonzeroimg_x[left_lane_ind]] = [255, 0, 0]
    out_img[nonzeroimg_y[right_lane_ind], nonzeroimg_x[right_lane_ind]] = [0, 0, 255]

    left_line_wd1 = np.array([np.transpose(np.vstack([left_fitx-100, ploty]))])
    left_line_wd2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+100, ploty])))])
    left_line_pts = np.hstack((left_line_wd1, left_line_wd2))
    right_line_wd1 = np.array([np.transpose(np.vstack([right_fitx-100, ploty]))])
    right_line_wd2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+100, ploty])))])
    right_line_pts = np.hstack((right_line_wd1, right_line_wd2))

    cv2.fillPoly(wd_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(wd_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, wd_img, 0.3, 0)
    
    return result, left_fitx, right_fitx

##find chess
ob_points = []
img_points = []

ob_pt = np.zeros((5*9,3), np.float32)
ob_pt[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)

cal_img = cv2.imread("./camera_cal/calibration1.jpg")

cal_grayimg = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(cal_grayimg, (9, 5), None)

drawchess = True
if drawchess == True:
    cv2.drawChessboardCorners(cal_img, (9, 5), corners, ret)
    cv2.imshow("img", cal_img)
    cv2.imwrite("./output_images/drawchess.jpg", cal_img)
    cv2.waitKey(0)

ob_points.append(ob_pt)
img_points.append(corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(ob_points, img_points, cal_grayimg.shape[::-1], None, None)

##undistort
chess_dst = cv2.undistort(cal_img, mtx, dist, None, mtx)

chess_undistort = True
if chess_undistort == True:
    cv2.imshow("chess_dst", chess_dst)
    cv2.imwrite("./output_images/chess_undistort.jpg", chess_dst)
    cv2.waitKey(0)

##make sure that input is image or video
# input_type = 'image'
input_type = 'video'

#image
if input_type == 'image':
	road_img = cv2.imread("./test_images/test1.jpg")
	road_dst = cv2.undistort(road_img, mtx, dist, None, mtx)

	road_undistort = True
	if road_undistort == True:
		cv2.imshow("img", road_dst)
		cv2.imwrite("./output_images/road_undistort.jpg", road_dst)
		cv2.waitKey(0)

	#sobel
	# sobel_x = find_sobel(road_dst, 'x', 3, 35, 100)
	# sobel_y = find_sobel(road_dst, 'y', 3, 30, 255)
	sobel_x = find_sobel(road_dst, 'x', 3, 35, 255)
	sobel_y = find_sobel(road_dst, 'y', 3, 15, 255)

	mag_binary = find_mag(road_dst, 3, (30, 255))
	dir_binary = find_dir(road_dst, 15, (0.7, 1.3))

	combined = np.zeros_like(dir_binary)
	combined[((sobel_x == 1) & (sobel_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

	road_binary = True
	if road_binary == True:
		cv2.imshow("combined", combined)
		cv2.imshow("sobel_x", sobel_x*255)
		cv2.imshow("sobel_y", sobel_y*255)
		cv2.imshow("mag_binary", mag_binary*255)
		cv2.imshow("dir_binary", dir_binary)
		cv2.imwrite("./output_images/road_binary.jpg", combined)
		cv2.waitKey(0)

	##hsl
	hls = cv2.cvtColor(road_dst, cv2.COLOR_RGB2HLS)
	l_img = hls[:,:,1]
	s_img = hls[:,:,2]
	# Sobel x
	sobel_x = np.absolute(cv2.Sobel(l_img, cv2.CV_64F, 1, 0))
	abs_sobel_x = np.absolute(sobel_x)
	scaled_sobel = np.uint8(255*sobel_x/np.max(sobel_x))

	# Threshold x gradient
	scaled_sobel_thr = np.zeros_like(scaled_sobel)
	scaled_sobel_thr[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

	# Threshold color channel
	s_img_thr = np.zeros_like(s_img)
	s_img_thr[(s_img >= 170) & (s_img <= 255)] = 1

	combined_thr_img = np.zeros_like(scaled_sobel_thr)
	combined_thr_img[(s_img_thr == 1) | (scaled_sobel_thr == 1)] = 1

	road_hsl_binary = True
	if road_hsl_binary == True:
		cv2.imshow("combined_thr_img", combined_thr_img*255)
		cv2.imwrite("./output_images/road_hsl_binary.jpg", combined_thr_img*255)
		cv2.waitKey(0)

	##grad + hsl
	final_binary = np.zeros_like(combined_thr_img).astype(np.uint8)
	final_binary[((combined >= 1) | (combined_thr_img >=1))] = 255

	road_final_binary = True
	if road_final_binary == True:
		cv2.imshow("final_binary", final_binary)
		cv2.imwrite("./output_images/road_final_binary.jpg", final_binary)
		cv2.waitKey(0)

	##warp
	input_coord = np.float32([[257, 719], [604,456], [719, 451], [1135, 717]])
	output_coord = np.float32([[257, 719],[257, 100],[1135, 100],[1135, 717]])

	M = cv2.getPerspectiveTransform(input_coord, output_coord)
	M_back = cv2.getPerspectiveTransform(output_coord, input_coord)

	road_img_shape = (final_binary.shape[1], final_binary.shape[0])

	warped_img = cv2.warpPerspective(final_binary, M, road_img_shape)
	road_warp = True
	if road_warp == True:
		cv2.imshow("warped_img", warped_img)
		cv2.imwrite("./output_images/road_warp.jpg", warped_img)
		cv2.waitKey(0)

	## ploy fit
	poly_fit_img = find_polynomial(warped_img)

	road_poly = True
	if road_poly == True:
		cv2.imshow("poly_fit_img", poly_fit_img)
		cv2.imwrite("./output_images/road_poly.jpg", poly_fit_img)
		cv2.waitKey(0)

	##should be previous value
	left_fit = np.array([-5.85273880e-04, 9.59699964e-01,-1.00255034e+02])
	right_fit = np.array([-5.95152593e-04, 1.00622601e+00, 7.34232276e+02])

	##search from prior
	lane_img, left_points, right_points = find_near_poly(warped_img)

	road_lane = True
	if road_lane == True:
		cv2.imshow("lane_img", lane_img)
		cv2.imwrite("./output_images/road_lane.jpg", lane_img)
		cv2.waitKey(0)

	ypixel_to_m = 30/720 
	xpixel_to_m = 3.7/700 

	ploty = np.linspace(0, 719, num=720)
	leftx = np.array([left_points])
	rightx = np.array([right_points])

	leftx = leftx[::-1] 
	rightx = rightx[::-1] 

	leftx_sq = np.squeeze(leftx, axis=0)
	rightx_sq = np.squeeze(rightx, axis=0)

	left_fit = np.polyfit(ploty*ypixel_to_m, leftx_sq*xpixel_to_m, 2)
	right_fit = np.polyfit(ploty*ypixel_to_m, rightx_sq*xpixel_to_m, 2)

	y_eval = np.max(ploty)
	    
	##calculate curvature
	left_curverad = ((1 + (2*left_fit[0]*y_eval*ypixel_to_m + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
	right_curverad = ((1 + (2*right_fit[0]*y_eval*ypixel_to_m + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
	print("left_curverad = ")
	print(left_curverad)
	print("right_curverad = ")
	print(right_curverad)
	print("min_curverad = ")
	print((left_curverad+right_curverad)/2)

	##calculate how long distance from middle
	print("moved from middle(+ is to right, - is to left) = ")
	print(abs(640-(leftx_sq[-1]+rightx_sq[-1])/2)*xpixel_to_m)

	##draw line
	drawed_lane_img = np.zeros_like(lane_img)
	left_plotx, right_plotx = leftx_sq, rightx_sq

	left_pts_l = np.array([np.transpose(np.vstack([leftx_sq - 56/5, ploty]))])
	left_pts_r = np.array([np.flipud(np.transpose(np.vstack([leftx_sq + 56/5, ploty])))])
	left_pts = np.hstack((left_pts_l, left_pts_r))
	right_pts_l = np.array([np.transpose(np.vstack([rightx_sq - 56/5, ploty]))])
	right_pts_r = np.array([np.flipud(np.transpose(np.vstack([rightx_sq + 56/5, ploty])))])
	right_pts = np.hstack((right_pts_l, right_pts_r))

	cv2.fillPoly(drawed_lane_img, np.int_([left_pts]), (255, 0, 255))
	cv2.fillPoly(drawed_lane_img, np.int_([right_pts]), (255, 0, 255))

	pts_left = np.array([np.transpose(np.vstack([leftx_sq+56/5, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-56/5, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	cv2.fillPoly(drawed_lane_img, np.int_([pts]), (0, 255, 0))
	result = cv2.addWeighted(lane_img, 1, drawed_lane_img, 0.3, 0)

	color_result = cv2.warpPerspective(drawed_lane_img, M_back, (1280, 720))
	lane_to_img = np.zeros_like(road_dst)
	lane_to_img = color_result

	result = cv2.addWeighted(road_dst, 1, lane_to_img, 0.3, 0)

	final_lane_result = True
	if final_lane_result == True:
		text = 'Lane curvature: ' + '{:.0f}'.format((left_curverad+right_curverad)/2) + 'm'
		cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
		text = 'moved from middle(+ is to right, - is to left): ' + '{:.3f}'.format(abs(640-(leftx_sq[-1]+rightx_sq[-1])/2)*xpixel_to_m) + 'm'
		cv2.putText(result, text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imshow('result',result)
		cv2.imwrite("./output_images/final_lane_result.jpg", result)
		cv2.waitKey(0)

elif input_type == 'video':
	videoFile = "./project_video.mp4"
	cap = cv2.VideoCapture(videoFile)

	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

	while(cap.isOpened()):
		ret, frame = cap.read()

		road_img = frame
		road_dst = cv2.undistort(road_img, mtx, dist, None, mtx)

		#sobel
		# sobel_x = find_sobel(road_dst, 'x', 3, 35, 100)
		# sobel_y = find_sobel(road_dst, 'y', 3, 30, 255)
		sobel_x = find_sobel(road_dst, 'x', 3, 35, 255)
		sobel_y = find_sobel(road_dst, 'y', 3, 15, 255)
		mag_binary = find_mag(road_dst, 3, (30, 255))
		dir_binary = find_dir(road_dst, 15, (0.7, 1.3))

		combined = np.zeros_like(dir_binary)
		combined[((sobel_x == 1) & (sobel_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

		##hsl
		hls = cv2.cvtColor(road_dst, cv2.COLOR_RGB2HLS)
		l_img = hls[:,:,1]
		s_img = hls[:,:,2]
		# Sobel x
		sobel_x = np.absolute(cv2.Sobel(l_img, cv2.CV_64F, 1, 0))
		scaled_sobel = np.uint8(255*sobel_x/np.max(sobel_x))

		# Threshold x gradient
		scaled_sobel_thr = np.zeros_like(scaled_sobel)
		scaled_sobel_thr[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

		# Threshold color channel
		s_img_thr = np.zeros_like(s_img)
		s_img_thr[(s_img >= 170) & (s_img <= 255)] = 1

		combined_thr_img = np.zeros_like(scaled_sobel_thr)
		combined_thr_img[(s_img_thr == 1) | (scaled_sobel_thr == 1)] = 1

		##grad + hsl
		final_binary = np.zeros_like(combined_thr_img).astype(np.uint8)
		final_binary[((combined >= 1) | (combined_thr_img >=1))] = 255

		##warp
		input_coord = np.float32([[257, 719], [604,456], [719, 451], [1135, 717]])
		output_coord = np.float32([[257, 719],[257, 100],[1135, 100],[1135, 717]])

		M = cv2.getPerspectiveTransform(input_coord, output_coord)
		M_back = cv2.getPerspectiveTransform(output_coord, input_coord)

		road_img_shape = (final_binary.shape[1], final_binary.shape[0])

		warped_img = cv2.warpPerspective(final_binary, M, road_img_shape)

		## ploy fit
		poly_fit_img = find_polynomial(warped_img)

		##should be previous value
		left_fit = np.array([-5.85273880e-04, 9.59699964e-01,-1.00255034e+02])
		right_fit = np.array([-5.95152593e-04, 1.00622601e+00, 7.34232276e+02])

		##search from prior
		lane_img, left_points, right_points = find_near_poly(warped_img)

		ypixel_to_m = 30/720 
		xpixel_to_m = 3.7/700 

		ploty = np.linspace(0, 719, num=720)
		leftx = np.array([left_points])

		rightx = np.array([right_points])

		leftx = leftx[::-1]  
		rightx = rightx[::-1]  

		leftx_sq = np.squeeze(leftx, axis=0)
		rightx_sq = np.squeeze(rightx, axis=0)

		left_fit = np.polyfit(ploty*ypixel_to_m, leftx_sq*xpixel_to_m, 2)
		right_fit = np.polyfit(ploty*ypixel_to_m, rightx_sq*xpixel_to_m, 2)

		y_eval = np.max(ploty)
		
		##calculate curvature
		left_curverad = ((1 + (2*left_fit[0]*y_eval*ypixel_to_m + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
		right_curverad = ((1 + (2*right_fit[0]*y_eval*ypixel_to_m + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
		print("left_curverad = ")
		print(left_curverad)
		print("right_curverad = ")
		print(right_curverad)
		print("min_curverad = ")
		print((left_curverad+right_curverad)/2)

		##calculate how long distance from middle
		print("moved from middle(+ is to right, - is to left) = ")
		print(abs(640-(leftx_sq[-1]+rightx_sq[-1])/2)*xpixel_to_m)
		
		##draw line
		drawed_lane_img = np.zeros_like(lane_img)
		left_plotx, right_plotx = leftx_sq, rightx_sq

		left_pts_l = np.array([np.transpose(np.vstack([leftx_sq - 56/5, ploty]))])
		left_pts_r = np.array([np.flipud(np.transpose(np.vstack([leftx_sq + 56/5, ploty])))])
		left_pts = np.hstack((left_pts_l, left_pts_r))
		right_pts_l = np.array([np.transpose(np.vstack([rightx_sq - 56/5, ploty]))])
		right_pts_r = np.array([np.flipud(np.transpose(np.vstack([rightx_sq + 56/5, ploty])))])
		right_pts = np.hstack((right_pts_l, right_pts_r))

		cv2.fillPoly(drawed_lane_img, np.int_([left_pts]), (255, 0, 255))
		cv2.fillPoly(drawed_lane_img, np.int_([right_pts]), (255, 0, 255))

		pts_left = np.array([np.transpose(np.vstack([leftx_sq+56/5, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-56/5, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		cv2.fillPoly(drawed_lane_img, np.int_([pts]), (0, 255, 0))
		result = cv2.addWeighted(lane_img, 1, drawed_lane_img, 0.3, 0)

		color_result = cv2.warpPerspective(drawed_lane_img, M_back, (1280, 720))
		lane_to_img = np.zeros_like(road_dst)
		lane_to_img = color_result

		result = cv2.addWeighted(road_dst, 1, lane_to_img, 0.3, 0)

		text = 'Lane curvature: ' + '{:.0f}'.format((left_curverad+right_curverad)/2) + 'm'
		cv2.putText(result, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
		text = 'moved from middle(+ is to right, - is to left): ' + '{:.3f}'.format(abs(640-(leftx_sq[-1]+rightx_sq[-1])/2)*xpixel_to_m) + 'm'
		cv2.putText(result, text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

		cv2.imshow("video", result)
		cv2.waitKey(1)
		out.write(result)		

	cap.release()
	out.release()
	cv2.destroyAllwds()