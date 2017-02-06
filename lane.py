import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# Class for intinsic camera calibration
class cam_calibration():
    def __init__(self):
        # Prepare object points
        xn = 9 # Number of inside corners in any given row
        yn = 6 # Number of inside corners in any given column
        # Load checkboard images
        images = glob.glob("camera_cal/calibration*.jpg")
        # Initialise image and object point arrays
        objpoints = []
        imgpoints = []
        # Generate object points
        objp = np.zeros((xn*yn,3), np.float32)
        objp[:,:2] = np.mgrid[0:xn,0:yn].T.reshape(-1,2) # x, y coordinates
        for fname in images:
            # Read in image
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (xn, yn), None)
            if ret == True:
                # Fill image point and object point arrays
                imgpoints.append(corners)
                objpoints.append(objp)
        # Test undistortion on an image
        img_size = (img.shape[1], img.shape[0])
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        self.mtx = mtx
        self.dist = dist

# Class for perspective transform
class PTransform():
    def __init__(self):
        src = np.float32([[132,703],[550,466],[730,466],[1147,703]])
        dst = np.float32([[200,720],[200,0],[1100,0],[1100,720]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

# Class for setting up binary mask for images
class Binary_Threshold():
    def __init__(self, image):
        ksize = 3
        self.gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh_min=2, thresh_max=210)
        self.grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh_min=5, thresh_max=180)
        self.mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 200))
        self.dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.3, 1.))
        self.ylw_binary = self.yellow_select(image)
        self.wht_binary = self.white_select(image)
        combined_thresh = np.zeros_like(self.dir_binary)
        combined_thresh[((self.ylw_binary == 1) | (self.wht_binary == 1)) & ((self.gradx == 1)) | ((self.mag_binary == 1) & (self.dir_binary == 1))] = 1
        self.mask = combined_thresh

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return binary_output

    def yellow_select(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        v_channel = yuv[:,:,2]
        binary_output = np.zeros_like(v_channel)
        binary_output[(v_channel > 8) & (v_channel <= 112)] = 1
      
        return binary_output

    def white_select(self, img):
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y_channel = yuv[:,:,0]
        binary_output = np.zeros_like(y_channel)
        binary_output[(y_channel > 188) & (y_channel <= 255)] = 1

        return binary_output

# Implements a linear Kalman filter.
class KalmanFilter:
    def __init__(self,_A, _B, _H, _x, _P, _Q, _R):
        self.A = _A                      # State transition matrix.
        self.B = _B                      # Control matrix.
        self.H = _H                      # Observation matrix.
        self.current_state_estimate = _x # Initial state estimate.
        self.current_prob_estimate = _P  # Initial covariance estimate.
        self.Q = _Q                      # Estimated error in process.
        self.R = _R                      # Estimated error in measurements.
    def GetCurrentState(self):
        return self.current_state_estimate
    def Step(self,control_vector,measurement_vector):
        # prediction
        predicted_state_estimate = self.A * self.current_state_estimate + self.B * control_vector
        predicted_prob_estimate = (self.A * self.current_prob_estimate) * np.transpose(self.A) + self.Q
        # observation
        innovation = measurement_vector - self.H*predicted_state_estimate
        innovation_covariance = self.H*predicted_prob_estimate*np.transpose(self.H) + self.R
        # update
        kalman_gain = predicted_prob_estimate * np.transpose(self.H) * np.linalg.inv(innovation_covariance)
        self.current_state_estimate = predicted_state_estimate + kalman_gain * innovation
        size = self.current_prob_estimate.shape[0]
        self.current_prob_estimate = (np.eye(size)-kalman_gain*self.H)*predicted_prob_estimate



# Class for lane detection on video
class lane_detect():
    def __init__(self, mtx, dist, M, Minv, KF=True):
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.Minv = Minv
        self.initlane = False
        A = np.matrix([1])
        H = np.matrix([1])
        B = np.matrix([0])
        Q = np.matrix([0.00001])
        R = np.matrix([0.1])
        xhat = np.matrix([0])
        P    = np.matrix([1])
        self.offsetfilter = KalmanFilter(A,B,H,xhat,P,Q,R)
        A = np.matrix([1])
        H = np.matrix([1])
        B = np.matrix([0])
        Q = np.matrix([0.00001])
        R = np.matrix([0.1])
        xhat = np.matrix([0])
        P    = np.matrix([1])
        self.curvefilter = KalmanFilter(A,B,H,xhat,P,Q,R)
        

    def drawlaneline(self, image):
        image = cv2.undistort(image, mtx, dist, None, mtx)
        self.mask = Binary_Threshold(image).mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imshape = self.mask.shape
        self.binary_warped = cv2.warpPerspective(self.mask, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        self.image_warped = cv2.warpPerspective(image, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
              
        if not self.initlane:
            self.lane_init()
        else:
            self.lane_update()

        margin = 15
        window_img = np.zeros_like(self.image_warped)
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,0,255))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))

        left_line = np.array([np.transpose(np.vstack([self.left_fitx[::-1], self.ploty[::-1]]))])
        right_line = np.array([np.transpose(np.vstack([self.right_fitx, self.ploty]))])
        line_pts = np.hstack((left_line, right_line))
        cv2.fillPoly(window_img, np.int_([line_pts]), (255,0,0))

        laneimage = cv2.warpPerspective(window_img, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        self.outimage = cv2.addWeighted(image, 1, laneimage, 0.7, 0)


        # lane curvature
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
        self.left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1]*xm_per_pix)**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1]*xm_per_pix)**2)**1.5) / np.absolute(2*self.right_fit[0])
        self.curvature = (self.left_curverad + self.right_curverad) / 2.0
        self.curvefilter.Step(np.matrix([0]),np.matrix([self.curvature]))
        cv2.putText(self.outimage, 'Radius of Curvature = %dm' % self.curvefilter.GetCurrentState()[0][0], (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # vehicle offset
        self.offset = ((self.left_fitx[10] + self.right_fitx[10] - imshape[1]) / 2) * xm_per_pix
        self.offsetfilter.Step(np.matrix([0]),np.matrix([self.offset]))
        if self.offsetfilter.GetCurrentState()[0][0] > 0:
            self.side = 'right'
        else:
            self.side = 'left'
        cv2.putText(self.outimage, 'Offset from Center = %.2fm %s' % (np.abs(self.offsetfilter.GetCurrentState()[0][0]), self.side), (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    def lane_init(self):
        binary_warped = self.binary_warped
        # Take a histogram of the image
        histogram = np.sum(binary_warped[:,:], axis=0)         
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 11
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
        self.ploty = ploty
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        self.initlane = True

    def lane_update(self):
        binary_warped = self.binary_warped
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 60
        self.left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin))) 
        self.right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds] 
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        self.left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]


if __name__ == "__main__":
    from moviepy.editor import *
    # camera calibration
    undistort = cam_calibration()
    mtx = undistort.mtx
    dist = undistort.dist
    # percpeitive transform
    M = PTransform().M
    Minv = PTransform().Minv
    # initialize lane line ditection
    init_lane_detecter = False
    lane_detecter = lane_detect(mtx, dist, M, Minv)

    # load video
    filename = 'project_video.mp4'
    video = VideoFileClip(filename)

    clip = []
    for img in video.iter_frames():
        if not init_lane_detecter:
            lane_detecter.drawlaneline(img)
            outimg = lane_detecter.outimage
            init_lane_detecter = True
        else:
            lane_detecter.drawlaneline(img)
            outimg = lane_detecter.outimage
        
        cv2.imshow('frame', outimg)
        cv2.waitKey(10)

        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        clip += [outimg]

    outvideo = ImageSequenceClip(clip, fps=12)
    outvideo.write_videofile("annotated_project_video.mp4")