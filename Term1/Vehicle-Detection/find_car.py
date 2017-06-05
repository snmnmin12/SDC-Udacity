import cv2 
import numpy as np
import pickle
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
from skimage.feature import hog
from lesson import *
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    img = img.astype(np.float32)/255
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        # print(features)
        # X_scaler = StandardScaler().fit(features.reshape(1,-1))
        # test_features = X_scaler.transform(features.reshape(1,-1))
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # print('test_features:\n', test_features)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
#     draw_img = np.copy(img)
    ystart, ystop = y_start_stop[0], y_start_stop[1]
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
#     print('ch1 shape: ', ch1.shape)
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
#     print('hog1 shape: ', hog1.shape)
#     print('nblocks_per_window: ', nblocks_per_window)
    all_windows = []
    hot_windows = []
    for xb in range(nxsteps+1):
        for yb in range(nysteps+1):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            #print(spatial_features.shape, hist_features.shape, hog_features.shape)
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            
            all_windows.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
            if test_prediction == 1:
#                 cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                hot_windows.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
    return all_windows, hot_windows

from scipy.ndimage.measurements import label
##add heapmap
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

##If your classifier is working well, then the "hot" parts of the map are where the cars are, 
##and by imposing a threshold, you can reject areas affected by false positives. So let's write
##a function to threshold the map as well.
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def process_image(image):
	draw_img = np.copy(image)
	scales = [1,1.2,1.5]
	y_start_stop = [(400,500),(400,500),(400,550)]
	hot_windows = []
	for i,scale in enumerate(scales):
	    all_areas,hot_areas = find_cars(image, y_start_stop[i], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)                  
	    hot_windows += hot_areas
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat,hot_windows)
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,1)
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(draw_img, labels)
	return draw_img
    #plt.imshow(draw_img)

def process_image2(image):
	
	draw_image = np.copy(image)

	color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off

	s_windows = [(64,64),(96,96),(128,128)]
	y_start_stops = [(400, 500),(400,500),(400,600)]
	overlaps = [(0.25,0.25),(0.5,0.5),(0.75,0.75)]
	# all_windows = []
	hot_windows = []

	for i,window in enumerate(s_windows):
	    
	    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stops[i], 
	                        xy_window=window, xy_overlap=overlaps[i])
	    
	    hot_areas = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
	                            spatial_size=spatial_size, hist_bins=hist_bins, 
	                            orient=orient, pix_per_cell=pix_per_cell, 
	                            cell_per_block=cell_per_block, 
	                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
	                            hist_feat=hist_feat, hog_feat=hog_feat)
	    # all_windows += windows
	    hot_windows += hot_areas

	window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

	# return window_img
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	# Add heat to each box in box list
	heat = add_heat(heat, hot_windows)
	# Apply threshold to help remove false positives
	heat = apply_threshold(heat,1)
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(draw_image, labels)
	return draw_img

if __name__=='__main__':

	image = mpimg.imread('test_images/test4.jpg')
	# spatial_feat = True # Spatial features on or off
	# hist_feat = True # Histogram features on or off
	# hog_feat = True # HOG features on or off
	# ystart = 400
	# ystop = 656
	# scale = 2
	dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
	svc = dist_pickle["svc"]
	X_scaler = dist_pickle["scaler"]
	orient = dist_pickle["orient"]
	pix_per_cell = dist_pickle["pix_per_cell"]
	cell_per_block = dist_pickle["cell_per_block"]
	spatial_size = dist_pickle["spatial_size"]
	hist_bins = dist_pickle["hist_bins"]

	print('type of svc: ', type(svc))
	print('orient: ', orient)
	print('pix_per_cell: ', pix_per_cell)
	print('cell_per_block: ',cell_per_block)
	print('spatial_size: ', spatial_size)
	print('hist_bins: ', hist_bins)

	#areas
	# out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)                  
	
	# out_img = process_image2(image)

	# plt.imshow(out_img)
	# plt.show()
	test_video = 'project_video.mp4'
	output = 'output3.mp4'
	clip1 = VideoFileClip(test_video)
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(output, audio=False)

	# heat = np.zeros_like(image[:,:,0]).astype(np.float)
	# # Add heat to each box in box list
	# heat = add_heat(heat,areas)
	    
	# # Apply threshold to help remove false positives
	# heat = apply_threshold(heat,1)

	# # Visualize the heatmap when displaying    
	# heatmap = np.clip(heat, 0, 255)

	# # Find final boxes from heatmap using label function
	# labels = label(heatmap)
	# draw_img = draw_labeled_bboxes(np.copy(image), labels)
	# plt.imshow(draw_img)
	# plt.show()