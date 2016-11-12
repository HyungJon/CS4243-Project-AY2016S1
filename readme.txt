This readme will explain in brief what each python file does. Note that some of these files may not run successfully because of missing image/video files or folders, and because they must be run in a certain order, with necessary manual steps in between. So hopefully this will help you understand our code without actually having to run them.

extractAllFrames.py - Extract all the frames from the video as jpeg images

detectCornersAndLines.py - Auto-detect the court lines and corners in each video frame using color detection and Hough Line Transform. Also can detect the base of the yellow net poles. Not very accurate because of the noise in the frames

betterCornerDetection.py - A more accurate algorithm for auto-detecting the corner points. To use this program, first mark out the position of a corner point in the first frame, then the algo will track the corner point for the rest of the video by only searching in the region around the point.

extrapolation.py - This is to get the coordinates of the corner points that are out of view. It extends the lines that are drawn by detectCornersAndLines.py, and use that to determine the location of the non-visible corner point(s).

plotCornerByHand.py - For the corner points that cannot be autodetected or extrapolated accurately, we resort to plotting them by hand, but with the aid of this program we can do it quickly. Using this tool we first mark out the point using the mouse, and then subsequently we use the WASD keys to adjust the point from frame to frame.


computeHomography.py - compute the homography matrix with the extracted corners' coordinates from the corner detection

bg.py and bg_grayscale - used for extracting the background when we are trying to do background subtraction

extrapolateBall.py - Estimate the position and motion of the ball with the knowledge of when the ball is being hit or received by a player


Execute these programs in the following order to get a top down view of the court:
1) getTopDownCoords.py --> Transform the original coordinates of the pixels to the coordinates from the top down view

2) coordsSmoother.py --> Reduce the shaking of the top down view by making the movement of points from frame to frame smoother

3) drawTopDownView.py --> Output the frames as seen from the top down view, with the court outline shown and the players' and ball's locations labelled.
Also computes and outputs the statistics of the players (distance moved and number of jumps).


panorama.py - Stitch all the frames together to form a panorama 

outputVideo.py - Combine the original video, panorama video, top-down video and statistics together to form a 4-screen output video

playResult.y --> Retrieve frames from a folder and flash the frames on the screen (this is to 'play' the frames without having to convert the frames to video)

