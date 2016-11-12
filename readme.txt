detectCornersAndLines.py - Auto-detect the court lines and corners in each video frame using color detection and Hough Line Transform. Also can detect the base of the yellow net poles. Not very accurate because of the noise in the frames

betterCornerDetection.py - A more accurate algorithm for auto-detecting the corner points. First mark out the position of a corner point, then the algo will track the corner point by only searching through the region around the point.

computeHomography.py - compute the homography matrix with the extracted corners' coordinates from the corner detection

extrapolateBall.py - Estimate the position and motion of the ball with the knowledge of when the ball is being hit or received by a player

panorama.py - Stitch all the frames together to form a panorama 

outputVideo.py - Combine the original video, panorama video, top-down video and statistics together to form a 4-screen output video

Execute these programs in the following order to get a top down view of the court:
getTopDownCoords.py
coordsSmoother.py
drawTopDownView.py