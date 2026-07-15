import cv2 as cv

# 1. Your Custom A1 Board Parameters
squares_X = 6
squares_Y = 8

# Using standard DICT_5X5_100 (Extremely reliable for this size)
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)

# The physical lengths don't strictly matter for the image generation, 
# but we define them here to match the 9cm reality.
square_length = 0.09 # 9 cm
marker_length = 0.07 # 7 cm

# 2. Create the board object
board = cv.aruco.CharucoBoard((squares_X, squares_Y), square_length, marker_length, dictionary)

# 3. Generate a 4K-equivalent High-Res Image for A1 Printing
# (400 pixels per square = 2400 x 3200 image)
width_pixels = squares_X * 400
height_pixels = squares_Y * 400
board_image = board.generateImage((width_pixels, height_pixels))

# 4. Save it
cv.imwrite("A1_Charuco_Board_6x8.png", board_image)
print("Massive 6x8 board saved! Send this to the print shop.")