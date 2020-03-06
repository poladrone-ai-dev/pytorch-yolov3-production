# import the necessary packages
import imutils

# smaller scale yields more layers, larger scales yields fewer layers
def pyramid(image, scale=2, minSize=(1250, 1250)):
    # yield the original image
    yield image
    print("Yield original image of size [" + str(image.shape[1]) + ", " + str(image.shape[0]) + "]")

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image
        print("Yield rescaled image of size [" + str(image.shape[1]) + ", " + str(image.shape[0]) + "]")

        
def sliding_window(image, x_stepSize, y_stepSize, windowSize, x_coord, y_coord):
	# slide a window across the image
	y1 = 0
	y2 = 0

	for y in range(0, image.shape[0], y_stepSize):
	
		if ((y + (windowSize[1])) > image.shape[0]) :		
			y1 = (image.shape[0]) - windowSize[1]
			y2 = image.shape[0]
		else :
			y1 = y
			y2 = y + (windowSize[1])  		
		
		x1 = 0 
		x2 = 0
		for x in range(0, image.shape[1], x_stepSize):	
		
			if ((x + (windowSize[0])) > image.shape[1]) :	
			   x1 =(image.shape[1]) - windowSize[0]
			   x2 = image.shape[1]
			else :
			   x1 = x
			   x2 = x + (windowSize[0])  
		
			y_coord += 1	
			yield (x1, y1, image[y1:y2, x1:x2], x_coord, y_coord)
			
			
		y_coord = -1
		x_coord += 1