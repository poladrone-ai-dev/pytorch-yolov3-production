# import the necessary packages
import imutils
import pyimagesearch.global_var as global_var

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

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

def sliding_window(image, x_stepSize, y_stepSize, windowSize):

    for y in range(0, image.shape[0], windowSize[0] - y_stepSize):
        global_var.next_row = False
        for x in range(0, image.shape[1], x_stepSize):
            global_var.y_coord += 1
            print("x_coord: " + str(global_var.x_coord))
            print("y_coord: " + str(global_var.y_coord))

            # last_window = (global_var.y_coord == global_var.max_x and global_var.x_coord == global_var.max_y)
            if y + windowSize[0] > image.shape[0]:
                new_x = image.shape[0] - windowSize[0]
                if global_var.x_coord < global_var.max_y:
                    yield (new_x, y, image[new_x:image.shape[0], x:x + windowSize[0]])

            elif x + windowSize[1] > image.shape[1]:
                new_y = image.shape[1] - windowSize[1]
                if global_var.y_coord < global_var.max_x:
                    yield (x, new_y, image[y:y + windowSize[1], new_y:image.shape[1]])

            else:
                if global_var.y_coord < global_var.max_x:
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

            # yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

        global_var.next_row = True
        global_var.y_coord = -1
        global_var.x_coord += 1