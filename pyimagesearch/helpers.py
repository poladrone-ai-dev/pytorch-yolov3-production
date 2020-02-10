# import the necessary packages
import imutils
import pyimagesearch.global_var as global_var

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

def sliding_window(image, x_stepSize, y_stepSize, windowSize):
    for y in range(0, image.shape[0], windowSize[0] - y_stepSize):
        for x in range(0, image.shape[1], x_stepSize):
            global_var.y_coord += 1

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

        global_var.y_coord = -1
        global_var.x_coord += 1