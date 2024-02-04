import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

foodIm = cv2.imread("Food.jpg", cv2.IMREAD_GRAYSCALE)

# QUESTION 1
invalidRange1 = [-5, 200] # contains a negative
invalidRange2 = [.75, 240.5] # contains non-int types
invalidRange3 = [230, 10] # contains a larger 1st element than 2nd element
validRange = [50, 200] 

def Scaling(inputIm, myRange):

    # check if 'myRange' is the proper type (list) and size (2)
    if (not isinstance(myRange, list)) or (len(myRange) != 2):
        print("Error: 'myRange' is not the type (list) or size (2)")

    minimum, maximum = myRange

    # ensure 'myRange' is a positive int with its first element being smaller than the second element
    if (not isinstance(minimum, int)) or (not isinstance(maximum, int)):
        print(f"Error: 'myRange' must only contain integers: [{minimum}, {maximum}]")
        return

    if (minimum < 0) or (maximum < 0):
        print(f"Error: 'myRange' contains a negative value: [{minimum}, {maximum}]")
        return

    if minimum >= maximum:
        print(f"Error: 'myRange' cannot contain a larger than or equal to first value than second value: [{minimum}, {maximum}]")
        return

    originalMin, originalMax = np.min(inputIm), np.max(inputIm)
    scale = (maximum - minimum) / (originalMax - originalMin)
    
    # np.arrange will generate an array with elements 0 to 255 in ascending order
    # np.clip will limit the values of the array to be between 0 and 255
    transFunc = ((np.arange(256) - originalMin) * scale + minimum).clip(0, 255).astype(np.uint8)
    scaledIm = (scale * (inputIm - originalMin) + minimum).clip(0, 255).astype(np.uint8)
    
    return scaledIm, transFunc

Scaling(foodIm, invalidRange1)
Scaling(foodIm, invalidRange2)
Scaling(foodIm, invalidRange3)
scaledFoodIm, transFunc = Scaling(foodIm, validRange)

plt.figure(figsize=(10,5)) # Figure 1
plt.plot(transFunc)
plt.title('Transformation Function')
plt.xlabel('Original Intensity Value')
plt.ylabel('Transformed Intensity Value')

# QUESTION 2:
def CalHist(inputIm, normalized=False):
    # .ravel() will flatten the image to be a 1D array
    # np.bincount calculates the histogram on 1D arrays
    histogram = np.bincount(inputIm.ravel(), minlength=256)
    
    if normalized:
        histogram = histogram / histogram.sum()
    return histogram

histogramScaledFood = CalHist(scaledFoodIm)
normalizedScaledFood = CalHist(scaledFoodIm, normalized=True)

plt.figure(figsize=(10, 5)) # Figure 2
plt.subplot(1, 2, 1)
plt.bar(range(256), histogramScaledFood, color='gray')
plt.title("Regular Histogram of scaledFoodIm")

plt.subplot(1, 2, 2)
plt.bar(range(256), normalizedScaledFood, color='gray')
plt.title("Normalized Histogram of scaledFoodIm")
plt.tight_layout()

# QUESTION 3: Following the 4 steps outlined on slide 'Point Processing Methods-- Discrete Histogram Equalization' of Ch.3.1
def HistEqualization(inputIm, min=0, maximumPossibleIntensity=255):
    # Step 1: Obtain the histogram
    histogram = np.bincount(inputIm.ravel(), minlength=256)

    # Adjust the histogram based on min and maximumPossibleIntensity
    histogram = histogram - min
    histogram = np.clip(histogram, 0, maximumPossibleIntensity - min)

    # Calculate the cumulative normalized histogram and transform function as before
    cumulativeHistogram = np.cumsum(histogram)
    normalizedCumulativeHistogram = cumulativeHistogram / float(cumulativeHistogram[-1])
    transFunc = ((maximumPossibleIntensity - min) * normalizedCumulativeHistogram).astype(np.uint8) + min

    # Step 4: Scan image and set pixel with the intensity
    enhancedIm = transFunc[inputIm]

    return enhancedIm, transFunc

startTime = time.time()
equalizedFoodIm, transFuncEqualization = HistEqualization(foodIm)
endTime = time.time()
print(f"Running time of HistEqualization function: {endTime - startTime:.6f} seconds")

# QUESTION 4:
startTime = time.time()
equalizedFoodImCV2 = cv2.equalizeHist(foodIm)
endTime = time.time()
print(f"Running time of cv2.equalizeHist function: {endTime - startTime:.6f} seconds")

# QUESTION 5
def BBHE(inputIm):
    # the mean intensity serves as a threshold to split the histogram
    meanIntensity = np.mean(inputIm)
    
    # split the image into two sub-images based on the mean
    lowerImage = inputIm[inputIm <= meanIntensity]
    upperImage = inputIm[inputIm > meanIntensity]
    
    # equalize each sub-image separately
    lowerEqualized, transFuncLower = HistEqualization(lowerImage,0, meanIntensity)
    upperEqualized, transFuncUpper = HistEqualization(upperImage, meanIntensity, 255)
    
    # combine the two halves to create a whole equalized image
    equalizedIm = inputIm.copy()
    equalizedIm[inputIm <= meanIntensity] = lowerEqualized
    equalizedIm[inputIm > meanIntensity] = upperEqualized

    transFunc = np.concatenate([transFuncLower, transFuncUpper])
    
    return equalizedIm, transFunc

startTime = time.time()
BBHEFoodIm, transFuncBBHE = BBHE(foodIm)
endTime = time.time()

print(f"Running time of BBHE function: {endTime - startTime:.6f} seconds")

# Display enhanced images side by side
plt.figure(figsize=(18, 6)) # Figure 3
plt.subplot(1, 3, 1)
plt.imshow(equalizedFoodIm, cmap='gray')
plt.title("Histogram Equalized Image - Problem 3")

plt.subplot(1, 3, 2)
plt.imshow(equalizedFoodImCV2, cmap='gray')
plt.title("Histogram Equalized Image (built-in) - Problem 4")

plt.subplot(1, 3, 3)
plt.imshow(BBHEFoodIm, cmap='gray')
plt.title("BBHE Image - Problem 5")

# Plot the transform functions side by side
plt.figure(figsize=(18, 6)) # Figure 4
plt.subplot(1, 3, 1)
plt.plot(transFuncEqualization)
plt.title("Transform Function - Problem 3")

plt.subplot(1, 3, 3)
plt.plot(transFuncBBHE)
plt.title("Transform Function - Problem 5")

def PSNR(original, processed):
    # obtain the Mean-Square Error using numpy techniques rather than manually summing and dividing
    mse = np.mean((original - processed) ** 2)

    # ensures there isn't a divsion of zero   
    if mse == 0:
        return float('inf')
    maximumPossibleIntensity = 255
    psnr = 10 * np.log10((maximumPossibleIntensity ** 2) / mse)
    return psnr

psnrEqualized = PSNR(foodIm, equalizedFoodIm)
psnrBBHE = PSNR(foodIm, BBHEFoodIm)

print(f"PSNR for equalizedFoodIm - Problem 3: {psnrEqualized:.2f} dB")
print(f"PSNR for BBHEFoodIm - Problem 5: {psnrBBHE:.2f} dB")

plt.show()