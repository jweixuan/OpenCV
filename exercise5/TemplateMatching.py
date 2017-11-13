import numpy as np
import cv2

def TemplateMatching(src, temp, stepsize): # src: source image, temp: template image, stepsize: the step size for sliding the template
    mean_t = 0;
    var_t = 0;
    location = [0, 0];
    # Calculate the mean and variance of template pixel values
    # ------------------ Put your code below ------------------ 
    mean_t,var_t = cv2.meanStdDev(temp)
    _sum = 0;                
    max_corr = 0;
    # Slide window in source image and find the maximum correlation
    for i in np.arange(0, src.shape[0] - temp.shape[0], stepsize):
        for j in np.arange(0, src.shape[1] - temp.shape[1], stepsize):
            var_s = 0
            corr = 0
            mean_s = 0
            # Calculate the mean and variance of source image pixel values inside window
            # ------------------ Put your code below ------------------ 
            mean_s = np.mean(src[i:i+temp.shape[0],j:j+temp.shape[i]])
            var_s = np.var(src[i:i+temp.shape[0],j:j+temp.shape[i]])

            # Calculate normalized correlation coefficient (NCC) between source and template
            # ------------------ Put your code below ------------------ 
	    	#_sum += (src[i,j] - mean_s)*(temp[i,j]-mean_t)            
	    	#corr = (1/(i+1)*(j+1))*_sum / ((var_s**2) * (var_t**2))
            a = (src[i:i+temp.shape[0],j:j+temp.shape[1]]-means)*(temp-temp_mean)
            _sum = sum(sum(prod[i]) for i in range(len(prod)))
            corr = (1/float((temp.shape[0])*(temp.shape[1]))) * total / ((temp_variances)*(variances))
            if corr > max_corr:
                max_corr = corr;
                location = [i, j];
    
    return location

# load source and template images
source_img = cv2.imread('E:/EC601/source_img.jpg') # read image in grayscale
temp = cv2.imread('E:/EC601/template.jpg') # read image in grayscale
location = TemplateMatching(source_img, temp, 20);
print(location)
match_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

# Draw a red rectangle on match_img to show the template matching result
# ------------------ Put your code below ------------------ 
location2 = [location(0)+temp.shape(0),location(0)+temp.shape(1)]
match_img = cv2.Rectangle(match_img,location,location2,(255,0,0),3)
cv2.rectangle(match_img,(location[1],location[0]),(location[1]+temp.shape[1],location[0]+temp.shape[0]),(0,0,255),5) 

# Save the template matching result image (match_img)
# ------------------ Put your code below ------------------ 
cv2.imwrite('match_img.jpg',match_img)

# Display the template image and the matching result
cv2.namedWindow('TemplateImage', cv2.WINDOW_NORMAL)
cv2.namedWindow('MyTemplateMatching', cv2.WINDOW_NORMAL)
cv2.imshow('TemplateImage', temp)
cv2.imshow('MyTemplateMatching', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()