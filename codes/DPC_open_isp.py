import numpy as np

#########################################################################################################

# Define Class DPC

class DPC:
    def __init__(self, img, size, threshold):
        self.img = img.copy()
        self.height = size[0]
        self.width = size[1]
        self.threshold = threshold
        self.mode      = "gradient" 
    
    def padding(self):
        img_pad = np.pad(self.img, (2, 2), 'reflect')
        return img_pad
    
    def clipping(self, clip):
        if np.amax(self.img.copy().ravel())>4095 or np.amin(self.img.copy().ravel())<0:
            print("clipping after DPC")
        self.img = np.uint16(np.clip(self.img, 0, clip, out=self.img))
        return self.img
    
    def execute(self):
        
        """Replace the dead pixel value with corrected pixel value and returns 
        the corrected image."""
        # Mirror padding is applied to self.img.
        img_padded = np.float32(self.padding())
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1])).astype("uint16")
        dpc_img   = np.empty((self.img.shape[0], self.img.shape[1]), np.float32)     # size of the original image without padding
        
        for y in range(img_padded.shape[0] - 4):        # looping over padded image
            for x in range(img_padded.shape[1] - 4):
                top_left  = img_padded[y, x]
                top_mid   = img_padded[y, x + 2]
                top_right = img_padded[y, x + 4]
                
                left_of_center_pixel  = img_padded[y + 2, x]
                center_pixel          = img_padded[y + 2, x + 2]    # pixel under test
                right_of_center_pixel = img_padded[y + 2, x + 4]
                
                bottom_right = img_padded[y + 4, x]
                bottom_mid   = img_padded[y + 4, x + 2]
                bottom_left  = img_padded[y + 4, x + 4]
                
                neighbors    = np.array([top_left, top_mid, top_right, left_of_center_pixel, right_of_center_pixel,
                                bottom_right, bottom_mid, bottom_left])

                # center_pixel is good if pixel value is between min and max of a 3x3 neighborhhood.
                if not(min(neighbors) < center_pixel < max(neighbors)):
                    
                    # ""center_pixel is corrected only if the difference of center_pixel and every 
                    # neighboring pixel is greater than the speciified threshold.
                    # The two if conditions are used in combination to reduce False positives.""
                    
                    diff_with_center_pixel = abs(neighbors-center_pixel)
                    thresh                 = np.full_like(diff_with_center_pixel, self.threshold)
                    
                    if np.all(diff_with_center_pixel > thresh):
                        # Compute gradients
                        vertical_grad       = abs(2 * center_pixel - top_mid - bottom_mid)
                        horizontal_grad     = abs(2 * center_pixel - left_of_center_pixel - right_of_center_pixel)
                        left_diagonal_grad  = abs(2 * center_pixel - top_left - bottom_left)
                        right_diagonal_grad = abs(2 * center_pixel - top_right - bottom_right)
                        
                        min_grad = min(vertical_grad, horizontal_grad, left_diagonal_grad, right_diagonal_grad)
                        
                        # Correct value is computed using neighbors in the direction of minimum gradient. 
                        if ( min_grad == vertical_grad):
                            center_pixel = (top_mid + bottom_mid) / 2
                        elif (min_grad == horizontal_grad):
                            center_pixel = (left_of_center_pixel + right_of_center_pixel) / 2
                        elif (min_grad == left_diagonal_grad):
                            center_pixel = (top_left + bottom_left) / 2
                        else:
                            center_pixel = (top_right + bottom_right) / 2
                
                """Corrected pixels are placed in non-padded image."""
                dpc_img[y, x] = center_pixel
                if self.img[y, x]!=center_pixel:
                    self.mask[y, x] = center_pixel
        
        self.img = dpc_img
        return self.clipping(4095)      # not needed as all the corrected values are within 12 bit scale.       

#########################################################################################################