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
        np.clip(self.img, 0, clip, out=self.img)
        return self.img
    
    def execute(self):
        
        """Replace the dead pixel value with corrected pixel value and returns 
        the corrected image."""
        
        img_padded = self.padding()
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1])).astype("uint16")
        dpc_img   = np.empty((self.img.shape[0], self.img.shape[1]), np.uint16)     # size of the original image without padding
        
        for y in range(img_padded.shape[0] - 4):        # looping over padded image
            for x in range(img_padded.shape[1] - 4):
                p0 = img_padded[y + 2, x + 2]
                p1 = img_padded[y, x]
                p2 = img_padded[y, x + 2]
                p3 = img_padded[y, x + 4]
                p4 = img_padded[y + 2, x]
                p5 = img_padded[y + 2, x + 4]
                p6 = img_padded[y + 4, x]
                p7 = img_padded[y + 4, x + 2]
                p8 = img_padded[y + 4, x + 4]
                
                """p0 is good if pixel value is between min and max of a 3x3 neighborhhood."""
                
                if not(min([p1, p2, p3, p4, p5, p6, p7, p8]) < p0 < max([p1, p2, p3, p4, p5, p6, p7, p8])):     
                    if (abs(int(p1) - int(p0)) > self.threshold) and (abs(int(p2) - int(p0)) > self.threshold) and (abs(int(p3) - int(p0)) > self.threshold) \
                            and (abs(int(p4) - int(p0)) > self.threshold) and (abs(int(p5) - int(p0)) > self.threshold) and (abs(int(p6) - int(p0)) > self.threshold) \
                            and (abs(int(p7) - int(p0)) > self.threshold) and (abs(int(p8) - int(p0)) > self.threshold):
                        if self.mode == 'mean':
                            p0 = (p2 + p4 + p5 + p7) / 4
                        elif self.mode == 'gradient':
                            
                            dv = abs(2 * int(p0) - int(p2) - int(p7))
                            dh = abs(2 * int(p0) - int(p4) - int(p5))
                            ddl = abs(2 * int(p0) - int(p1) - int(p8))
                            ddr = abs(2 * int(p0) - int(p3) - int(p6))
                            
                            if (min(dv, dh, ddl, ddr) == dv):
                                p0 = (p2 + p7) / 2
                            elif (min(dv, dh, ddl, ddr) == dh):
                                p0 = (p4 + p5) / 2
                            elif (min(dv, dh, ddl, ddr) == ddl):
                                p0 = (p1 + p8) / 2
                            else:
                                p0 = (p3 + p6) / 2
                
                dpc_img[y, x] = p0
                if self.img[y, x]!=p0:
                    self.mask[y, x] = p0
        
        self.img = dpc_img
        return self.clipping(4095)      # not needed as all the corrected values are within 12 bit scale.       

#########################################################################################################