from operator import index
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import colour_demosaicing as cd

def introduce_defect(img, total_defective_pixels, padding):
    
    """This function randomly replaces pixels values with extremely high or low pixel values to create dead pixels.
    Note that the defect is never introduced on the periphery image to ensure that there are no adjacent DPs. 
    The function returns defective image: image containing specified (by TOTAL_DEFECTIVE_PIXELS) 
    number of dead pixels, orig_val: containing original values of the dead pixels introduced."""
    
    if padding:
        padded_img = np.pad(img, ((2,2), (2,2)), "symmetric")
    else:
        padded_img  = img.copy()
   
    orig_val   = np.zeros((padded_img.shape[0], padded_img.shape[1])) 
    
    while total_defective_pixels:
        defect     = [random.randrange(1,15), random.randrange(4081, 4095)]   # stuck low int b/w 1 and 15, stuck high float b/w 4081 and 4095
        defect_val = defect[random.randint(0,1)] 
        random_row, random_col   = random.randint(2, img.shape[0]-3), random.randint(2, img.shape[1]-3)
        left, right  = orig_val[random_row, random_col-2], orig_val[random_row, random_col+2]
        top, bottom  = orig_val[random_row-2, random_col], orig_val[random_row+2, random_col]
        neighbours   = [left, right, top, bottom]
        
        if not any(neighbours) and orig_val[random_row, random_col]==0: # if all neighbouring values in orig_val are 0 and the pixel itself is not defective
            orig_val[random_row, random_col]   = padded_img[random_row, random_col]
            padded_img[random_row, random_col] = defect_val
            total_defective_pixels-=1
    
    return padded_img, orig_val 

#########################################################################################################

def Evaluation(true_img, pred_img, true_mask, pred_mask):
    
    """This function computes and returns the confusion matrix and MSE of a single DPC corrected image."""

    assert np.size(true_mask)==np.size(pred_mask), "Sizes of the input images must match."
    
    # Compute error
    error = np.round((np.add(true_img.astype("float32"), -pred_img.astype("float32"))**2).sum()/true_mask.size, 4)
    pred_mask[pred_mask>0] = 1
    true_mask[true_mask>0] = 1

    conf_mat     = confusion_matrix(true_mask.ravel(), pred_mask.ravel())
    # pos_pred_val = round(conf_mat[1,1]/(conf_mat[1,1]+conf_mat[0,1]), 4)    # TP/(TP+FP)
    
    print("---------------------")
    print("Error: ", error)
    print("---------------------")
    print("True positives: ",  conf_mat[1,1])
    print("False positives: ", conf_mat[0,1])
    print("True negatives: ",  conf_mat[0,0])
    print("False negatives: ", conf_mat[1,0])
    print("---------------------")
    conf_mat = conf_mat.ravel().tolist()
    conf_mat.append(error)                   #extend([error, pos_pred_val])
    return conf_mat

#######################################################################################################
# Object to compile results in a csv file
class Results:
    def __init__(self):
        self.confusion_pd = pd.DataFrame(np.zeros((1,6)), columns=["Filename", "TN", "FP","FN", "TP", "MSE"])
    
    def add_row(self,row):
        self.confusion_pd = pd.concat([self.confusion_pd, pd.DataFrame(np.array(row).reshape(1,6), columns=["Filename", "TN", "FP","FN", "TP", "MSE"])], ignore_index=False)
    
    def save_csv(self, path, filename):
        self.confusion_pd.to_csv(path + "/" +filename + ".csv", index=False)

#########################################################################################################
# Functions to display RGB image.

def demosaic_raw(img, bayer):

        bpp = 12
        img = np.float32(img) / np.power(2, bpp)
        hs_raw = np.uint8(img*255)
        img = np.uint8(cd.demosaicing_CFA_Bayer_bilinear(hs_raw, bayer))
        return img

def get_color(x, y):
    is_even = lambda x: x%2==0

    if is_even(x) and is_even(y):
        return "R"
    elif not(is_even(x)) and not(is_even(y)):
        return "B"
    else:
        return "G"


def white_balance(img, R_gain, B_gain, G_gain):
    img = img.astype("float32")
    for x in range(img.shape[1]): #col
        for y in range(img.shape[0]):   #row
            color = get_color(y, x)
            if color=="R":
                img[y][x] *= R_gain 
            elif color=="G":
                img[y][x] *= G_gain 
            else:
                img[y][x] *= B_gain
    # img = np.clip(img, 0, 4095)               
    img = ((img/4095)*(4095)).astype("uint16")
    return img 

def gamma(img):
    img = np.float32(img)/255
    img = (img**(1/2.2))*255
    return img.astype("uint8")    

####################################################################################
# Object to record FPs in a csv file. 
class FPs:
    def __init__(self):
        self.FPs_pd = pd.DataFrame(np.zeros((1,9)), columns=["x-coord_p1", "y-coord_p1","loc","loc","loc", "val","val","val", "corrected_value"])
    
    def add_row(self,matrix):
        self.FPs_pd = pd.concat([self.FPs_pd, pd.DataFrame(np.array(matrix), columns=["x-coord_p1", "y-coord_p1","loc","loc","loc", "val","val","val", "corrected_value"])], ignore_index=False)
    
    def save_csv(self, path, filename):
        self.FPs_pd.to_csv(path + "/" +filename + ".csv", index=False)

####################################################################################
from pathlib import Path

def save_FPs_as_csv(orig_img_arr, gt_arr, mask_arr, save_path, total_fp):
    
    """This function identifies a FP and saves its 5x5 neighborhood in a csv file along
    with the coordinates of the top left pixel (p1).The total number of FPs saved can be 
    specified using TOTAL_FP"""

    save_pd = FPs()
    for i in range(total_fp):#(2, gt_arr.shape[0]-2):
        for j in range(total_fp):#(2, gt_arr.shape[1]-2):

            FP_flag = True if gt_arr[i+2,j+2]==0 and mask_arr[i+2,j+2]>0 else False
            if FP_flag:
                p0 = orig_img_arr[i + 2, j + 2] # center pixel
                p1 = orig_img_arr[i, j]         # top left
                p2 = orig_img_arr[i, j + 2]     # top mid
                p3 = orig_img_arr[i, j + 4]     # top right
                p4 = orig_img_arr[i + 2, j]     # mid row left
                p5 = orig_img_arr[i + 2, j + 4] # mid row right
                p6 = orig_img_arr[i + 4, j]     # bottom row left
                p7 = orig_img_arr[i + 4, j + 2] # bottom row mid
                p8 = orig_img_arr[i + 4, j + 4] # bottom right

                # Pixel values are saved after Black level correction
                neighbors = np.array([[i, j,"p1", "p2", "p3", p1-200, p2-200, p3-200, mask_arr[i+2,j+2]], 
                                     ["","","p4", "p0", "p5", p4-200, p0-200, p5-200, "" ],
                                     ["","","p6", "p7", "p8", p6-200 ,p7-200, p8-200, ""], 
                                     ["","", "",  "",  "",  "",  "", "", ""]])
                save_pd.add_row(neighbors)
            
    save_pd.save_csv(str(Path(save_path).parent),Path(save_path).stem) 
####################################################################################    

def crop_to_1920x1080(img):
    
    """Chops off the sides of the input image using array slicing and returns 
    the img of size 1920x1080."""

    crop_cols = (img.shape[1]-1920)  
    crop_rows = (img.shape[0]-1080)
    print(img.shape[1], crop_cols//2)
    assert crop_cols!=0 or crop_rows!=0, "Image size must be greater than 1080x1920." 
    return img[crop_rows//2:-crop_rows//2,crop_cols//2:-crop_cols//2]

