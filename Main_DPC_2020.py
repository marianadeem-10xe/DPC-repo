import os
import codes.DPC_open_isp as openisp
from codes.utils import demosaic_raw, white_balance, Evaluation, gamma, Results
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# Define variables

raw_path        = "./in_frames/Defective_100_ISO100_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407210640_BNR_OFF.raw"
org_img_path    = "./in_frames/Undefected_ISO100_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407210640_BNR_OFF.raw"
GT_path         = "./in_frames/GT_ISO100_HisiRAW_2592x1536_12bits_RGGB_Linear_20220407210640_BNR_OFF.raw"
size            = (1536, 2592)                       #(height, width)

raw_filename    = Path(raw_path).stem.split(".")[0]
out_img_path    = "./out_frames/OUT_" + raw_filename +".png"
out_mask_path   = "./out_frames/OUT_mask_" + raw_filename +".raw"
result          = Results()

# Flags
run_DPC    = True
evaluate   = True

# Read the raw image
print("Reading raw file...")
raw_file = np.fromfile(raw_path, dtype="uint16").reshape(size)      # Construct an array from data in a text or binary file.

# Read the defective image
if run_DPC:
    def_img = np.clip(np.float32(raw_file)-200, 0, 4095).astype("uint16")    # BLC
    
    # convert to 3 channel image before saving
    print("Saving input image...")
    save_img = gamma(demosaic_raw(white_balance(def_img.copy(), 320/256, 740/256, 256/256), "RGGB")) 
    plt.imsave("./in_frames/No_DPC_" + raw_filename + ".png", save_img)

    print("Running DPC...")    
    dpc        = openisp.DPC(def_img, size, 80) 
    corr_img   = dpc.execute() 
    corr_mask  = dpc.mask
    
    # Save the corrected 3 channel image after white balancing
    print("Saving DP corrected image...")
    save_corr_img = gamma(demosaic_raw(white_balance(corr_img.copy(), 320/256, 740/256, 256/256), "RGGB"))
    plt.imsave(out_img_path, save_corr_img)
    with open(out_mask_path, "wb") as file:
        corr_mask.astype("uint16").tofile(file)
    

# Evaluate DPC
if evaluate:
    print("Evaluating results...")
    img_GT  = np.fromfile(org_img_path, dtype="uint16").reshape(size)
    mask_GT = np.fromfile(GT_path, dtype="uint16").reshape(size)   
    
    confusion_matrix  = Evaluation(img_GT, corr_img, mask_GT, corr_mask)
    confusion_matrix.insert(0, raw_filename)
    result.add_row(confusion_matrix)

result.save_csv("./results/", "openISP_th80_results")
print("Done!")
#############################################################################