import time
import os
import shutil
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

        
def save_validation_images(ldct_img, sdct_img, output, save_path, model_name, i, ldct_sdct_metrics, output_sdct_metrics):
    ldct_img = ldct_img.detach().cpu()
    sdct_img = sdct_img.detach().cpu()
    output = output.detach().cpu()
    
    # create figure with subplots
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    
    ax[0].imshow(ldct_img[0,0,:,:], cmap=plt.cm.gray)
    ax[0].set_title('Low-dose (50)', fontsize=24)
    ax[0].set_xlabel("MSE: {:.4f}\nRMSE: {:.4f}\nPSNR: {:.4f}\nSSIM: {:.4f}".format(ldct_sdct_metrics['mse'],
                                                                                    ldct_sdct_metrics['rmse'],
                                                                                    ldct_sdct_metrics['psnr'],
                                                                                    ldct_sdct_metrics['ssim']), fontsize=20)

    ax[1].imshow(sdct_img[0,0,:,:], cmap=plt.cm.gray)
    ax[1].set_title('Standard-dose (100)', fontsize=24)
    

    ax[2].imshow(output[0,0,:,:], cmap=plt.cm.gray)
    ax[2].set_title(f'{model_name} output', fontsize=24)
    ax[2].set_xlabel("MSE: {:.4f}\nRMSE: {:.4f}\nPSNR: {:.4f}\nSSIM: {:.4f}".format(output_sdct_metrics['mse'],
                                                                                     output_sdct_metrics['rmse'],
                                                                                     output_sdct_metrics['psnr'],
                                                                                     output_sdct_metrics['ssim']), fontsize=20)

    fig.savefig(os.path.join(save_path, f'{model_name}_{i}.png'))
    plt.close()
        

                
def create_csv(data_path, csv_path):
    """Creates a csv file with a list of images

    Columns: subject, phase, resolution, algorithm, dose, path
    """
    
    # list of rows
    list_ = []

    for i, path in enumerate(glob.glob(os.path.join(data_path, '*', '*', '*.jpg'))):
        # decompose names
        ID = os.path.basename(path)
        experiment = os.path.basename(os.path.dirname(path))
        subject = os.path.basename(os.path.dirname(os.path.dirname(path)))
        
        # decompose properties
        matches = re.match('(.*)_(.*)_(.*)_(.*)', experiment)
        phase, resolution, algorithm, dose = matches.groups()
        # append to list
        _ = [subject, phase, resolution, algorithm, dose, path]
        list_.append(_)
        
    data = pd.DataFrame(list_, columns=['subject', 'phase', 'resolution', 'algorithm', 'dose', 'path'])
    data.to_csv(csv_path, index=False)