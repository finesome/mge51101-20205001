import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import create_img_pairs, prepare_data, split_img_pairs, ImgPairsDataset
from metrics import compute_metrics
from utils import save_validation_images
from AE.models import Encoder, Decoder, FeatureExtractor
    

if __name__ == "__main__":
    # initialize a command line parser
    parser = argparse.ArgumentParser()
    # define arguments
    parser.add_argument('--csv_path', type=str, default='./all_imgs.csv')
    parser.add_argument('--data_path', type=str, default='/home/DATA_Lia/data_04/DATASET_ALI/CT_enhancement/data/')
    parser.add_argument('--phases', type=str, default='art,dly,por')
    parser.add_argument('--ldct_algo', type=str, default='FBP')
    parser.add_argument('--sdct_algo', type=str, default='IR')
    parser.add_argument('--model', type=str, default='AE')
    parser.add_argument('--crop_size', type=int, default=(400, 400))
    parser.add_argument('--patch_size', type=int, default=(400, 400))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--initial_lr', type=float, default=1e-4)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    
    # parse all the specified arguments
    args = parser.parse_args()
    
    # use the built-in cuDNN auto-tuner
    cudnn.benchmark = True
    
    # set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # prepare train, validation and test splits of CT image pairs
    train_data, val_data, test_data = prepare_data(args.csv_path, args.data_path, args.phases, args.ldct_algo, args.sdct_algo)
    
    # datasets and dataloaders
    train_dataset = ImgPairsDataset(train_data, crop_size=args.crop_size, patch_size=args.patch_size)
    val_dataset = ImgPairsDataset(val_data, crop_size=args.crop_size, patch_size=args.patch_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)
    
    # initialize models
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    feature_extractor = FeatureExtractor().to(device)
    
    # set VGG to evaluation mode
    feature_extractor.eval()
        
    # training hyperparameters
    lr = args.initial_lr
    betas = (args.b1, args.b2)
    
    # loss function and optimizer
    # we compute reconstruction after decoder so use MSE
    # in order to use multi parameters with one optimizer, concat parameters after changing into list
    params = list(encoder.parameters()) + list(decoder.parameters())
    mse_loss_fn = nn.MSELoss()
    vgg_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(params, lr=lr, betas=betas)
    
    # define path for saving the results
    save_path = f'/home/DATA_Lia/data_04/DATASET_ALI/CT_enhancement/experiments/{args.model}'
    checkpoints_save_path = os.path.join(save_path, 'checkpoints')
    results_save_path = os.path.join(save_path, 'results')
    
    # create folders
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    if not os.path.exists(checkpoints_save_path):
        os.makedirs(checkpoints_save_path)
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
        
    # save training losses
    training_losses = {}
    # save validation losses and metrics
    val_losses = {}
    val_metrics = {}
    
    for epoch in range(args.num_epochs):
        print('--------------------------------------------------')
        print(f'Training; epoch: {epoch}')
        print('--------------------------------------------------')
        
        # set models to training mode
        encoder.train()
        decoder.train()
        
        # record running losses
        running_mse_loss, running_vgg_loss, running_loss = 0.0, 0.0, 0.0
        
        # training
        for i, (ldct_img, sdct_img) in enumerate(tqdm(train_dataloader)):
            ldct_img = ldct_img.to(device)
            sdct_img = sdct_img.to(device)

            # zero gradients
            optimizer.zero_grad()
            # forward pass
            output = encoder(ldct_img)
            output = decoder(output)
            # compute losses
            mse_loss = mse_loss_fn(output, sdct_img)
            with torch.no_grad():
                # vgg_loss = vgg_loss_fn(output, sdct_img)
                
                output_features = feature_extractor(output.repeat(1, 3, 1, 1))
                sdct_features = feature_extractor(sdct_img.repeat(1, 3, 1, 1))
                vgg_loss = vgg_loss_fn(output_features, sdct_features)
                
            # combine loss
            loss = mse_loss + vgg_loss
            # backpropagation
            loss.backward()
            # update weights
            optimizer.step()
            
            # record losses
            running_mse_loss += mse_loss.item()
            running_vgg_loss += vgg_loss.item()
            running_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch: {epoch}; batch: {i}; MSE loss: {running_mse_loss / (i + 1)}, VGG loss: {running_vgg_loss / (i + 1)}, loss: {running_loss / (i + 1)}')
                
        # store training losses
        training_losses[epoch] = {'mse_loss': running_mse_loss / len(train_dataloader), 'vgg_loss': running_vgg_loss / len(train_dataloader), 'loss': running_loss / len(train_dataloader)}

        print('--------------------------------------------------')
        print(f'Validation; epoch: {epoch}')
        print('--------------------------------------------------')
        
        # validation
        with torch.no_grad():
            # set models to evaluation mode
            encoder.eval()
            decoder.eval()
            
            # record running losses
            running_mse_loss, running_vgg_loss, running_loss = 0.0, 0.0, 0.0
            
            # LDCT vs SDCT metrics
            ldct_sdct_mse, ldct_sdct_rmse, ldct_sdct_psnr, ldct_sdct_ssim = 0.0, 0.0, 0.0, 0.0
            # Output vs SDCT metrics
            output_sdct_mse, output_sdct_rmse, output_sdct_psnr, output_sdct_ssim = 0.0, 0.0, 0.0, 0.0
            
            epoch_results_save_path = os.path.join(results_save_path, str(epoch))
            if not os.path.exists(epoch_results_save_path):
                os.makedirs(epoch_results_save_path)
            
            for i, (ldct_img, sdct_img) in enumerate(tqdm(val_dataloader)):
                ldct_img = ldct_img.to(device)
                sdct_img = sdct_img.to(device)

                # forward pass
                output = encoder(ldct_img)
                output = decoder(output)
                
                # compute losses
                mse_loss = mse_loss_fn(output, sdct_img)
                # vgg_loss = vgg_loss_fn(output, sdct_img)
                
                output_features = feature_extractor(output.repeat(1, 3, 1, 1))
                sdct_features = feature_extractor(sdct_img.repeat(1, 3, 1, 1))
                vgg_loss = vgg_loss_fn(output_features, sdct_features)
                
                loss = mse_loss + vgg_loss
                
                running_mse_loss += mse_loss.item()
                running_vgg_loss += vgg_loss.item()
                running_loss += loss.item()
                
                # compute metrics
                ldct_sdct_metrics = compute_metrics(ldct_img, sdct_img)
                output_sdct_metrics = compute_metrics(output, sdct_img)

                ldct_sdct_mse += ldct_sdct_metrics['mse']
                ldct_sdct_rmse += ldct_sdct_metrics['rmse']
                ldct_sdct_psnr += ldct_sdct_metrics['psnr']
                ldct_sdct_ssim += ldct_sdct_metrics['ssim']

                output_sdct_mse += output_sdct_metrics['mse']
                output_sdct_rmse += output_sdct_metrics['rmse']
                output_sdct_psnr += output_sdct_metrics['psnr']
                output_sdct_ssim += output_sdct_metrics['ssim']
                
                # save images
                save_validation_images(ldct_img, sdct_img, output, epoch_results_save_path, args.model, i, ldct_sdct_metrics, output_sdct_metrics)
                
                
            # save validation losses
            val_losses[epoch] = {
                'mse_loss': running_mse_loss / len(val_dataloader), 
                'vgg_loss': running_vgg_loss / len(val_dataloader), 
                'loss': running_loss / len(val_dataloader)
            }
            
            # save validation metrics
            val_metrics[epoch] = {
                'ldct_sdct': {
                    'mse': ldct_sdct_mse / len(val_dataloader),
                    'rmse': ldct_sdct_rmse / len(val_dataloader),
                    'psnr': ldct_sdct_psnr / len(val_dataloader),
                    'ssim': ldct_sdct_ssim / len(val_dataloader)
                },
                'output_sdct': {
                    'mse': output_sdct_mse / len(val_dataloader),
                    'rmse': output_sdct_rmse / len(val_dataloader),
                    'psnr': output_sdct_psnr / len(val_dataloader),
                    'ssim': output_sdct_ssim / len(val_dataloader)
                }
            }
            
            # print results
            print('LDCT vs SDCT:\n\tMSE: {:.4f}, \n\tRMSE: {:.4f} \n\tPSNR: {:.4f} \n\tSSIM: {:.4f}'
                  .format(ldct_sdct_mse / len(val_dataloader), 
                          ldct_sdct_rmse / len(val_dataloader), 
                          ldct_sdct_psnr / len(val_dataloader), 
                          ldct_sdct_ssim / len(val_dataloader)))
            print('Autoencoder output vs SDCT:\n\tMSE: {:.4f}, \n\tRMSE: {:.4f} \n\tPSNR: {:.4f} \n\tSSIM: {:.4f}'
                  .format(output_sdct_mse / len(val_dataloader), 
                          output_sdct_rmse / len(val_dataloader), 
                          output_sdct_psnr / len(val_dataloader), 
                          output_sdct_ssim / len(val_dataloader)))
            
        # save model checkpoint
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': training_losses[epoch],
            'val_losses': val_losses[epoch],
            'val_metrics': val_metrics[epoch]
        }, os.path.join(checkpoints_save_path, str(epoch)))
        
    # save all results
    torch.save({
        'training_losses': training_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics
    }, os.path.join(save_path, f'{args.model}_results'))
        
    
    
    
    
    
    