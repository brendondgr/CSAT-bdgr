import os
import gc
import math
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
import concurrent.futures
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import reduce
import wandb

from torchvision.ops import sigmoid_focal_loss
from utils.util import (get_pickles, ids, fold_operation, split, load_one_pickle)
from eval.eval import Meter

from model.encoder import Encoder

wb = False
seed_val = 3500
torch.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)

TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]' #'{l_bar}{r_bar}' #'{l_bar}{r_bar}' # tqdm bar format
SAVE_PATH = 'runs/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from utils.plots import plot_images

from utils.plots import plot_images
import json

def setup(rank, world_size):
    if not dist.is_initialized():
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:28457",
            rank=rank,
            world_size=world_size
        )
        # Set the GPU to use
        torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def printer(vals, names):
    print('\n')
    for val, name in zip(vals, names):
        print(f'{name}: {val.shape}')


class SiameseDataset(Dataset):
    def __init__(self, rank, data, phase, transform=None, percent=1.0, num_workers=8, task='train'):
        self.phase = phase
        self.transform = transform
        self.data = data
        self.task = task
        


    def __getitem__(self, index):
        file = self.data[index]

        im_a = load_one_pickle(file[0])['img']
        im_b = load_one_pickle(file[1])['img']

        target = torch.ones((1))*file[2]

        assert im_a is not None 
        assert im_b is not None
        assert target is not None

        if self.transform:
            im_a = self.transform(im_a)
            im_b = self.transform(im_b)            

        if self.task =='infer':
            return im_a, im_b, target.squeeze(-1), file[0], file[1]

        else:
            return im_a, im_b, target.squeeze(-1), file[0], file[1]



    def __len__(self):
        return len(self.data)

class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super(SiameseNetwork, self).__init__()
        self.encoder = encoder
        self.act = nn.Softsign() #nn.Tanh()  
        
    def forward(self, x1, x2):
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)

        return embedding1, embedding2

def get_distance(a,b, eps=1e-5):
    # # Print out a and b before any calculations are done to either.
    # print(f'a. Shape: {a.shape}, {a}')
    # print(f'b. Shape: {b.shape}, {b}')
    
    # Get the cosine distance and normalize it to [0,1]
    distance = nn.CosineSimilarity(dim=1, eps=1e-5)(a, b)
    # print(f'1. Distance: {distance.shape}, {distance}')
    
    # Normalize to [0,1]
    mi, ma = -1, 1
    
    # Normalize
    distance = (distance-mi)/(ma-mi)
    # print(f'2. Distance: {distance.shape}, {distance}')
    
    # eps = 1e-5
    return_val = 1-distance+eps
    # print(f'3. Distance: {return_val.shape}, {return_val}')
    
    # Return
    return return_val


def contrastive_focal_loss(rank, emb1, emb2, target, gamma=3, eps=1e-5, alpha=0.7, phase='train', nc=2):  
    # Get logits
    logit = get_distance(emb1, emb2)
    logit = torch.clamp(logit, min=eps, max=1-eps)  
    
    # Prepare Logits for Softmax
    x = torch.zeros(target.shape[0], nc).to(rank)
    x[...,0] = logit 
    x[...,1] = 1 - logit
    
    # Print out the logit and x
    # print(f'Logit Shape: {logit.shape}, Logit: {logit}')
    # print(f'X Shape: {x.shape}, X: {x}')
    
    # If the X Tensor contains Any NaN Values

    # Class Weights
    alpha_t = torch.tensor([alpha, 1-alpha]).to(rank)
    
    # Softmax/NLL
    nll_loss = nn.NLLLoss(weight=alpha_t, reduction='none')
    log_p = F.log_softmax(x, dim=-1)
    ce = nll_loss(log_p.float(), target.long())
    
    # Focal Loss
    all_rows = torch.arange(len(x))
    log_pt = log_p[all_rows, target.long()]
    pt = log_pt.exp()
    focal_term = (1-pt)**gamma
    
    # Final Loss
    loss = focal_term*ce
    # loss_list = loss
    loss = torch.sum(loss)
    # loss_nonan = torch.sum(loss[~torch.isnan(loss)])
    
    # print(f'The Loss is {loss}, without nan-values {loss_nonan}. Based on {loss_list}')
    
    d_name = 'distance_' + phase

    if wb:
        wandb.log({d_name: torch.mean(1-logit)})
    return loss


def save_model(root, siamese_net, epoch, optimizer, acc, best_accuracy, fold, opt):
    if acc>=best_accuracy:
        best_accuracy = acc
        name = f'bb_best_pretrainer_{opt.cf}.pth'
    else:
        name = f'bb_last_pretrainer_{opt.cf}.pth'
    
    save_path = root + SAVE_PATH + str(fold) + name
    
    # If best_accuracy is of type float. Otherwise, get item() from the tensor.
    if isinstance(best_accuracy, torch.Tensor):
        best_accuracy = best_accuracy.item()    
    
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_accuracy,
        }
    
    print(f'Here is ')
    
    torch.save(checkpoint, save_path)
    return best_accuracy


def bdg_analysis(file_a, file_b):
    # Combine the lists file_a and file_b into a single list.
    combined_list = file_a + file_b
    
    # Load each file... it is in dictionary format:
    # {'img':image, 'box':bounding_boxes, 'label':SCR_label, 'name':filename}
    loaded_files = [load_one_pickle(file) for file in combined_list]
    
    # For each 'img', load as PIL Image, save it into ../test_images/ with the nme corresponding to the filename in the dictionary.
    for file in loaded_files:
        img = file['img']
        name = file['name']
        
        # img is in Tensor Format, Convert to PIL Image
        image = transforms.ToPILImage()(img)
        
        # Convert "img" which is tensor to np array.
        img = np.array(img)
        
        image.save(f'../test_images/{name}.png', format='PNG')
        
        # Analyze each image, check to see if there are any nan values that are incorrect.
        if np.isnan(img).any() or np.isinf(img).any():
            # Save a text file saying saying that the image contains NaN or Inf values.
            with open(f'../test_images/{name}_nan_inf.txt', 'w') as f:
                f.write(f'Image {name} contains NaN or Inf values!')
        
        # Do the same for the bounding box...
        if np.isnan(file['box']).any() or np.isinf(file['box']).any():
            # Save a text file saying saying that the image contains NaN or Inf values.
            with open(f'../test_images/{name}_nan_inf.txt', 'w') as f:
                f.write(f'Image {name} contains NaN or Inf values!')

def train_epoch(rank, siamese_net, fold, optimizer, train_loader, val_loader, best_accuracy, epoch, epochs, opt, running_loss=0):  
    losses = Meter(1, rank)
    if rank ==0:
        print(('\n' + '%44s'+'%22s' * 3) % ('Fold', 'Epoch', 'GPU Mem','Loss'))

    pbar = tqdm(enumerate(train_loader), bar_format=TQDM_BAR_FORMAT, total=len(train_loader))
    pairwise = nn.PairwiseDistance(p=2)
    # with torch.autograd.detect_anomaly():
    for batch_idx, (x1, x2, targets, file_a, file_b) in pbar:
        x1 = x1.to(rank, non_blocking=True)
        x2 = x2.to(rank, non_blocking=True)
        targets = targets.to(rank, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            embeddings1, embeddings2 = siamese_net(x1, x2)
            
            loss = contrastive_focal_loss(rank, embeddings1, embeddings2, targets, phase='train')

        loss = torch.sum(loss)
        losses.adds(loss)
        loss.backward()
        
        # Ajust if the gradient is too lrage...
        torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), 1.0)

        # Optimize
        optimizer.step()

        avg_ls = losses.returns(losses.means('r'))

        if wb:
            wandb.log({"train_loss": loss, "train_step":(epoch+1)*(batch_idx+1)})

        if rank==0:            
            mem = f'{torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%44s'+'%22s'*2 + '%22.4g') % 
                (f'{fold}', f'{epoch}/{epochs - 1}', mem, avg_ls))
        
        # if batch_idx == 20:
        #     break

    if wb:
        wandb.log({"epoch_loss": avg_ls})

    
    
    if rank==0:
        acc, diction =  validate(rank, siamese_net, val_loader, epoch, opt=opt)
        best_accuracy=save_model(opt.root, siamese_net, epoch, optimizer, acc, best_accuracy, fold, opt)
        if wb:
            wandb.log({"best_accuracy": best_accuracy})
        
        print(f'-- Best accuracy is {best_accuracy}')
    
    return best_accuracy



def validate(rank, siamese_net, val_loader, e, thres=0.1, opt=None):
    torch.cuda.empty_cache()
    gc.collect()
    
    all_distances = []

    total_loss = Meter(1, rank=rank)
    crr = Meter(1, rank=rank)
    pairwise = nn.PairwiseDistance(p=2)
    with torch.no_grad():
        # total_loss = 0 
        # corrects = 0
        tps = 0
        tns = 0
        fps = 0
        fns = 0
        total = 0

        if rank==0:
            print(('\n' + '%44s'+'%22s' * 4) % ('Correct', '(TP,P)', '(TN,N)', 'Accuracy', 'Loss'))
        pbar = tqdm(enumerate(val_loader), bar_format=TQDM_BAR_FORMAT,total=len(val_loader))

        for batch_idx, (x1, x2, targets, file_a, file_b) in pbar:
            x1 = x1.to(rank, non_blocking=True)
            x2 = x2.to(rank, non_blocking=True)
            targets = targets.to(rank, non_blocking=True)

            # Forward pass
            embeddings1, embeddings2 = siamese_net(x1, x2)
            
            dist= nn.CosineSimilarity(dim=1, eps=1e-5)(embeddings1, embeddings2) 
            
            ###############################
            # Save Distance
            all_distances.append({"val_distance": dist.mean().item(), "ground_truth": targets.cpu().numpy().tolist()})
            ###############################
            
            loss = sigmoid_focal_loss(dist, targets, alpha=0.15, gamma=2.0, reduction='sum') #contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets, phase='val')
            
            threshold = torch.ones_like(dist)*thres
            op = torch.relu(torch.sign(threshold-dist))
            
            # op = dist>thres


            total_loss.adds(loss)

            if wb:
                wandb.log({"val_loss": loss, "val_step":(e+1)*(batch_idx+1)})

            avg_loss = total_loss.returns(total_loss.means('r'))


            correct = op.eq(targets)
            incorrect = torch.logical_not(correct)
            tp = correct[op==1].sum().item()
            tn = correct[op==0].sum().item()
            fp = incorrect[op==1].sum().item()
            fn = incorrect[op==0].sum().item()

            p = targets.sum().item()
            n = len(targets) - p

            correct = correct.sum().item()
            tps += tp
            tns += tn
            fps += fp
            fns += fn
            total += targets.size(0)

            crr.adds(correct)


            if rank==0:
                pbar.set_description(('%44s'+'%22s'*2 +'%22.4g' * 2) % (correct, f'({tp},{p})', f'({tn},{n})', correct/(p+n), loss.item()))

            if batch_idx == 5:
                break
            
        print(f'Correct: {crr.returns(crr.sums("r"))}, Total: {total}')
        corrects = crr.returns(crr.sums('r'))
        incorrects = total - corrects
        accuracy = corrects / total
        if wb:
            wandb.log({"Correct": corrects, "Incorrect":incorrects, "Accuracy":accuracy})


    if rank==0:
        print(('\n'+ '%44s') % ('Validation stats:'))
        print(('%44s'+'%22s' * 5) % ('Total', 'TP', 'TN', 'Incorrect', 'avg_acc', 'avg_loss'))
        print(('%44s'+'%22s' * 3 + "%22.4g"*2) % (total, f'{tps}/{corrects}', f'{tns}/{corrects}', incorrects, accuracy, avg_loss))
        print(tps, tns, fps, fns, corrects, incorrects)
    

    # Save the validation distance into the info folder
    if opt.run_validation:
        # Create info directory if it doesn't exist
        info_dir = os.path.join(opt.root, 'info')
        os.makedirs(info_dir, exist_ok=True)

        # With all of the distance stored in a list of dictionaries, dump into a JSON file.
        dist_dict = {"distances": all_distances}
        
        print(f'\n\n{dist_dict}\n\n')
        
        num = opt.resume_weight.split('/')[-1].split('.')[0][-1]

        # Save as JSON file
        with open(os.path.join(info_dir, f'{num}_val_dist.json'), 'w') as f:
            json.dump(dist_dict, f)

    else:
        dist_dict = None
    
    # Clear All Tensors/Memory that Were used in the Validation Loop
    torch.cuda.empty_cache()
    gc.collect()
    
    return torch.Tensor([accuracy]).to(rank), dist_dict



def tx():
    tx_dict = {'train':transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        ]),

        'val': transforms.Compose([
        transforms.ToTensor(),
        ])
        }
    return tx_dict



def get_dataset(world_size, rank, data, phase, transform, batch_size=64, shuffle=False, num_workers=8, task='train'):

    dataset = SiameseDataset(rank, data, phase, task=task)
    
    if world_size>0:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler=None

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True)    
    
    return dataloader, sampler


def get_next_checkpoint_number(directory):
    # Get all items in the directory
    items = os.listdir(directory)
    
    # Filter out directories and get the first character of each item
    numbers = [int(item[0]) for item in items if os.path.isfile(os.path.join(directory, item)) and item[0].isdigit()]
    
    # Determine the highest number
    if numbers:
        highest_number = max(numbers)
        next_number = highest_number + 1
    else:
        next_number = 0
    
    print(f'The next highest number is: {next_number}\n\n')


def pretrainer(rank, world_size, opt, max_num_epochs=-1):
    num_epochs = opt.epochs
    batch_size = opt.batch_size 
    root = opt.root 
    phases = [opt.train_folder, opt.val_folder]
    resume = opt.resume
    resume_weight = opt.resume_weight
    folds = opt.folds
    fold = opt.cf
    validate_instead = opt.run_validation
    lr = 0.00001
    max_num_epochs = max_num_epochs
    
    get_next_checkpoint_number(root + SAVE_PATH)


    setup(rank, world_size)
    
    tx_dict = tx()

    # create model and optimizer
    encoder = Encoder(hidden_dim=768, num_encoder_layers=12, nheads=8, num_patch=12, multiscale=True).to(rank)
    siamese_net = SiameseNetwork(encoder).to(rank)

    # Wrap the model with DistributedDataParallel
    siamese_net = DDP(siamese_net, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accuracy = torch.Tensor([0]).to(rank)
    start_epoch = 0

    if resume:
        ckptfile = resume_weight #+ '.pth'
        ckpts = torch.load(ckptfile, map_location='cpu')
        siamese_net.load_state_dict(ckpts['model_state_dict'])
        optimizer.load_state_dict(ckpts['optimizer_state_dict'])
        start_epoch = ckpts['epoch']+1
        best_accuracy = ckpts['best_val_acc']

        if rank == 0:
            print('\nResuming training from epoch {}. Loaded weights from {}. Last best accuracy was {}'
                .format(start_epoch, ckptfile, best_accuracy))

    # Train the network
    train, val = split(folds, fold)
    train_loader, train_sampler = get_dataset(world_size, rank, train,
                                            phase=phases[0], 
                                            transform=tx_dict['train'], 
                                            batch_size=batch_size)
    val_loader, val_sampler = get_dataset(world_size, rank, val,
                                        phase=phases[1],transform=tx_dict['val'], 
                                        batch_size=batch_size)

    if wb:
        wandb.login()
        wandb.init(
            project="Pretrain", 
            name=f"train", 
            config={
            "architecture": "Siamese",
            "dataset": "SCR",
            "epochs": opt.epochs,
            })
    if wb:
        wandb.define_metric("train_loss", step_metric='train_step')
        wandb.define_metric("val_loss", step_metric='val_step')
        wandb.define_metric("epoch_loss", step_metric='epoch')
        wandb.define_metric("best_accuracy", step_metric='epoch')
        wandb.define_metric("correct", step_metric='epoch')
        wandb.define_metric("incorrect", step_metric='epoch')
        wandb.define_metric("accuracy", step_metric='epoch')
        wandb.define_metric("distance_train", step_metric='train_step')
        wandb.define_metric("distance_val", step_metric='val_step')

        if rank==0:
            wandb.define_metric("best_accuracy", summary="max")   
    
    if not validate_instead:
        for epoch in range(start_epoch, num_epochs):
            train_sampler.set_epoch(epoch)
            if wb:
                wandb.log({"epoch":epoch})
            best_accuracy = train_epoch(
                        rank, siamese_net, fold, optimizer, train_loader, 
                        val_loader, best_accuracy,
                        epoch, num_epochs, opt, running_loss=0
                        )
            lr_scheduler.step() 
            torch.cuda.empty_cache()
            gc.collect()
    else:
        thingy, save_this_dict = validate(rank, siamese_net, val_loader, 0, opt=opt)
    if wb:
        wandb.finish()
    
    return save_this_dict


__all__ = ['pretrainer', 'train_epoch', 'SiameseDataset', 'SiameseNetwork', 'contrastive_focal_loss', 'get_distance',
           'validate', 'tx', 'get_dataset', 'setup', 'cleanup']


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/work/bdgr/CSAT_2/', help='project root path')
    parser.add_argument('--world_size', type=int, default=1, help='World size')
    parser.add_argument('--resume', type=bool, default=False, help='To resume or not to resume')
    parser.add_argument('--train_folder', type=str, default='train2', help='name of the directory containing training samples')
    parser.add_argument('--val_folder', type=str, default='val2', help='name of the directory containing validation samples')    
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--folds', type=int, default=5, help='number of dataset folds for training')
    parser.add_argument('--cf', type=int, default=2, help='fold number to train. Must be provided if resume is not False')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--run_validation', type=bool, default=False, help='run validation instead of training')
    parser.add_argument('--resume_weight', type=str, default='/work/bdgr/CSAT_2/runs/bb_last_pretrainer_4.pth', help='path to trained weights file')

    return parser.parse_args()



if __name__ == '__main__':

    iterate = False
    opt = arg_parse()
    
    # Print the opt arguments, to show the values that are specified in the test.
    for arg in vars(opt):
        print(f'{arg}: {getattr(opt, arg)}')
    
    print("\n\n")

    save_dictionary = mp.spawn(pretrainer, args=(opt.world_size, opt), nprocs=opt.world_size, join=True)