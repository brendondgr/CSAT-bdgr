import argparse
import torch
import gc
import os
from tqdm import tqdm
import wandb
import torch.multiprocessing as mp
import torch.nn as nn
from pretrain import Encoder, SiameseNetwork, DDP 
from pretrain import setup, split, get_dataset, tx, contrastive_focal_loss, get_next_checkpoint_number
from eval.eval import Meter
from torchvision.ops import sigmoid_focal_loss
import json


TQDM_BAR_FORMAT = '{desc} {n_fmt}/{total_fmt} [{elapsed} | {remaining} | {rate_fmt}]' #'{l_bar}{r_bar}' #'{l_bar}{r_bar}' # tqdm bar format
wb = False

def save_model(root, siamese_net, epoch, optimizer, acc, best_accuracy, fold, opt):
    # Define the directory for saving models
    save_dir = os.path.join(root, opt.temp_save, str(opt.cf))
    os.makedirs(save_dir, exist_ok=True)
    
    print('#########################################')
    print(f'Saving model for epoch {epoch}... (save_model function)')
    # Determine the filename for the current epoch
    current_model_name = f'model_epoch_{epoch}.pth'
    current_model_path = os.path.join(save_dir, current_model_name)
    
    # Save the current model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': siamese_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_accuracy.item() if isinstance(best_accuracy, torch.Tensor) else best_accuracy,
    }
    torch.save(checkpoint, current_model_path)
    print(f'Model saved at {current_model_path}.')

    # If this is the best model so far, save it as 'model_best.pth'
    if acc.to(best_accuracy.device) >= best_accuracy:
        print(f'New best accuracy: {acc} (Previous best: {best_accuracy})')
        print(f'Saving best model at {save_dir}/model_best.pth')
        best_model_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(checkpoint, best_model_path)
        best_accuracy = acc
    
    print('#########################################')

    return best_accuracy

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
        
        # if batch_idx == 10:
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
    
    if opt.run_validation: all_distances = []

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
            if opt.run_validation: all_distances.append({"val_distance": dist.mean().item(), "ground_truth": targets.cpu().numpy().tolist()})
            ###############################
            
            loss = sigmoid_focal_loss(dist, targets, alpha=0.15, gamma=2.0, reduction='sum') #contrastive_focal_loss(rank, embeddings1[:,-1], embeddings2[:,-1], targets, phase='val')
            
            threshold = torch.ones_like(dist)*thres
            op = torch.relu(torch.sign(threshold-dist))


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

            # if batch_idx == 5:
            #     break
            
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
    
    # get_next_checkpoint_number(root + SAVE_PATH)

    
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
        # print(ckpts['optimizer_state_dict'])
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
    
    # For this, Return the Model...
    return siamese_net, optimizer, best_accuracy

def main(rank, args):
    total_epochs = args.epochs
    save_interval = args.save_per_x
    rank = args.rank
    world_size = args.world_size
    root = args.root
    save_folder = args.temp_save

    # Loop through epochs in batches of save_interval
    for start_epoch in range(0, total_epochs, save_interval):
        end_epoch = start_epoch + save_interval
        
        # Run pretrainer for this batch of epochs
        if start_epoch == 0:
            resume_weight = ''
            resume = False
        else:
            resume_weight = os.path.join(root, save_folder, str(args.cf), f'model_epoch_{start_epoch}.pth')
            resume = True
        
        # Adjust args.epochs to end_epoch, resume_weight to resume_weight, and resume to resume
        args.epochs = end_epoch
        args.resume_weight = resume_weight
        args.resume = resume
        
        siamese_model, optimizer, best_accuracy = pretrainer(rank, world_size, args)

        # Save model and clear memory
        model_path = os.path.join(root, save_folder, str(args.cf), f'model_epoch_{end_epoch}.pth')
        print('\n#########################################')
        print(f'Saving model for epochs {start_epoch} to {end_epoch}.')
        torch.save({
            'epoch': end_epoch,
            'model_state_dict': siamese_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_accuracy
        }, model_path)
        print(f'Model Saved at {model_path}.')
        torch.cuda.empty_cache()
        gc.collect()
        print('#########################################\n')

def naming_dict():
    return {
        'epochs': 'Epochs',
        'save_per_x': 'Save Interval',
        'rank': 'Rank',
        'world_size': 'World Size',
        'root': 'Root',
        'resume': 'Resume',
        'train_folder': 'Train Folder',
        'val_folder': 'Validation Folder',
        'folds': 'Folds',
        'cf': 'Current Fold',
        'batch_size': 'Batch Size',
        'run_validation': 'Run Validation',
        'resume_weight': 'Weight to Resume',
        'temp_save': 'Save Folder for Temp Models',
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain Looper')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of epochs to run')
    parser.add_argument('--save_per_x', type=int, default=10, help='Number of epochs between each save')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the process')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of processes')
    parser.add_argument('--root', type=str, default='/work/bdgr/CSAT_2/', help='Root directory for saving models')
    parser.add_argument('--resume', type=bool, default=False, help='To resume or not to resume')
    parser.add_argument('--train_folder', type=str, default='train2', help='Name of the directory containing training samples')
    parser.add_argument('--val_folder', type=str, default='val2', help='Name of the directory containing validation samples')
    parser.add_argument('--folds', type=int, default=5, help='Number of dataset folds for training')
    parser.add_argument('--cf', type=int, default=0, help='Fold number to train. Must be provided if resume is not False')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--run_validation', type=bool, default=False, help='Run validation instead of training')
    parser.add_argument('--resume_weight', type=str, default='', help='Path to trained weights file')
    parser.add_argument('--temp_save', type=str, default='runs_new', help='Directory for saving temporary models')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    
    dict_args = naming_dict()
    
    print('#########################################')
    for arg in vars(opt):
        name = dict_args[arg]
        print(f'{name}: {getattr(opt, arg)}')
    print('#########################################\n')
    
    mp.spawn(main, args=(opt,), nprocs=opt.world_size, join=True)