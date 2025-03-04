# General Imports
import numpy as np
from tqdm import tqdm
import datetime
from glob import glob
import argparse
from collections import OrderedDict
import json
import pickle
import os

# PyTorch Imports
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# Custom Imports
from test2 import compute_val_stats, calculate_map, get_ap
from detector import get_loader, CSAT_detection  # assuming CSAT_detection is your detection model
from model.decoder import CSAT_detection
from model.encoder import Encoder
from utils.dataloader import create_dataloader
from loss.loss_criterion import loss_functions
from pretrain import *

def main_worker(rank, opts):
    # Set the rank for the current process
    opts.rank = rank

    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:12426',
                            world_size=opts.world_size,
                            rank=rank,
                            timeout=datetime.timedelta(seconds=5000)
                            )

    torch.cuda.set_device(rank)

    # Create the validator and run the validation
    validator = DetectorValidator(opts)
    info = validator.run()
    
    # Save info to ./info/info.json and ./info/info.pkl
    if os.path.exists('./info') == False: os.mkdir('./info')
    with open('./info/info.json', 'w') as f:
        json.dump(info, f)
    with open('./info/info.pkl', 'wb') as f:
        pickle.dump(info, f)

    # Clean up the process group
    dist.destroy_process_group()

class DetectorValidator:
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.encoder = self.initialize_encoder()
        self.val_loader = self.get_val_loader()

    def initialize_encoder(self):
        # Initialize the encoder with specified parameters
        encoder = Encoder(768, 8, 6, 0.1, multiscale=True, detector=True)
        
        # Wrap the encoder in a DataParallel model for multi-GPU training
        siamese_net = DataParallel(SiameseNetwork(encoder))
        
        # Load the checkpoint from the specified path
        ckpt = torch.load(self.opts.pretrained_weights_path, map_location='cpu')
        
        # Load the state dictionary into the model
        siamese_net.load_state_dict(ckpt['model_state_dict'])
        
        # Remove the DataParallel wrapper to access the original model
        siamese_net = siamese_net.module
        
        # Move the encoder to the specified device (GPU or CPU)
        encoder = siamese_net.encoder.to(self.device)
        
        # Set requires_grad to True for all encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Return the initialized encoder
        return encoder

    def get_val_loader(self):
        val_dataset = self.get_dataset(self.opts.dataroot, 'val2', self.opts.batch_size, self.opts.r, self.opts.space)
        val_loader = self.get_loader(val_dataset, self.opts.val_batch_size)
        return val_loader

    def get_loader(self, dataset, batch_size):
        data_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=6,
                                drop_last=False,
                                collate_fn=dataset.collate_fn)
        return data_loader

    def get_dataset(self, dataroot, phase, batch_size, r, space):
        data = glob(dataroot + '/*.pkl')
        dataset = create_dataloader(data,
                                    dataroot,
                                    batch_size,
                                    self.opts.rank,
                                    cache='ram',
                                    workers=6,
                                    phase=phase,
                                    shuffle=True,
                                    r=r,
                                    space=space)
        return dataset

    def load_saved_model(self, weights_path, root, M, O=None):
        ckptfile = weights_path
        ckpts = torch.load(ckptfile, map_location='cpu')
        ckpt = ckpts['model_state_dict']
        if O is None:
            new_state_dict = OrderedDict()
            for key, value in ckpt.items():
                new_key = key.replace('module.encoder.', '')
                new_state_dict[new_key] = value
        M.load_state_dict(new_state_dict)

    # Detector validation function
    def detector_validate(self, model, val_loader, device, iou_threshold=0.5):
        all_tps = []
        all_fps = []
        all_precisions = []
        all_recalls = []
        stuff_list_of_stuff = []

        with torch.no_grad():
            for i, (images, targets, _) in enumerate(tqdm(val_loader, desc="Validation")):
                if self.opts.debug and self.opts.validation_iter_stop:
                    if i >= self.opts.validation_stop_round:
                        print(f"Validation stopped after {self.opts.validation_stop_round} iterations.")
                        break
                
                # Move images and targets to the appropriate device
                images = images.to(device)
                targets = [t.to(device) for t in targets]  # Ensure targets are on the same device

                # Get model predictions
                outputs = model(images)
                
                # Print Targets and First Outputs
                if self.opts.debug: print(f"Targets: {targets}")
                # if self.opts.debug: print(f"Outputs: {outputs[0][0][0]}")
                
                # Concatenate targets into a single tensor for this batch
                formatted_targets = torch.cat(targets, dim=0)  # Combine all targets from the batch

                # Calculate TPs and FPs using calculate_map
                tps, fps, list_of_stuff = calculate_map(formatted_targets, compute_val_stats(outputs, opts=self.opts), opts=self.opts)

                stuff_list_of_stuff.append(list_of_stuff)
                
                # Print tps and fps...
                if self.opts.debug: print(f"\n\nBatch {i+1}: TPs: {tps}, FPs: {fps}")
                
                # Append results to lists
                all_tps.extend(tps)
                all_fps.extend(fps)
                
                # Calculate precision and recall for this batch
                precision = [tp / (tp + fp) if (tp + fp) > 0 else 0 for tp, fp in zip(tps, fps)]
                recall = [tp / len(formatted_targets) for tp in tps if tp > 0]

                all_precisions.extend(precision)
                all_recalls.extend(recall)

        # Perform calculations outside the loop
        avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
        avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0

        # Print Above Calculations.
        # if self.opts.debug: print("Total Ground Truth Boxes:", total_gt_boxes)
        if self.opts.debug: print("All TPs:", all_tps)
        if self.opts.debug: print("All FPs:", all_fps)
        if self.opts.debug: print("Average Precision:", avg_precision)
        if self.opts.debug: print("Average Recall:", avg_recall)
        
        # Calculate Average Precision (AP)
        ap = get_ap(all_precisions, all_recalls, opts=self.opts)
        
        print("Validation mAP: {:.4f}".format(ap))
        
        return stuff_list_of_stuff


    def run(self):
        model = CSAT_detection(encoder=self.encoder, multiscale=True)
        model = model.to(self.device)
        stuff_of_stuff = self.detector_validate(model, self.val_loader, device=self.device, iou_threshold=0.5)
        return stuff_of_stuff
        
class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_args()
        self.args = self.parser.parse_args()
        self.print_args()

    def add_args(self):
        self.parser.add_argument('--rank', default=0, type=int)
        self.parser.add_argument('--world_size', default=1, type=int)
        self.parser.add_argument('--root', default='/work/bdgr/CSAT_2/', type=str)
        self.parser.add_argument('--dataroot', default='/work/bdgr/CSAT_2/pickle', type=str)
        self.parser.add_argument('--nc', default=3, type=int)
        self.parser.add_argument('--epochs', default=152, type=int)
        self.parser.add_argument('--r', default=3, type=int)
        self.parser.add_argument('--space', default=1, type=int)
        self.parser.add_argument('--batch_size', default=8, type=int)
        self.parser.add_argument('--val_batch_size', default=1, type=int)
        self.parser.add_argument('--chkpt', default='/work/bdgr/CSAT_2/outputs/detection_26_c.pth', type=str)
        self.parser.add_argument('--pretrained_weights_path', default='/work/bdgr/CSAT_2/runs_new/0/model_best.pth', type=str)
        self.parser.add_argument('--debug', default=False, type=bool)
        self.parser.add_argument('--validation_iter_stop', default=False, type=bool)
        self.parser.add_argument('--validation_stop_round', default=100, type=int)
        
    def print_args(self):
        print(f'#############################')
        print(f'## Options/Args ##')
        for arg in vars(self.args):
            print(f'{arg}: {getattr(self.args, arg)}')
        print(f'#############################\n')

if __name__ == '__main__':
    opts = Options().args

    # Set the number of processes to spawn
    opts.world_size = torch.cuda.device_count()

    # Spawn the processes
    mp.spawn(main_worker, nprocs=opts.world_size, args=(opts,))