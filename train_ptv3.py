import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import os, time
import numpy as np
import yaml, argparse
from functools import partial
from model import PointTransformerV3
from gs_dataloader import load_data_cfg, log2file, reClick, Data_Input_GS_1, mergelargeGS_1
from tqdm import tqdm
from metrics import ConfusionMatrix, AverageMeter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

logger = log2file()

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg



def train(model, dataloader, optimizer, criterion, device, scheduler, cfg,epoch=1):
    model.train()
    cm = ConfusionMatrix(num_classes=cfg['num_classes'], ignore_index=None)
    loss_meter = AverageMeter()
    pbar = tqdm(enumerate(dataloader), total=dataloader.__len__())
    for idx, data in pbar:
        points, labels = data
        (B1,N1,C1) = points.shape
        feat = points[:, :, 3:]
        coord= points[:, :, :3]
        feat_flat = feat.view(B1 * N1, -1)     
        coord_flat = coord.view(B1 * N1, 3)    
        batch = torch.arange(B1, dtype=torch.long).unsqueeze(1).repeat(1, N1).view(-1)  
        grid_size = cfg['grid_size']
        data_dict = {
            'feat': feat_flat.to(device),  
            'coord': coord_flat.to(device), 
            'batch': batch.to(device),
            'grid_size':grid_size
        }

        optimizer.zero_grad()
        logits = model(data_dict)  
        logits = logits.view(-1, logits.shape[-1])  
        labels = labels.view(-1).to(device)  
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        cm.update(logits.argmax(dim=-1), labels)
        loss_meter.update(loss.item())
    pbar.set_description(f"Train Epoch [{epoch}/{cfg['epoch']}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs

def test(model, dataloader, criterion, device, cfg):
    model.eval()
    cm = ConfusionMatrix(num_classes=cfg['num_classes'], ignore_index=None)
    loss_meter = AverageMeter()
    start_time = time.time()

    pre_list = []
    logits_list = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=dataloader.__len__())       
        for idx, data in pbar:
            points, labels = data
            (B1,N1,C1) = points.shape
            feat = points[:, :, 3:]
            coord= points[:, :, :3]
            
            feat_flat = feat.view(B1 * N1, -1)     
            coord_flat = coord.view(B1 * N1, 3)    
            
            batch = torch.arange(B1, dtype=torch.long).unsqueeze(1).repeat(1, N1).view(-1)  
            grid_size = cfg['grid_size']
            
            data_dict = {
                'feat': feat_flat.to(device),  
                'coord': coord_flat.to(device), 
                'batch': batch.to(device),
                'grid_size':grid_size
            }
            logits = model(data_dict)  
            logits = logits.view(-1, logits.shape[-1])  
            labels = labels.view(-1).to(device)  
            loss = criterion(logits, labels)
            cm.update(logits.argmax(dim=-1), labels)
            loss_meter.update(loss.item())

            pred = np.argmax(logits.cpu(), -1)
            pred = np.array(pred)
            pred = np.round(pred)
            pred = pred.reshape(-1,cfg['num_point'])
            pred = pred.astype(int)
            
            pre_list.append(pred)
              
            logits_list.append(logits.cpu().reshape(-1,cfg['num_point'],cfg['num_classes']))
    pre_list = np.concatenate(pre_list, axis=0)
    logits_list = np.concatenate(logits_list, axis=0)
    end_time = time.time()  # record the time
    miou, macc, oa, ious, accs = cm.all_metrics()
    elpase_time = end_time - start_time

    return pre_list, logits_list, loss_meter.avg, miou, macc, oa, ious, accs, elpase_time

    
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointTransformerV3(in_channels=2, num_classes=cfg['num_classes']).to(device)     
    criterion = nn.CrossEntropyLoss()
    ############## 
    train_loader, val_loader = load_data_cfg(cfg)
    ############## 
    # 
    if cfg['optimizer']['type'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg['optimizer']['type']}")
    
    # 
    if cfg['scheduler']['type'] == "OneCycleLR":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg['scheduler']['max_lr'],  
            steps_per_epoch=len(train_loader),  
            epochs=cfg['epoch'],                  
            pct_start=cfg['scheduler']['pct_start'],
            anneal_strategy=cfg['scheduler']['anneal_strategy'],
            div_factor=cfg['scheduler']['div_factor'],
            final_div_factor=cfg['scheduler']['final_div_factor']
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {cfg['scheduler']['type']}")
    ## -----------------------------------------------------------------------------
    
    num_epochs = cfg['epoch']
    miou_best = 0.0
    # logger = log2file()
    for epoch in range(num_epochs):
        loss_t, miou_t, macc_t, oa_t, ious_t, accs_t    = train(model, train_loader, optimizer, criterion, device, scheduler, cfg)
        _, _, loss, miou, macc, oa, ious, accs, elpase_time   = test(model, val_loader, criterion, device, cfg)

        #################### 
        lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch} LR {lr:.6f}  '
                     f'train_miou {miou_t:.2f}, val_miou {miou:.2f}, best val miou {miou_best:.2f}')        
        
        if miou_best < miou:
            torch.save(model.state_dict(), cfg['model_path'])
            logger.info(
                        f'Find a better ckpt @E{epoch}, val_miou {miou:.2f} val_macc {macc:.2f}, val_oa {oa:.2f}'
                        f'\nmious: {ious}')
            miou_best = miou



def infer_onescene(scene_name, model, cfg):
    test_miou, test_macc, test_oa, test_ious, test_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    preds_list = []
    clicks_list = []
    thr = 0.99
    scene_path = os.path.join(BASE_DIR,cfg['data_dir'],cfg['testname'],scene_name)
    start_time = time.time()
    test_dataset, test_clicks, pos_labels, one_shot = Data_Input_GS_1(scene_path, cfg)
    end_time = time.time()
    print(f'Data pre-processing loading time:{end_time-start_time}')
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=cfg['batch_size'],
                                             num_workers=0,
                                             shuffle=False,
                                             pin_memory=False)
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred, logits, loss, miou, macc, oa, ious, accs,time2 = test(model, val_loader, criterion, device, cfg)

    cc = np.expand_dims(pred, axis=0)
    preds_list.append(cc)
    # print(cc.shape)
    cc = np.expand_dims(test_clicks[0:pred.shape[0], ...], axis=0)
    clicks_list.append(cc)
    # print(cc.shape)

    click_num_p = np.ones((pred.shape[0]), dtype=int)
    if test_miou is not None:
        with np.printoptions(precision=2, suppress=True):
            logger.info(
                f'test_oa:{test_oa:.2f} , test_macc:  {test_macc:.2f}, test_miou: {test_miou:.2f}, '
                f'\n epoch:{1},iou per cls is: {test_ious}, avg_click:{np.mean(click_num_p)}, time: {time2:.6f}')

    new_test = test_dataset
    avg_time = 0
    for i in range(1, cfg['n_clicks']):
        new_test, click_num_p, test_clicks = reClick(pred, new_test, click_num_p, thr)

        test_loader = torch.utils.data.DataLoader(new_test,
                                                  batch_size=cfg['batch_size'],
                                                  num_workers=0,
                                                  shuffle=False,
                                                  pin_memory=False)
        pred, logits, loss, test_miou, test_macc, test_oa, test_ious, test_accs,time2 = test(model, test_loader,  criterion, device, cfg)
        avg_time += time2
        cc = np.expand_dims(pred, axis=0)
        preds_list.append(cc)
        # print(cc.shape)
        cc = np.expand_dims(test_clicks, axis=0)
        clicks_list.append(cc)
        # print(cc.shape)
        if test_miou is not None:
            with np.printoptions(precision=2, suppress=True):
                logger.info(
                    f'epoch:{i + 1},test_oa:{test_oa:.2f} , test_macc:  {test_macc:.2f}, test_miou: {test_miou:.2f}, '
                    f'iou per cls is: {test_ious}, avg_click:{np.mean(click_num_p)}, time: {time2:.6f}')

    # preds_list = np.concatenate(preds_list, axis=0)
    # clicks_list = np.concatenate(clicks_list, axis=0)
    # saveForfig3('data/testsave/'+cfg['dataset'].common.NAME,test_dataset,preds_list,clicks_list,click_num_p)
    avg_time = avg_time / cfg['n_clicks']
    logger.info(f'avg_time per click:{avg_time:.6f}')
    mergelargeGS_1(one_shot, scene_name, pred, pos_labels, logits, cfg, logger)


"""
Test a completed scene
"""
def testScene(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointTransformerV3(in_channels=2, num_classes=cfg['num_classes']).to(device)    
    model_path = BASE_DIR + cfg['read_modelpath']##os.path.join(BASE_DIR, cfg['read_modelpath'])
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()
    ############## 
    scene_path = os.path.join(BASE_DIR,cfg['data_dir'],cfg['testname'])
    # print(scene_path)
    scene_names = os.listdir(scene_path)    
    for scene_name in scene_names:
        # print(scene_name)
        infer_onescene(scene_name, model, cfg)
    
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, default='tree_ptv3_b.yaml', help='config file')
    parser.add_argument('--data_channel', type=str, default='0,1,2', help='set the input channels of network')
    parser.add_argument('--grid_size', type=float, default=0.01)
    parser.add_argument('--log_file', type=str, default='ptv3.log')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--model', default="checkpoints/b80_gr_002_8192_92_58.pth")
    parser.add_argument('--testscene', action='store_true', default=False, help='set to test whole scene or just h5 dataset')
    parser.add_argument('--testGS', action='store_true', default=False, help='set to test gaussian scenes')
    parser.add_argument('--n_clicks', type=int, default=2, help='set the num of clicks for each objects')
    parser.add_argument('--force_Divide', action='store_true', default=False,
                        help='when points are not classified, force to divide')
    args, opts = parser.parse_known_args()
    cfg = load_cfg(args.cfg)
    cfg['data_channel'] = args.data_channel
    cfg['grid_size'] = args.grid_size
    cfg['log_file'] = args.log_file
    cfg['read_modelpath'] = args.model
    cfg['n_clicks'] = args.n_clicks
    cfg['force_Divide'] = args.force_Divide
    st_time = time.time()
    if args.testscene:
        testScene(cfg)
    else:
        main(cfg)
    ed_time = time.time()
    print(f'running time:{ed_time-st_time} s')