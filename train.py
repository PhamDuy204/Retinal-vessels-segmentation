import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
from set_up_seed import *
import importlib
import torch
import torch.nn as nn
from utils import *
from tqdm.auto import tqdm
from eval import eval_for_seg
from datetime import datetime
import numpy as np
from data_preparation import get_all_training_set
from torch.multiprocessing import Process, Queue
from load_model import load_model_class,load_loss_class
import wandb
import math
set_seed(42)
parser = argparse.ArgumentParser(description="Input params")
parser.add_argument("-b", "--batch_size",type=int, default=1)
parser.add_argument("-e", "--epochs",type=int, default=100)
parser.add_argument("-lf", "--loss",type=str, default='abe_dice_loss')
parser.add_argument("-m", "--model",type=str, default='unet')
parser.add_argument("-lr", "--learning_rate",type=float, default=1e-4)
parser.add_argument("-p", "--patches",type=int, default=500)
parser.add_argument("-ps", "--patch_size",type=int, default=64)
parser.add_argument("-ch", "--chunk_size",type=int, default=None)
parser.add_argument("-k", "--key",type=str, default=None)
args = parser.parse_args()

wandb.login(key=args.key)

datasets = get_all_training_set('./data',args.batch_size,args.patches,args.patch_size)


class Trainer:
    def __init__(self,model,train_loader
                 ,val_loader,criterion,optimizer,scheduler,gpu_id,name,save_dir='./checkpoints',patch=False):
        self.model=model
        self.train_loader=train_loader
        self.val_loader= val_loader
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.gpu_id=gpu_id
        self.save_dir=save_dir
        self.name=name
        self.class_labels = {0: "background", 1: "object"}
        model_class_name = type(self.model).__name__
        self.model_name = model_class_name
        self.patch=patch
    def train(self,epochs=100):
        torch.cuda.set_device(self.gpu_id)
        self.model.cuda()

        wandb.watch(self.model, self.criterion, log="all", log_freq=100)

        best_avg = -1.0
        best_eval_score={
            'best_f1' :0,
            'best_acc' :0,
            'best_iou' : 0,
            'best_recall' : 0,
            'best_spe' : 0,
            'best_auc': 0,
            'best_dice': 0
        }

        best_metrics = None
        best_params=None
        save_e=0
        current_lr = self.optimizer.param_groups[0]['lr']
        for e in range(epochs):
            self.model.train()
            training_loss=0
            for sample in tqdm(self.train_loader):
                image,mask,edge=sample.values()
                if len(image.shape)>4:
                    image=image.flatten(0,1)
                    mask=mask.flatten(0,1)
                    edge=edge.flatten(0,1)
                image = image.cuda()
                mask = mask.cuda()
                edge = edge.cuda()
                if args.chunk_size is None:

                    chunk_size=min(math.ceil(image.shape[0]/args.batch_size),8*args.batch_size)
                else:
                    chunk_size = args.chunk_size
                image_chunks=torch.chunk(image,chunk_size)
                mask_chunks=torch.chunk(mask,chunk_size)
                edge_chunks=torch.chunk(edge,chunk_size)
                for n_image,n_mask,n_egde in zip(
                    image_chunks,mask_chunks,edge_chunks
                ):
                    if check_model_forward_args(self.model)==2:
                        pred_mask = self.model(n_image,n_egde)
                    else:
                        pred_mask = self.model(n_image)
                    loss = self.criterion(pred_mask,n_mask)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    training_loss+=loss.item()
            self.scheduler.step(training_loss)
            acc,f1,iou,recall,spe,auc,dice=eval_for_seg(self.model,self.val_loader,self.gpu_id,self.patch)
            scores={
                'acc':acc,
                'f1':f1,
                'iou':iou,
                'recall':recall,
                'spe':spe,
                'auc':auc,
                'dice':dice
            }
            for best_method in best_eval_score.keys():
                method = best_method.split('_')[-1]
                if scores[method]>best_eval_score[best_method]:
                    best_eval_score[best_method]=scores[method]
            avg_metric = (acc + f1 + iou + recall + spe + auc+dice) / 7
            with open("temp.log", "a") as f:
                f.write(
                    f"[Epoch {e+1}/{epochs}] Dataset: {self.name} | "
                    f"Loss: {training_loss:.4f} | "
                    f"Acc: {acc:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
                    f"Recall: {recall:.4f} | Specificity: {spe:.4f} | "
                    f"DiceScore: {dice:.4f}\n"
                )
            wandb.log({
                "epoch": e+1,
                "loss": training_loss,
                "val_acc": acc,
                "val_f1": f1,
                "val_iou": iou,
                "val_recall": recall,
                "val_specificity": spe,
                "val_auc": auc,
                "val_dice": dice,
                "val_avg_metric": avg_metric,
                "lr": current_lr,
            })
            if avg_metric > best_avg:
                best_avg = avg_metric
                best_metrics = (acc, f1, iou, recall, spe, auc)
                best_params=self.model.state_dict()
                save_e = e
        if best_metrics and best_params:
                best_model=load_model_class(args.model)(1,1)
                best_model.load_state_dict(best_params)
                best_model.eval()
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{args.model}_on_{self.name}_best.pt")
                torch.save(best_model, save_path)

                artifact = wandb.Artifact(name=f"{args.model}_{self.name}_pt", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
                wandb.save(save_path)
                with torch.no_grad():                                                                                                                                                        
                    ex_image,ex_mask,ex_edge = next(iter(self.val_loader)).values()
                    if check_model_forward_args(self.model)==2:
                        ex_pred_mask = best_model(ex_image,ex_edge)
                    else:
                        ex_pred_mask = best_model(ex_image)
                    if self.patch:
                        h,w = ex_mask.shape[-2:]
                        ex_pred_mask=ex_pred_mask[:,:,:h,:w]
                    ex_pred_mask=torch.where(ex_pred_mask>0.5,1,0)
                    for i in range(len(ex_image)):
                        image_np = ex_image[i].squeeze().detach().cpu().numpy()
                        if image_np.max() <= 1.0:
                            image_np=image_np*0.5+0.5
                            image_np = (image_np * 255).astype(np.uint8)
                        else:
                            image_np = image_np.astype(np.uint8)

                        pred_mask_np = ex_pred_mask[i].squeeze().detach().cpu().numpy().astype(np.uint8) * 255
                        true_mask_np = ex_mask[i].squeeze().detach().cpu().numpy().astype(np.uint8) * 255

                        wandb.log({
                            f"example_{i}_overlay_pred_mask": wandb.Image(
                                image_np,
                                masks={
                                    "pred": {
                                        "mask_data": pred_mask_np,
                                        "class_labels": self.class_labels,
                                    }
                                },
                                caption=f"Example {i} - Pred Mask"
                            ),
                            f"example_{i}_overlay_pred_true_mask": wandb.Image(
                                true_mask_np,
                                masks={
                                    "pred": {
                                        "mask_data": pred_mask_np,
                                        "class_labels": self.class_labels,
                                    }
                                },
                                caption=f"Example {i} - Pred Mask"
                            ),
                            f"example_{i}_true_mask_only": wandb.Image(
                                true_mask_np,
                                caption=f"Example {i} - True Mask Only"
                            ),
                            f"example_{i}_pred_mask_only": wandb.Image(
                                pred_mask_np,
                                caption=f"Example {i} - Pred Mask Only"
                            )
                        })
        wandb.summary["best_avg_metric"] = best_avg
        wandb.summary["best_epoch"] = save_e
        for best_method in best_eval_score.keys():
             wandb.summary[best_method]=best_eval_score[best_method]
        return best_avg 
def gpu_worker(gpu_id, task_queue, result_queue):

    torch.cuda.set_device(gpu_id)
    while not task_queue.empty():
        try:
            dataset_id = task_queue.get_nowait()
        except:
            break
        info = datasets[dataset_id]
        train_loader = info['train_loader']
        val_loader   = info['val_loader']
        name         = info['name']
        patch = info['patches']
        seg_model=load_model_class(args.model)
        model = seg_model(1,1)
        num_params=count_trainable_params(model)
        model_class_name = type(model).__name__
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        try:
            wandb.init(
                    entity='phamdinhanhduy-university-of-information-and-technology',
                    project="Retinal-Vessels-Segmentation",
                    name=f"{name} GPU{gpu_id} Model {args.model} in {timestamp}",
                    config={
                        "dataset": name,
                        "model": model_class_name,
                        "optimizer": "Adam",
                        "lr": args.learning_rate,
                        "epochs": args.epochs,
                        "gpu": gpu_id,
                    },
                    reinit=True,
                )

            criterion = load_loss_class(args.loss)()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.6, patience=5
            )
            # ----------------------------------------------------------------
            trainer = Trainer(
                model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                gpu_id, name, save_dir='./checkpoints',patch=patch
            )

            best_avg = trainer.train(epochs=args.epochs) 

            result_queue.put((name, best_avg))
            wandb.summary["num params"] = num_params
            wandb.finish()
        except Exception as ex:
            print(f"[GPU {gpu_id}] train on {name} has error: {ex}")
            wandb.finish()



if __name__ == '__main__':
    set_seed(42)

    torch.multiprocessing.set_start_method('spawn')

    num_datasets = len(datasets)
    print((num_datasets))

    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    NUM_GPUS = min(NUM_GPUS, 4) 
    if NUM_GPUS == 0:
        raise RuntimeError("Không tìm thấy GPU nào, phải chạy trên ít nhất 1 GPU.")

    task_queue = Queue()
    for idx in range(num_datasets):
        task_queue.put(idx)

    result_queue = Queue()

    processes = []
    for gpu_id in range(NUM_GPUS):
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()
    results = []
    while not result_queue.empty():
        try:
            name, best_avg = result_queue.get_nowait()
            results.append((name, best_avg))
        except:
            break

    # Tìm dataset (mô hình) có best_avg cao nhất
    if len(results) > 0:
        # results: list of (name, best_avg)
        results.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = results[0]
        print("\n\n=========================")
        print(f"Dataset/mô hình có Eval cao nhất: {best_name} với AvgMetric = {best_score:.4f}")
        print("=========================")
    else:
        print("Không có kết quả nào được trả về từ các process.")
        
            
