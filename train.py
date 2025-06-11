import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
from set_up_seed import *
import torch
import torch.nn as nn
from utils import check_model_forward_args
from tqdm.auto import tqdm
from eval import eval_for_seg
from datetime import datetime
import numpy as np
from data_preparation import get_all_training_set

from torch.multiprocessing import Process, Queue
from loss import AbeDiceLoss
from load_model import load_model_class
import wandb
wandb.login(key="a5e9e41c35cd5e4b1c8c726d53b6b5700cd55b0d")

set_seed(42)
parser = argparse.ArgumentParser(description="Input params")
parser.add_argument("-b", "--batch_size",type=int, default=1)
parser.add_argument("-e", "--epochs",type=int, default=100)
parser.add_argument("-m", "--model",type=str, default='unet')
args = parser.parse_args()
datasets = get_all_training_set('./data',args.batch_size)


class Trainer:
    def __init__(self,model,train_loader
                 ,val_loader,criterion,optimizer,scheduler,gpu_id,name,save_dir='./checkpoints'):
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
    def train(self,epochs=100):
        torch.cuda.set_device(self.gpu_id)
        self.model.cuda()

        wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        best_avg = -1.0
        best_metrics = None
        best_params=None
        save_e=0
        current_lr = self.optimizer.param_groups[0]['lr']
        for e in range(epochs):
            self.model.train()
            training_loss=0
            for sample in tqdm(self.train_loader):
                image,mask,edge=sample.values()
                image = image.cuda()
                mask = mask.cuda()
                edge = edge.cuda()

                if check_model_forward_args(self.model)==2:
                    pred_mask = self.model(image,edge)
                else:
                    pred_mask = self.model(image)
                loss = self.criterion(pred_mask,mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss+=loss.item()
            self.scheduler.step(training_loss)
            acc,f1,iou,recall,spe,auc=eval_for_seg(self.model,self.val_loader,self.gpu_id)
            avg_metric = (acc + f1 + iou + recall + spe + auc) / 6
            print(
                f"[Epoch {e+1}/{epochs}] Dataset: {self.name} | "
                f"Loss: {training_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
                f"Recall: {recall:.4f} | Specificity: {spe:.4f} | "
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
                best_model.cuda().eval()
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{self.model_name}_on_{self.name}_best.onnx")
                torch.onnx.export(best_model, torch.rand(1,1,512,512).cuda(), save_path, opset_version=11)

                artifact = wandb.Artifact(name=f"{self.model_name}_{self.name}_onnx", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)
                wandb.save(save_path)
                with torch.no_grad():
                    ex_image,ex_mask,ex_edge = next(iter(self.val_loader)).values()
                    if check_model_forward_args(self.model)==2:
                        ex_pred_mask = best_model(ex_image.cuda(),ex_edge.cuda())
                    else:
                        ex_pred_mask = best_model(ex_image.cuda())
                    ex_pred_mask=torch.where(ex_pred_mask>0.5,1,0)
                    for i in range(len(ex_image)):

                        masks = { 
                            "pred": {
                                "mask_data": ex_pred_mask[i].squeeze().detach().cpu().numpy().astype(int),
                                "class_labels":  self.class_labels,
                            },
                            "true": {
                                "mask_data": ex_mask[i].squeeze().detach().cpu().numpy().astype(int),
                                "class_labels":  self.class_labels,
                            },
                        }
                        wandb.log({
                            f"example_{i}": wandb.Image(
                                ex_image[i].squeeze().detach().cpu().numpy(),
                                masks=masks,
                                caption=f"Example {i}"
                            )
                        })
        wandb.summary["best_avg_metric"] = best_avg
        wandb.summary["best_epoch"] = save_e
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
        seg_model=load_model_class(args.model)
        model = seg_model(1,1)
        model_class_name = type(model).__name__
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        try:
            wandb.init(
                    project="Retinal-Segmentation ",
                    name=f"{name}_GPU{gpu_id}_{timestamp}",
                    config={
                        "dataset": name,
                        "model": model_class_name,
                        "optimizer": "Adam",
                        "lr": 1e-3,
                        "epochs": args.epochs,
                        "gpu": gpu_id,
                    },
                    reinit=True,
                )

            criterion = AbeDiceLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=5
            )
            # ----------------------------------------------------------------
            trainer = Trainer(
                model, train_loader, val_loader,
                criterion, optimizer, scheduler,
                gpu_id, name, save_dir='./checkpoints'
            )

            best_avg = trainer.train(epochs=args.epochs) 

            result_queue.put((name, best_avg))
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
        
            
