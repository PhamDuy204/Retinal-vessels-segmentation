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
<<<<<<< HEAD
import traceback # <<< THÊM DÒNG NÀY

=======

from adabelief_pytorch import AdaBelief
>>>>>>> origin/main
set_seed(42)
parser = argparse.ArgumentParser(description="Input params")
parser.add_argument("-b", "--batch_size",type=int, default=4)
parser.add_argument("-e", "--epochs",type=int, default=100)
parser.add_argument("-lf", "--loss",type=str, default='abe_dice_loss')
parser.add_argument("-m", "--model",type=str, default='unet')
parser.add_argument("-lr", "--learning_rate",type=float, default=0.001)
parser.add_argument("-p", "--patches",type=int, default=500)
parser.add_argument("-ps", "--patch_size",type=int, default=64)
parser.add_argument("-tt", "--train_type",type=str, default='patch')
parser.add_argument("-ch", "--chunk_size",type=int, default=None)
parser.add_argument("-k", "--key",type=str, default=None)
parser.add_argument("-ts", "--type_split",type=str, default='window')
args = parser.parse_args()

wandb.login(key=args.key)

datasets = get_all_training_set('./data',args.batch_size,args.patches,args.patch_size,args.train_type,args.type_split)


class Trainer:
    def __init__(self,model,train_loader
                 ,val_loader,criterion,optimizer,scheduler,gpu_id,name,save_dir='./checkpoints',patch=False,type_split=args.type_split):
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
        self.type_split=type_split
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
            'best_dice': 0,
            'best_threshold':0
        }

        best_metrics = None
        best_params=None
        save_e=0
        current_lr = None
        for e in range(epochs):
            self.model.train()
            training_loss=0
            for sample in tqdm(self.train_loader):
                image,mask,edge=sample.values()

                if len(image.shape)>4:
                    image=image.flatten(0,1)
                    mask=mask.flatten(0,1)
                    edge=edge.flatten(0,1)
                
                random_index = torch.randperm(image.size(0))
                image=image[random_index]
                edge=edge[random_index]
                mask=mask[random_index]

                if args.chunk_size is None:
                    chunk_size=max(min(math.ceil(image.shape[0]/args.batch_size),16*args.batch_size),1)
                else:
                    chunk_size = args.chunk_size
                image_chunks=torch.chunk(image,chunk_size)
                mask_chunks=torch.chunk(mask,chunk_size)
                edge_chunks=torch.chunk(edge,chunk_size)
                for n_image,n_mask,n_egde in zip(
                    image_chunks,mask_chunks,edge_chunks
                ):
                    n_image = n_image.cuda()
                    n_mask = n_mask.cuda()
                    n_egde = n_egde.cuda()

                    if check_model_forward_args(self.model)==2:
                        pred_mask = self.model(n_image,n_egde)
                    else:
                        pred_mask = self.model(n_image)
                    loss = self.criterion(pred_mask,n_mask)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    training_loss+=loss.item()
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            acc,f1,iou,recall,spe,auc,dice,best_threshold=eval_for_seg(self.model,self.val_loader,self.gpu_id,self.patch,args.patch_size,self.type_split)
            scores={
                'acc':acc,
                'f1':f1,
                'iou':iou,
                'recall':recall,
                'spe':spe,
                'auc':auc,
                'dice':dice,
                'threshold':best_threshold
            }
            for best_method in best_eval_score.keys():
                method = best_method.split('_')[-1]
                if scores[method]>best_eval_score[best_method]:
                    best_eval_score[best_method]=scores[method]
            avg_metric = (acc + f1 + iou + recall + spe + auc+dice) / 7
            best_metric_eval = (f1 + recall+auc+acc)/4
            with open("temp.log", "a") as f:
                f.write(
                    f"[Epoch {e+1}/{epochs}] Dataset: {self.name} | "
                    f"Loss: {training_loss:.4f} | "
                    f"Acc: {acc:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
                    f"Recall: {recall:.4f} | Specificity: {spe:.4f} | "
                    f"DiceScore: {dice:.4f} | AUC: {auc:.4f}\n"
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
                "val_threshold": best_threshold,
                "val_avg_metric": avg_metric,
                "lr": current_lr,
            })
            if best_metric_eval > best_avg:
                best_avg = best_metric_eval
                best_metrics = (acc, f1, iou, recall, spe, auc)
                best_params=self.model.state_dict()
                save_e = e
        if best_metrics and best_params:
                torch.cuda.empty_cache()
                best_model=load_model_class(args.model)(1,1).cuda()
                if self.patch:
                    _=best_model(torch.rand(1,1,args.patch_size,args.patch_size).cuda())
                best_model.zero_grad()
                best_model.load_state_dict({k: v.cuda() for k, v in best_params.items()},strict=False)
                best_model.eval()
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{args.model}_on_{self.name}_best.pt")
                save_model_folder_path=os.path.join(os.path.dirname(__file__),f'models/{args.model}/')
                torch.save(best_model, save_path)

                artifact = wandb.Artifact(name=f"{args.model}_{self.name}_pt", type="model")
                artifact.add_file(save_path)
                artifact.add_file(os.path.join(os.path.dirname(__file__),f'loss/{args.loss}.py'))
                artifact.add_dir(save_model_folder_path)
                wandb.log_artifact(artifact)
                wandb.save(save_path)
                wandb.save(save_model_folder_path)
                with torch.inference_mode():                                                                                                                                                        
                    ex_image,ex_mask,ex_edge = next(iter(self.val_loader)).values()

                    ex_image=mirror_padding_v2(ex_image)
                    ex_edge=mirror_padding_v2(ex_edge)

                    B,C,H,W = ex_image.shape           
                    ex_image=ex_image.cuda()
                    ex_mask=ex_mask.cuda()
                    ex_edge=ex_edge.cuda()

                    stride=None
                    if self.patch and self.type_split!='random':
                        num_patch=((H-args.patch_size)//32+1,(W-args.patch_size)//8+1)
                        ex_image,tmp_stride = extract_patches_with_target_count(ex_image,args.patch_size,num_patch)
                        ex_edge,_ = extract_patches_with_target_count(ex_edge,args.patch_size,num_patch)
                        stride=tmp_stride
                        if len(ex_image.shape)>4:
                            ex_image=ex_image.flatten(0,1)
                            ex_edge=ex_edge.flatten(0,1)
                    out_sample=[]
                    chunk_size = max(ex_image.shape[0]//200,1)
                    chunk_image = torch.chunk(ex_image,chunk_size,0)
                    chunk_edge = torch.chunk(ex_edge,chunk_size,0)
                    for chunk in zip(chunk_image,chunk_edge):
                        c_image,c_edge=chunk
                        if check_model_forward_args(best_model) == 2:
                            prob = best_model(c_image, c_edge)
                        else:
                            prob = best_model(c_image)
                        out_sample.append(prob)
                    ex_pred_mask= torch.cat(out_sample,0)
                    if self.patch:
                        if stride is not None:
                            ex_pred_mask = ex_pred_mask.view(B,-1,1,args.patch_size,args.patch_size)
                            ex_image = ex_image.view(B,-1,C,args.patch_size,args.patch_size)
                            ex_pred_mask=reverse_to_original_image(ex_pred_mask,(H,W),args.patch_size,stride)
                            ex_image=reverse_to_original_image(ex_image,(H,W),args.patch_size,stride)
                        h,w = ex_mask.shape[-2:]
                        ex_pred_mask=ex_pred_mask[:,:,:h,:w]
                        ex_image=ex_image[:,:,:h,:w]
<<<<<<< HEAD
                    ex_pred_mask=torch.where(ex_pred_mask>=0.485,1,0)

=======
                    ex_pred_mask=torch.where(ex_pred_mask>=0.487,1,0)
                    # print(ex_pred_mask.shape)
                    # print(ex_image.shape)
>>>>>>> origin/main
                    for i in range(len(ex_image)):
                        image_np = ex_image[i].permute(1,2,0).mean(-1).squeeze().detach().cpu().numpy()
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
        model = seg_model(1,1).cuda()
        model.apply(init_weights_kaiming)
        try:
            for m in [model.awl, model.up_f_0[-1], model.up_f_1[-1]]:
                m.apply(init_weights_xavier)
        except:
            pass
        num_params=count_trainable_params(model)
        print(num_params)
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
                        "batch_size": args.batch_size,
                        "num_p": num_params,
                        "num_patch": None if ~patch else args.patch_size,
                        'type_split': None if ~patch else args.type_split,
                        "optimizer": "Adam",
                        "lr": args.learning_rate,
                        "epochs": args.epochs,
                        "gpu": gpu_id,
                    },
                    reinit=True,
                )

            criterion = load_loss_class(args.loss)()
            optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=2e-6)
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
        # <<< KHỐI CATCH LỖI ĐÃ ĐƯỢC CHỈNH SỬA >>>
        except Exception as ex:
            # Lấy thông tin traceback chi tiết dưới dạng chuỗi
            detailed_error = traceback.format_exc()
            
            # In thông báo lỗi ngắn gọn và cả traceback chi tiết
            print(f"[GPU {gpu_id}] train on {name} has error: {ex}")
            print("\n---------- Full Traceback ----------")
            print(detailed_error)
            print("------------------------------------\n")
            
            # Đảm bảo wandb được kết thúc đúng cách
            if wandb.run is not None:
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