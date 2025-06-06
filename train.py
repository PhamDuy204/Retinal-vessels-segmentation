import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from utils import check_model_forward_args
from tqdm import tqdm
from eval import eval_for_seg
import logging
from data_preparation import get_all_training_set
from datetime import datetime
from torch.multiprocessing import Process, Queue
from models.unet.unet import UNETModel
from loss import AbeDiceLoss
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

        model_class_name = type(self.model).__name__
        timestamp = datetime.now().strftime('%Y%m%d_%H')
        log_filename = f"exp_on_{self.name}_{model_class_name}_{timestamp}.log"
        log_path = os.path.join('./logs', log_filename)
        os.makedirs('./logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger()
    def train(self,epochs=100):
        torch.cuda.set_device(self.gpu_id)
        self.model.cuda()
        best_avg = -1.0
        best_metrics = None
        best_params=None
        save_e=0
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
            self.logger.info(
                f"[Epoch {e+1}/{epochs}] Dataset: {self.name} | "
                f"Loss: {training_loss:.4f} | "
                f"Acc: {acc:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | "
                f"Recall: {recall:.4f} | Specificity: {spe:.4f} | "
            )
            if avg_metric > best_avg:
                best_avg = avg_metric
                best_metrics = (acc, f1, iou, recall, spe, auc)
                best_params=self.model.state_dict()
                save_e = e
        if best_metrics and best_params:
                os.makedirs(self.save_dir, exist_ok=True)
                save_path = os.path.join(self.save_dir, f"{self.name}_best.pth")
                torch.save(best_params, save_path)
                self.logger.info(f"Saved best model for {self.name} at epoch {save_e}")
                self.logger.info(f"Best metrics: Acc={best_metrics[0]:.4f}, F1={best_metrics[1]:.4f}, "
                                f"IoU={best_metrics[2]:.4f}, Recall={best_metrics[3]:.4f}, "
                                f"Spe={best_metrics[4]:.4f}, AUC={best_metrics[5]:.4f}, "
                )
        return best_avg
datasets = get_all_training_set('./data')  
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

        model = nn.Sequential(UNETModel(1,1),nn.Sigmoid())

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

        best_avg = trainer.train(epochs=100) 

        result_queue.put((name, best_avg))



if __name__ == '__main__':
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
        
            
