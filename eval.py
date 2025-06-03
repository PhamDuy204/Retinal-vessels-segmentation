import torch
from utils import check_model_forward_args
from torchmetrics.classification import Accuracy,BinaryF1Score,\
                                        AUROC, Recall, Specificity,\
                                        JaccardIndex
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
def eval_for_seg(model,val_loader,gpu_id):
    torch.cuda.set_device(gpu_id)
    with torch.no_grad():
        model.eval()
        probs= []
        truth_label=[]
        pred_label=[]
        for sample in tqdm(val_loader):
            image,mask,edge=sample.values()
            
            image=image.cuda()
            mask=mask.cuda()
            edge=edge.cuda()
            if check_model_forward_args(model)==2:
                prob = model(image,edge)
            else:
                prob = model(image)
            probs.extend(prob.detach().cpu().flatten().numpy().tolist())
            pred_mask = torch.where(prob>0.5,1,0).cpu().flatten()
            truth_label.extend(mask.flatten().tolist())
            pred_label.extend(pred_mask.detach().numpy().tolist())
        truth_label=torch.tensor(truth_label).cuda()
        pred_label= torch.tensor(pred_label).cuda()
        probs = torch.tensor(probs).cuda()
        return Accuracy(task='binary').cuda()(pred_label,truth_label).item(),BinaryF1Score().cuda()(pred_label,truth_label).item(),\
            JaccardIndex(task='binary').cuda()(pred_label,truth_label).item(),Recall(task='binary').cuda()(pred_label,truth_label).item(),\
            Specificity(task='binary').cuda()(pred_label,truth_label).item(),\
            AUROC(task='binary').cuda()(probs,truth_label).item(),\
            DiceScore(num_classes=2).cuda()(pred_label,truth_label).item()
            