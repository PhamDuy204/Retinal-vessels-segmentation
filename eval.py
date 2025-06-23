import torch
from utils import check_model_forward_args
from torchmetrics.classification import Accuracy,BinaryF1Score,\
                                        AUROC, Recall, Specificity,\
                                        JaccardIndex
from tqdm import tqdm
import kornia
def eval_for_seg(model,val_loader,gpu_id,patch=False):
    torch.cuda.set_device(gpu_id)
    with torch.no_grad():
        model.eval()
        probs= []
        truth_label=[]
        pred_label=[]
        for sample in tqdm(val_loader):
            if patch==False:
                image,mask,edge=sample.values()
                crop_points=None
            else:
                image,mask,edge,crop_points=sample.values()
            
            image=image.cuda()
            mask=mask.cuda()
            edge=edge.cuda()
            if check_model_forward_args(model)==2:
                prob = model(image,edge)
            else:
                prob = model(image)
            if crop_points is not None:
                crop_points=crop_points.cuda()
                h,w = mask.shape[-2:]
                prob=kornia.geometry.transform.crop_and_resize(prob, crop_points, size=(h, w))
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
            