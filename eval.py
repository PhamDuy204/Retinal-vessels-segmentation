import torch
from utils import check_model_forward_args
from torchmetrics.classification import Accuracy,BinaryF1Score,\
                                        AUROC, Recall, Specificity,\
                                        JaccardIndex
from tqdm import tqdm
import kornia
def eval_for_seg(model, val_loader, gpu_id, patch=False):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    model.eval()

    acc_metric    = Accuracy(task='binary').cuda()
    f1_metric     = BinaryF1Score().cuda()
    jaccard_metric= JaccardIndex(task='binary').cuda()
    recall_metric = Recall(task='binary').cuda()
    spec_metric   = Specificity(task='binary').cuda()
    auroc_metric  = AUROC(task='binary').cuda()

    with torch.inference_mode():
        for sample in tqdm(val_loader):
            

            image, mask, edge = sample.values()
            image = image.cuda()
            mask  = mask.cuda()
            edge  = edge.cuda()

            if check_model_forward_args(model) == 2:
                prob = model(image, edge)
            else:
                prob = model(image)

            if patch :
                h, w = mask.shape[-2:]
                prob = prob[:,:,:h,:w]


            prob= prob.squeeze().detach().cuda().flatten()
            mask = mask.squeeze().detach().cuda().flatten()
            # print(mask.dtype)
            # print(prob.dtype)

            pred_mask = torch.where(prob>0.5,1,0)

            acc_metric.update(pred_mask, mask)
            f1_metric.update(pred_mask, mask)
            jaccard_metric.update(pred_mask, mask)
            recall_metric.update(pred_mask, mask)
            spec_metric.update(pred_mask, mask)
            auroc_metric.update(prob, mask)
            torch.cuda.empty_cache()


    return (
        acc_metric.compute().item(),
        f1_metric.compute().item(),
        jaccard_metric.compute().item(),
        recall_metric.compute().item(),
        spec_metric.compute().item(),
        auroc_metric.compute().item(),
    )