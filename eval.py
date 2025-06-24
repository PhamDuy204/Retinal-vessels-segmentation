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

    # Khởi tạo metrics chạy trên CPU
    acc_metric    = Accuracy(task='binary').to('cpu')
    f1_metric     = BinaryF1Score().to('cpu')
    jaccard_metric= JaccardIndex(task='binary').to('cpu')
    recall_metric = Recall(task='binary').to('cpu')
    spec_metric   = Specificity(task='binary').to('cpu')
    auroc_metric  = AUROC(task='binary').to('cpu')

    with torch.inference_mode():
        for sample in tqdm(val_loader):
            # unpack
            if not patch:
                image, mask, edge = sample.values()
                crop_pts = None
            else:
                image, mask, edge, crop_pts = sample.values()


            image = image.to(gpu_id, non_blocking=True)
            mask  = mask.to(gpu_id, non_blocking=True)
            edge  = edge.to(gpu_id, non_blocking=True)

            # forward
            if check_model_forward_args(model) == 2:
                prob = model(image, edge)
            else:
                prob = model(image)

            # crop nếu cần
            if patch and crop_pts is not None:
                h, w = mask.shape[-2:]
                prob = kornia.geometry.transform.crop_and_resize(
                    prob, crop_pts.to(gpu_id, non_blocking=True), size=(h, w)
                )


            prob_cpu = prob.squeeze().detach().cpu()
            mask_cpu = mask.squeeze().detach().cpu()


            pred_cpu = (prob_cpu > 0.5).long()

            acc_metric.update(pred_cpu, mask_cpu)
            f1_metric.update(pred_cpu, mask_cpu)
            jaccard_metric.update(pred_cpu, mask_cpu)
            recall_metric.update(pred_cpu, mask_cpu)
            spec_metric.update(pred_cpu, mask_cpu)
            auroc_metric.update(prob_cpu, mask_cpu)

            del image, mask, edge, prob
            torch.cuda.empty_cache()


    return (
        acc_metric.compute().item(),
        f1_metric.compute().item(),
        jaccard_metric.compute().item(),
        recall_metric.compute().item(),
        spec_metric.compute().item(),
        auroc_metric.compute().item(),
    )