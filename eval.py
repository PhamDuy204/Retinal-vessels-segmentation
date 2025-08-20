import torch
from utils import *
from torchmetrics.classification import Accuracy,BinaryF1Score,\
                                        AUROC, Recall, Specificity,\
                                        JaccardIndex
from torchmetrics.segmentation import DiceScore
from tqdm import tqdm
# from timm.models.maxxvit import window_partition,window_reverse
# import kornia

def eval_for_seg(model, val_loader, gpu_id, patch=False,patch_size=64,type_split='random'):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()

    acc_metric    = Accuracy(task='binary').cuda()
    f1_metric     = BinaryF1Score().cuda()
    jaccard_metric= JaccardIndex(task='binary').cuda()
    recall_metric = Recall(task='binary').cuda()
    spec_metric   = Specificity(task='binary').cuda()
    auroc_metric  = AUROC(task='binary').cuda()
    dice_metric  = DiceScore(num_classes=2, average='macro').cuda()

    with torch.inference_mode():
        for sample in tqdm(val_loader):
            out_sample=[]
            model.eval()
            image, mask, edge = sample.values()
            image=mirror_padding_v2(image)
            edge=mirror_padding_v2(edge)
            B,C,H,W = image.shape
            image = image.cuda()
            mask  = mask.cuda()
            edge  = edge.cuda()
            stride=None
            if patch and type_split!='random':
                # condition=int(H>W)
                num_patch=(64,64)
                image,tmp_stride = extract_patches_with_target_count(image,patch_size,num_patch)
                edge,_ = extract_patches_with_target_count(edge,patch_size,num_patch)
                stride=tmp_stride
                if len(image.shape)>4:
                    image=image.flatten(0,1)
                    edge=edge.flatten(0,1)
            #   edge = kornia.contrib.extract_tensor_patches(edge, patch_size, patch_size//4).flatten(0,1)

                # image = window_partition(image.permute(0,2,3,1).contiguous(),[patch_size,patch_size]).permute(0,3,1,2).contiguous()
                # edge = window_partition(edge.permute(0,2,3,1).contiguous(),[patch_size,patch_size]).permute(0,3,1,2).contiguous()
            # chunk_size = max(image.shape[0]//500,1)
            # chunk_image = torch.chunk(image,chunk_size,0)
            # chunk_edge = torch.chunk(edge,chunk_size,0)
            # for chunk in zip(chunk_image,chunk_edge):
            #     c_image,c_edge=chunk
            # print(image.shape)
            if check_model_forward_args(model) == 2:
                prob = model(image, edge)
            else:
                prob = model(image)
                # out_sample.append(prob)
            # prob= torch.cat(out_sample,0)
            if patch:
                # prob = kornia.contrib.combine_tensor_patches(prob.view(B,-1,1,patch_size,patch_size), original_size=(H,W),window_size=patch_size,stride=patch_size//4)
                # prob=window_reverse(prob.permute(0,2,3,1).contiguous(),[patch_size,patch_size],[H,W]).permute(0,3,1,2).contiguous()
                # if type_split=='random':
                #     h, w = mask.shape[-2:]
                #     prob = prob[:,:,:h,:w]
                if stride is not None:
                    prob = prob.view(B,-1,1,patch_size,patch_size)
                    prob=reverse_to_original_image(prob,(H,W),patch_size,stride)
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
            dice_metric.update(pred_mask.unsqueeze(0).unsqueeze(0).long(), mask.unsqueeze(0).unsqueeze(0).long())
            torch.cuda.empty_cache()


    return (
        acc_metric.compute().item(),
        f1_metric.compute().item(),
        jaccard_metric.compute().item(),
        recall_metric.compute().item(),
        spec_metric.compute().item(),
        auroc_metric.compute().item(),
        dice_metric.compute().item(),
    )