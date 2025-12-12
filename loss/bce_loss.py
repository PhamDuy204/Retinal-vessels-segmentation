class BceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, truth):
        pred = pred.squeeze(1).float()
        truth = truth.squeeze(1).float().detach()
        return self.bce(pred, truth)

    def forward(self, preds, truth):

        if not isinstance(preds, tuple):
            preds = (preds,)

        # detach target
        truth = truth.detach()

        # compute multi-scale loss
        if len(preds) > 1:
            losses = []
            for p in preds:
                losses.append(self.compute_loss(p, truth))
            return sum(losses) / len(losses)
        else:
            return self.compute_loss(preds[0], truth)
