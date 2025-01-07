from visdom import Visdom
import os

def visdom_initialization_TA(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Train Loss', 'Train Sup1', 'Train Sup2', 'Train Unsup'], width=400, height=300))
    visdom.line([[0., 0.]], [0.], win='train_jc', opts=dict(title='Train Px-Jc', xlabel='Epoch', ylabel='Train Px-Jc', legend=['Train Px-Jc1', 'Train Px-Jc2'], width=400, height=300))
    visdom.line([[0., 0.]], [0.], win='train_be', opts=dict(title='Train BetaE', xlabel='Epoch', ylabel='Train BetaE', legend=['Train BetaE1', 'Train BetaE2'], width=400, height=300))
    visdom.line([[0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['Val Sup1', 'Val Sup2'], width=400, height=300))
    visdom.line([[0., 0.]], [0.], win='val_jc', opts=dict(title='Val Px-Jc', xlabel='Epoch', ylabel='Val Px-Jc', legend=['Val Px-Jc1', 'Val Px-Jc2'], width=400, height=300))
    visdom.line([[0., 0.]], [0.], win='val_be', opts=dict(title='Val BetaE', xlabel='Epoch', ylabel='Val BetaE', legend=['Val BetaE1', 'Val BetaE2'], width=400, height=300))
    return visdom
def visualization_TA(vis, epoch, train_loss, train_loss_sup1, train_loss_sup2, train_loss_cps, train_jc1, train_jc2, train_be1, train_be2, val_loss_sup1, val_loss_sup2, val_jc1, val_jc2, val_be1, val_be2):
    vis.line([[train_loss, train_loss_sup1, train_loss_sup2, train_loss_cps]], [epoch], win='train_loss', update='append')
    vis.line([[train_jc1, train_jc2]], [epoch], win='train_jc', update='append')
    vis.line([[train_be1, train_be2]], [epoch], win='train_be', update='append')
    vis.line([[val_loss_sup1, val_loss_sup2]], [epoch], win='val_loss', update='append')
    vis.line([[val_jc1, val_jc2]], [epoch], win='val_jc', update='append')
    vis.line([[val_be1, val_be2]], [epoch], win='val_be', update='append')
def visual_image_TA(vis, mask_train, pred_train1, pred_train2, mask_val, pred_val1, pred_val2):
    vis.heatmap(mask_train, win='train_mask', opts=dict(title='Train Mask', colormap='Viridis', width=400, height=300))
    vis.heatmap(pred_train1, win='train_pred1', opts=dict(title='Train Pred1', colormap='Viridis', width=400, height=300))
    vis.heatmap(pred_train2, win='train_pred2', opts=dict(title='Train pred2', colormap='Viridis', width=400, height=300))
    vis.heatmap(mask_val, win='val_mask', opts=dict(title='Val Mask', colormap='Viridis', width=400, height=300))
    vis.heatmap(pred_val1, win='val_pred1', opts=dict(title='Val Pred1', colormap='Viridis', width=400, height=300))
    vis.heatmap(pred_val2, win='val_pred2', opts=dict(title='Val Pred2', colormap='Viridis', width=400, height=300))

