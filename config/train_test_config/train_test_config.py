import numpy as np
from config.eval_config.eval import evaluate
import torch
import os
from PIL import Image

def print_train_loss_TA(train_loss_sup_1, train_loss_sup_2, train_loss_cps, train_loss, num_batches, print_num, print_num_half):
    train_epoch_loss_sup1 = train_loss_sup_1 / num_batches['train_sup']
    train_epoch_loss_sup2 = train_loss_sup_2 / num_batches['train_sup']
    train_epoch_loss_cps = train_loss_cps / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Sup Loss 1: {:.4f}'.format(train_epoch_loss_sup1).ljust(print_num_half, ' '), '| Train SUP Loss 2: {:.4f}'.format(train_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_cps).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup1, train_epoch_loss_sup2, train_epoch_loss_cps, train_epoch_loss

def print_val_loss(val_loss_sup_1, val_loss_sup_2, num_batches, print_num, print_num_half):
    val_epoch_loss_sup1 = val_loss_sup_1 / num_batches['val']
    val_epoch_loss_sup2 = val_loss_sup_2 / num_batches['val']
    print('-' * print_num)
    print('| Val Sup Loss 1: {:.4f}'.format(val_epoch_loss_sup1).ljust(print_num_half, ' '), '| Val Sup Loss 2: {:.4f}'.format(val_epoch_loss_sup2).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_sup1, val_epoch_loss_sup2

def print_train_eval(pred_list_train1, pred_list_train2, mask_list_train, print_num):
    eval_list1 = evaluate(pred_list_train1, mask_list_train)
    eval_list2 = evaluate(pred_list_train2, mask_list_train)
    print('| Train Px-Jc 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Train Px-Jc 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
    print('| Train Px-Dc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Train Px-Dc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
    print('| Train BetaE 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Train BetaE 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
    return eval_list1, eval_list2


def print_val_eval(pred_list_val1, pred_list_val2, mask_list_val, print_num):

    eval_list1 = evaluate(pred_list_val1, mask_list_val)
    eval_list2 = evaluate(pred_list_val2, mask_list_val)
    print('| Val Px-Jc 1: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Px-Jc 2: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
    print('| Val Px-Dc 1: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val Px-Dc 2: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
    print('| Val BetaE 1: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val BetaE 2: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
    return eval_list1, eval_list2


def save_val_best_TA(best_model, best_list, best_result, model1, model2, pred_list_val_1, pred_list_val_2, name_list_val, eval_list_1, eval_list_2, path_trained_model, path_seg_results, palette):

    if eval_list_1[2] > eval_list_2[2]:
        if best_list[2] > eval_list_2[2]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'

            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_BetaE_{:.4f}.pth'.format('Result2', best_list[2])))
            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_BetaE_{:.4f}_other.pth'.format('Result2', best_list[2])))

            assert len(name_list_val) == pred_list_val_2.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_list_val_2[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[2] > eval_list_1[2]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'

            torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_BetaE_{:.4f}.pth'.format('Result1', best_list[2])))
            torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_BetaE_{:.4f}_other.pth'.format('Result1', best_list[2])))

            assert len(name_list_val) == pred_list_val_1.shape[0]
            for i in range(len(name_list_val)):
                color_results = Image.fromarray(pred_list_val_1[i].astype(np.uint8), mode='P')
                color_results.putpalette(palette)
                color_results.save(os.path.join(path_seg_results, name_list_val[i]))
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    return best_list, best_model, best_result

def draw_pred_TA(mask_train, mask_val, pred_train1, pred_train2, pred_val1, pred_val2):
    mask_image_train = mask_train[0, :, :]
    mask_image_val = mask_val[0, :, :]

    pred_image_train1 = pred_train1[0, :, :]
    pred_image_train2 = pred_train2[0, :, :]
    pred_image_val1 = pred_val1[0, :, :]
    pred_image_val2 = pred_val2[0, :, :]

    return mask_image_train, pred_image_train1, pred_image_train2, mask_image_val, pred_image_val1, pred_image_val2

def print_best(best_val_list, best_model, best_result, path_trained_model, print_num):
    torch.save(best_model.state_dict(), os.path.join(path_trained_model, 'best_BetaE_{:.4f}.pth'.format(best_val_list[2])))

    print('| Best Val Model: {}'.format(best_result).ljust(print_num, ' '), '|')
    print('| Best Val Px-Jc: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
    print('| Best Val Px-Dc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
    print('| Best Val BetaE: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')

def print_test_eval(pred_list_test, mask_list_test, print_num):
    eval_list = evaluate(pred_list_test, mask_list_test)
    print('| Test Px-Jc: {:.4f}'.format(eval_list[0]).ljust(print_num, ' '), '|')
    print('| Test Px-Dc: {:.4f}'.format(eval_list[1]).ljust(print_num, ' '), '|')
    print('| Test BetaE: {:.4f}'.format(eval_list[2]).ljust(print_num, ' '), '|')
    return eval_list

def save_test(pred_list_test, name_list_test, path_seg_results, palette):

    assert len(name_list_test) == pred_list_test.shape[0]

    for i in range(len(name_list_test)):
        color_results = Image.fromarray(pred_list_test[i].astype(np.uint8), mode='P')
        color_results.putpalette(palette)
        color_results.save(os.path.join(path_seg_results, name_list_test[i]))



