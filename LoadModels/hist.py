import matplotlib.pyplot as plt
import numpy as np
import torch
import pdb
# type = "adv"
# gt = np.load("adv_teacher_gt_60k.npy")
# gt = np.array(gt)
#
# pred_clean = np.load("clean_teacher_wrn_pred_60k.npy".format(type))
# pred_clean = np.array(pred_clean)
# pred_adv = np.load("adv_teacher_pred_60k.npy".format(type))
# pred_adv = np.array(pred_adv)
gt_tensor = torch.load('download/adv_gt_ViT-B_32_tau2.pt')
pred_adv_tensor = torch.load('download/adv_pred_ViT-B_32_tau2.pt')
clean_adv_tensor = torch.load('download/adv_pred_ViT-B_32.pt')
pred_clean = clean_adv_tensor.numpy()
pred_adv = pred_adv_tensor.numpy()
gt = gt_tensor.numpy()
# for j in range(10):
#     pred_adv_temp = pred_adv[gt==j]
#     pred_clean_temp = pred_clean[gt == j]
#     plt.hist(pred_adv_temp[:,j], bins=30, alpha=0.5, label='Adv prediction')
#     plt.hist(pred_clean_temp[:,j], bins=30, alpha=0.5, label='clean prediction')
#     plt.legend(loc='upper right')
#     plt.savefig('vit_adv_clean/vit_adv_clean_pred_comp_gt_{}.png'.format(j))
#     plt.close()
    # exit()
# pdb.set_trace()
for j in range(10):
    pred_adv_temp = pred_adv[gt==j]
    pred_clean_temp = pred_clean[gt == j]
    # Generate some random data
    # pdb.set_trace()
    # fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    # axs = axs.flatten()
    # for i in range(10):
    for i in range(10):
        plt.hist(pred_adv_temp[:,i], bins=30, alpha=0.5, label='Adv prediction')
        plt.hist(pred_clean_temp[:,i], bins=30, alpha=0.5, label='clean prediction')
        plt.legend(loc='upper right')
        plt.savefig('vit_b_32/adv_clean_pred_comp_gt_{}_pred_{}.png'.format(j,i))
        plt.close()




    # for i in range(10):
    #     axs[i].hist(pred_temp[:,i], bins=20, color='blue')
    #     axs[i].set_title('Prediction {}'.format(i))
    #     axs[i].set_xlabel('softlabel')
    #     axs[i].set_ylabel('Frequency')
    #     axs[i].set_xlim([0, 1])
    #     axs[i].set_ylim([0, 1100])
        # plt.hist(pred_temp[:,i],bins=30, alpha=0.5, label='Pred Class {}'.format(i))
    # Plot a histogram
    # print(pred_temp.shape)
    # plt.hist(pred_temp, bins=10)
    # plt.title("current_label_{}".format(i))
    # plt.xlabel("softlabel")
    # plt.ylabel("Frequency")
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.savefig('adv_clean_comp/adv_clean_pred_comp_gt_{}_pred_{}.png'.format(j,i))
    # plt.subplots_adjust(wspace=0.5,hspace=0.8)
    # plt.show()

    # exit()
# all_prediction(numpy.array(gt_)==1)