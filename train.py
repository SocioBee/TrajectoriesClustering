import losses
import constants

from data_utils import DataOrderScaner, load_label
from cluster import update_cluster
from metrics import nmi_score, ami_score, ari_score, fms_score, cluster_acc, cluster_purity
from models import DTC
import umap
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def save_checkpoint(state, args):
    torch.save(state, args.checkpoint)
    shutil.copyfile(args.checkpoint, os.path.join(args.model, 'best_model.pt'))


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    devices = [torch.device("cuda:" + str(i)) for i in range(4)]
    for i in range(len(devices)):
        devices[i] = devices[args.cuda]
    loss_cuda = devices[args.cuda]

    # define criterion, model, optimizer
    dtc = DTC(args, devices)

    init_parameters(dtc)
    dtc.pretrain()

    optimizer = torch.optim.Adam(
        dtc.parameters(), lr=args.learning_rate)

    V, D = losses.load_dis_matrix(args)
    V, D = V.to(loss_cuda), D.to(loss_cuda)

    def rclossF(o, t):
        return losses.KLDIVloss(o, t, V, D, loss_cuda)

    start_epoch = 0
    iteration = 0
    # load model state and optmizer state
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        iteration = checkpoint["iteration"] + 1
        dtc.load_state_dict(checkpoint["dtc"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("=> No checkpoint found at '{}'".format(args.checkpoint))

    # Training
    print("=> Reading trajecoty data...")
    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data
    y = load_label(args.label_file)
    y_pred_last = np.zeros_like(y)

    print("=> Epoch starts at {} "
          "and will end at {}".format(start_epoch, args.epoch-1))

    best_loss = [-1, -1, -1, 0]

    for epoch in range(start_epoch, args.epoch):
        contexts = []
        # update target distribution p
        if epoch % args.update_interval == 0:
            with torch.no_grad():
                # q (datasize,n_clusters)
                vecs, tmp_q, p = update_cluster(
                    dtc, args, devices[0], devices[2])

            # evaluate clustering performance
            y_pred = tmp_q.numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc, per_class_acc,_ = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)

            if best_loss[0] < acc:
                best_loss[0] = acc
                best_loss[1] = nmi
                best_loss[2] = ari
                best_loss[3] = epoch
            else:
                if epoch - best_loss[3] > 5:
                    break

            if epoch > 0 and delta_label < args.tolerance:
                print('Delta_label {:.4f} < tolerance {:.4f}'.format(
                    delta_label, args.tolerance))
                print('=> Reached tolerance threshold. Stopping training.')
                break
            else:
                print('Epoch {0}\tAcc: {1:.4f}\tnmi: {2:.4f}\tari: {3:.4f}'.format(
                    epoch, acc, nmi, ari))
                print(per_class_acc)

        scaner.reload()
        while True:
            optimizer.zero_grad()
            gendata = scaner.getbatch(invp=False)
            if gendata is None:
                break


            reconstr_loss, context = losses.reconstructionLoss(
                gendata, dtc.autoencoder, dtc.rclayer, rclossF, args, devices[0], devices[1], loss_cuda)

            # quit()
            # (batch_size,n_clusters)
            p_select = p[scaner.start - args.batch:scaner.start]

            # KL Div Loss
            kl_loss,q = losses.clusteringLoss(
                dtc.clusterlayer, context, p_select, devices[2], loss_cuda)

            # reducer = umap.UMAP()
            # data_2d_umap = reducer.fit_transform(p_select.detach().cpu().numpy())
            # plt.figure(figsize=(10, 8))
            # plt.scatter(data_2d_umap[:, 0], data_2d_umap[:, 1], c=y[:256], cmap='viridis', s=1) # s is the size of points
            # plt.colorbar()
            # plt.title('Ours')
            # plt.show()
            # quit()

            # OT Loss
            ot_loss = losses.optimalTransport(dtc.clusterlayer,context,p_select,devices[2], loss_cuda)

            # Self-Supervised GE2E Loss
            class_labels = torch.argmax(q, dim=1)
            sorted_indices = torch.argsort(class_labels)
            sorted_features = context[sorted_indices]
            sorted_class_labels = class_labels[sorted_indices]
            unique_class_labels, counts_per_class = torch.unique(sorted_class_labels, return_counts=True)
            ge2e_loss = dtc.ge2elayer(sorted_features,counts_per_class)
            # amsoftmax_loss = dtc.amsftmax_layer(sorted_features,counts_per_class)
            #Triplet Loss
            anchor, positive, negative = scaner.getbatch_discriminative()
            tri_loss = losses.triLoss(
                anchor, positive, negative, dtc.autoencoder, loss_cuda)
            loss = reconstr_loss + 10*ot_loss[1] + 10*ge2e_loss + kl_loss

            # compute the gradients
            loss.backward()
            # clip the gradients
            clip_grad_norm_(dtc.parameters(), args.max_grad_norm)
            # one step optimization
            optimizer.step()

            # average loss for one word
            if iteration % args.print_freq == 0:
                print("Iteration: {0:}\tLoss: {1:.3f}\t"
                      "Rc Loss: {2:.3f}\tClustering Loss: {3:.3f}\Triplet Loss: {4:.4f}"
                      .format(iteration, loss, reconstr_loss, kl_loss.item(), tri_loss.item()))

            if iteration % args.save_freq == 0 and iteration > 0:
                # print("Saving the model at iteration {}  loss {}"
                #       .format(iteration, loss))
                save_checkpoint({
                    "iteration": iteration,
                    "best_loss": loss,
                    "dtc": dtc.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, args)

            iteration += 1
    with torch.no_grad():
        # q (datasize,n_clusters)
        vecs, tmp_q, p = update_cluster(
            dtc, args, devices[0], devices[2])
    print(vecs.shape)
    # evaluate clustering performance
    y_pred = tmp_q.numpy().argmax(1)
    delta_label = np.sum(y_pred != y_pred_last).astype(
        np.float32) / y_pred.shape[0]
    y_pred_last = y_pred


    print(p.shape, y.shape)
    data_np = vecs.detach().cpu().numpy()  # Convert to NumPy array if it's not already

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_np)
    # reducer = PCA(n_components = 6)
    # data_2d_umap = reducer.fit_transform(data_np)
    reducer = TSNE()
    data_2d_umap = reducer.fit_transform(data_scaled)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(y)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    # Now, we plot each label with its own color
    for i, label in enumerate(unique_labels):
        plt.scatter(data_2d_umap[y == label, 0], data_2d_umap[y == label, 1],
                    color=colors(i), label=label, s=5)

    # Adding the legend to map colors to labels
    # plt.legend(title="Labels")

    plt.show()
    # plt.colorbar()
    # plt.title('Ours')
    # plt.show()
    quit()
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)

    if best_loss[0] < acc:
        best_loss[0] = acc
        best_loss[1] = nmi
        best_loss[2] = ari
        best_loss[3] = epoch+1

    if epoch > 0 and delta_label < args.tolerance:
        print('Delta_label {:.4f} < tolerance {:.4f}'.format(
            delta_label, args.tolerance))
        print('=> Reached tolerance threshold. Stopping training.')
    else:
        print('Epoch {0}\tAcc: {1:.4f}\tnmi: {2:.4f}\tari: {3:.4f}'.format(
            epoch, acc, nmi, ari))
    print("=================")
    print('Best Epoch {0}\tAcc: {1:.4f}\tnmi: {2:.4f}\tari: {3:.4f}'.format(
        best_loss[3], best_loss[0], best_loss[1], best_loss[2]))
