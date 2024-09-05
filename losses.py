import torch
import torch.nn as nn
import h5py
import os
import constants
import torch.nn.functional as F
from sklearn.cluster import DBSCAN,KMeans
import numpy as np
from SinkhornDistance import loss_func


class GE2ELoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, loss_method='softmax'):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(init_b, dtype=torch.float32))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']
        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        elif self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, counts_per_class):
        # Calculate centroids based on counts_per_class
        centroids = []
        start_idx = 0
        for count in counts_per_class:
            end_idx = start_idx + count
            class_dvecs = dvecs[start_idx:end_idx]
            centroid = class_dvecs.mean(dim=0)
            centroids.append(centroid)
            start_idx = end_idx
        centroids = torch.stack(centroids)

        # Calculate cosine similarity between each dvec and each centroid
        expanded_dvecs = dvecs.unsqueeze(1).expand(-1, centroids.size(0), -1)
        expanded_centroids = centroids.unsqueeze(0).expand(dvecs.size(0), -1, -1)
        cos_sim_matrix = F.cosine_similarity(expanded_dvecs, expanded_centroids, dim=2)

        return cos_sim_matrix

    def embed_loss_softmax(self, cos_sim_matrix, counts_per_class):
        labels = self.generate_labels(counts_per_class).to(cos_sim_matrix.device)
        loss = -torch.log_softmax(cos_sim_matrix, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)
        return loss.mean()

    def generate_labels(self, counts_per_class):
        # Ensuring this is correctly recognized as a method
        labels = []
        for class_idx, count in enumerate(counts_per_class):
            labels += [class_idx] * count
        return torch.tensor(labels, device=counts_per_class.device)

    def embed_loss_contrast(self, cos_sim_matrix, counts_per_class):
        labels = self.generate_labels(counts_per_class).to(cos_sim_matrix.device)
        num_classes = cos_sim_matrix.size(1)
        true_sim = cos_sim_matrix.gather(1, labels.unsqueeze(1)).squeeze(1)
        max_other = cos_sim_matrix.scatter(1, labels.unsqueeze(1), -float('inf')).max(1)[0]
        loss = F.relu(1 + max_other - true_sim)  # Margin set to 1 for simplicity
        return loss.mean()

    def forward(self, dvecs, counts_per_class):
        # Calculate centroids and cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, counts_per_class)

        # Scale the cosine similarity matrix
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        # Calculate loss
        loss = self.embed_loss(cos_sim_matrix, counts_per_class)

        return loss


def KLDIVloss(output, target, V, D, loss_cuda):
    """
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    # (batch, k) index in vocab_size dimension
    # k-nearest neighbors for target
    indices = torch.index_select(V, 0, target)
    # (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)
    # (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)
    # KLDIVcriterion
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(outputk, targetk)


def dist2weight(D, dist_decay_speed=0.8):
    '''
    D is a matrix recording distances between each vocab and its k nearest vocabs
    D(k, vocab_size)
    weight: \frac{\exp{-|dis|*scale}}{\sum{\exp{-|dis|*scale}}}
    Divide 100
    '''
    D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    # The PAD should not contribute to the decoding loss
    D[constants.PAD, :] = 0.0
    return D


def load_dis_matrix(args):
    assert os.path.isfile(args.knearestvocabs),\
        "{} does not exist".format(args.knearestvocabs)
    with h5py.File(args.knearestvocabs, 'r') as f:
        V, D = f["V"], f["D"]
        V, D = torch.LongTensor(V), torch.FloatTensor(D)
    D = dist2weight(D, args.dist_decay_speed)
    return V, D


def clusterloss(q, p, loss_cuda):
    '''
    caculate the KL loss for clustering
    '''
    q, p = q.to(loss_cuda), p.to(loss_cuda)
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(q.log(), p)


def reconstructionLoss(gendata,
                       autoencoder,
                       rclayer,
                       lossF,
                       args,
                       cuda0,
                       cuda1,
                       loss_cuda):
    """
    One batch reconstruction loss
    cuda0 for autoencoder
    cuda1 for rclayer
    loss_cuda for reconstruction loss

    Input:
    gendata: a named tuple contains
        gendata.src (seq_len1, batch): input tensor
        gendata.lengths (1, batch): lengths of source sequences
        gendata.trg (seq_len2, batch): target tensor.
    autoencoder: map input to output.
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    context (cuda0)
    """
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg
    input = input.to(cuda0)
    lengths = lengths.to(cuda0)
    target = target.to(cuda0)

    # print("input size:", input.size())
    # print("lengths size:", lengths.size())
    # print("target size:", target.size())
    # Encoder & decoder
    # output (trg_seq_len, batch, hidden_size)
    # context (batch, hidden_size * num_directions)
    output, context = autoencoder(input, lengths, target)
    batch = output.size(1)
    loss = 0
    # we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    # generate words from autoencoder output
    for o, t in zip(output.split(args.gen_batch),
                    target.split(args.gen_batch)):
        # (seq_len, gen_batch, hidden_size) =>
        ## (seq_len*gen_batch, hidden_size)
        o = o.view(-1, o.size(2)).to(cuda1)
        # print("o size:", o.size())
        o = rclayer(o)
        # (seq_len*gen_batch,)
        t = t.view(-1)
        o, t = o.to(loss_cuda), t.to(loss_cuda)
        loss += lossF(o, t)

    return loss.div(batch), context


def clusteringLoss(clusterlayer, context, p, cuda2, loss_cuda):
    """
    One batch cluster KL loss

    Input:
    context: (batch, hidden_size * num_directions) last hidden layer from encoder
    clusterlayer: caculate Student’s t-distribution with clustering center

    p: (batch_size,n_clusters)target distribution

    Output:loss
    """
    batch = context.size(0)
    assert batch == p.size(0)
    q = clusterlayer(context.to(cuda2))
    kl_loss = clusterloss(q, p, loss_cuda)

    return kl_loss.div(batch),q

def optimalTransport(clusterlayer, context, p, cuda2, loss_cuda):
    batch = context.size(0)
    assert batch == p.size(0)
    q = clusterlayer(context.to(cuda2))
    loss = loss_func(q[:batch//2],q[batch//2:],p[:batch//2].cuda(),p[batch//2:].cuda())
    return loss

def triLoss(a, p, n, autoencoder, loss_cuda):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """

    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp

    a_src, a_lengths, a_invp = a_src.to(
        loss_cuda), a_lengths.to(loss_cuda), a_invp.to(loss_cuda)
    p_src, p_lengths, p_invp = p_src.to(
        loss_cuda), p_lengths.to(loss_cuda), p_invp.to(loss_cuda)
    n_src, n_lengths, n_invp = n_src.to(
        loss_cuda), n_lengths.to(loss_cuda), n_invp.to(loss_cuda)

    a_context = autoencoder.encoder_hn(a_src, a_lengths)
    p_context = autoencoder.encoder_hn(p_src, p_lengths)
    n_context = autoencoder.encoder_hn(n_src, n_lengths)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(loss_cuda)

    return triplet_loss(a_context[a_invp], p_context[p_invp], n_context[n_invp])

def cosineSimilarityLoss(a, p, n, autoencoder, loss_cuda):
    # Move data to the specified device
    a_src, a_lengths = a.src.to(loss_cuda), a.lengths.to(loss_cuda)
    p_src, p_lengths = p.src.to(loss_cuda), p.lengths.to(loss_cuda)
    n_src, n_lengths = n.src.to(loss_cuda), n.lengths.to(loss_cuda)

    # Encode the source sequences to get the context vectors
    a_context = autoencoder.encoder_hn(a_src, a_lengths)
    p_context = autoencoder.encoder_hn(p_src, p_lengths)
    n_context = autoencoder.encoder_hn(n_src, n_lengths)

    # combined_embeddings = torch.cat([a_context_normalized, p_context_normalized, n_context_normalized], dim=0)
    # combined_embeddings_np = a_context_normalized.cpu().detach().numpy()
    average_ap_embeddings = (a_context + p_context) / 2
    combined_embeddings = torch.cat([average_ap_embeddings, n_context], dim=0).detach().cpu().numpy()

    # Perform KMeans clustering
    n_clusters = 12  # as you mentioned knowing there are 12 unique clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(combined_embeddings)

    # Extract cluster labels
    labels = kmeans.labels_

    num_samples = a_context.size(0)

    # Labels for the averaged anchor-positive embeddings and negatives
    ap_labels = labels[:num_samples]
    n_labels = labels[num_samples:]
    a_labels = ap_labels
    p_labels = ap_labels
    # Normalize the context vectors
    a_context_normalized = F.normalize(a_context, dim=1)
    p_context_normalized = F.normalize(p_context, dim=1)
    n_context_normalized = F.normalize(n_context, dim=1)

    # Compute the cosine similarities
    positive_similarity = (a_context_normalized * p_context_normalized).sum(dim=1)
    negative_similarity = (a_context_normalized * n_context_normalized).sum(dim=1)

    # Compute the loss, ensuring it is non-negative
    # You want to maximize positive_similarity which means minimizing (1 - positive_similarity)
    # and minimize negative_similarity which is already in the range [-1, 1]
    # The loss should be non-negative, thus use ReLU to ensure that
    loss = F.relu(1 - positive_similarity) + F.relu(negative_similarity)

    # Return the mean loss over the batch
    return loss.mean()

def contrastiveLoss(a, p, n, autoencoder, loss_cuda, margin):
    # Assume 'a' and 'p' form a positive pair, 'a' and 'n' form a negative pair
    a_src, a_lengths = a.src.to(loss_cuda), a.lengths.to(loss_cuda)
    p_src, p_lengths = p.src.to(loss_cuda), p.lengths.to(loss_cuda)
    n_src, n_lengths = n.src.to(loss_cuda), n.lengths.to(loss_cuda)

    a_context = autoencoder.encoder_hn(a_src, a_lengths)
    p_context = autoencoder.encoder_hn(p_src, p_lengths)
    n_context = autoencoder.encoder_hn(n_src, n_lengths)
    positive_distance = (a_context - p_context).pow(2).sum(1)
    negative_distance = (a_context - n_context).pow(2).sum(1)

    losses = F.relu(positive_distance - negative_distance + margin)
    contrastive_loss = losses.mean()

    return contrastive_loss
