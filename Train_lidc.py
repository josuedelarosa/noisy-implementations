from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import dataloader_lidc as dataloader
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score


parser = argparse.ArgumentParser(description='PyTorch DivideMix LIDC Training')

# keep original args for compatibility / logging
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio (unused for LIDC unless you inject noise)')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', required=True, type=str, help='path to folder containing .npy patches')
parser.add_argument('--dataset', default='lidc', type=str)
parser.add_argument('--hw', default=32, type=int, help='final patch size (H=W)')
parser.add_argument('--test_ratio', default=0.2, type=float, help='holdout ratio for test split')
parser.add_argument('--split_seed', default=123, type=int, help='seed for train/test split')

# LIDC-specific
parser.add_argument('--csv', required=True, type=str, help='CSV with patient_id,case_number,nodule_number,malignancy')
parser.add_argument('--drop_m3', action='store_true', help='drop malignancy == 3 samples')

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (
                torch.softmax(outputs_u11, dim=1)
                + torch.softmax(outputs_u12, dim=1)
                + torch.softmax(outputs_u21, dim=1)
                + torch.softmax(outputs_u22, dim=1)
            ) / 4
            ptu = pu ** (1 / args.T)  # temperature sharpening
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temperature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[: batch_size * 2]
        logits_u = logits[batch_size * 2 :]

        Lx, Lu, lamb = criterion(
            logits_x,
            mixed_target[: batch_size * 2],
            logits_u,
            mixed_target[batch_size * 2 :],
            epoch + batch_idx / num_iter,
            warm_up,
        )

        # regularization
        prior = torch.ones(args.num_class) / args.num_class
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                Lx.item(),
                Lu.item(),
            )
        )
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader_):
    net.train()
    num_iter = (len(dataloader_.dataset) // dataloader_.batch_size) + 1

    # NOTE: for LIDC warmup loader returns (inputs, labels, index)
    for batch_idx, (inputs, labels, index) in enumerate(dataloader_):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)

        if args.noise_mode == 'asym':
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode == 'sym':
            L = loss

        L.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
            % (
                args.dataset,
                args.r,
                args.noise_mode,
                epoch,
                args.num_epochs,
                batch_idx + 1,
                num_iter,
                loss.item(),
            )
        )
        sys.stdout.flush()


from sklearn.metrics import roc_auc_score, confusion_matrix

def test(epoch, net1, net2):
    net1.eval()
    net2.eval()

    all_probs = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = net1(inputs) + net2(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_preds.append(preds.detach().cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy().astype(int)
    y_pred = torch.cat(all_preds).numpy().astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    eps = 1e-12
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)     # recall, TPR
    specificity = tn / (tn + fp + eps)     # TNR
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(y_true, y_prob)

    print(
        f"\n| Test Epoch #{epoch}"
        f"\n| Accuracy: {accuracy*100:.2f}%"
        f"  Sensitivity: {sensitivity:.4f}"
        f"  Specificity: {specificity:.4f}"
        f"  F1: {f1:.4f}"
        f"  AUC: {auc:.4f}\n"
    )

    test_log.write(
        f"Epoch:{epoch} Acc:{accuracy:.4f} Sens:{sensitivity:.4f} "
        f"Spec:{specificity:.4f} F1:{f1:.4f} AUC:{auc:.6f}\n"
    )
    test_log.flush()

    return auc



def eval_train(model, all_loss):
    model.eval()

    # LIDC: allocate by actual dataset length (do NOT assume 50000)
    n = len(eval_loader.dataset)
    losses = torch.zeros(n)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b].detach().cpu()

    # normalize to [0,1]
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    all_loss.append(losses)

    # original logic kept
    if args.r == 0.9:
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    # IMPORTANT: grayscale input
    model = ResNet18(num_classes=args.num_class, in_channels=1)
    model = model.cuda()
    return model


# logs (keep original naming style)
os.makedirs('./checkpoint', exist_ok=True)
stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

best_auc = -1

# warmup schedule
if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30
elif args.dataset == 'lidc':
    warm_up = 50
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")


# LIDC loader (same run() modes as cifar_dataloader)
loader = dataloader.lidc_dataloader(
    csv_file=args.csv,
    root_dir=args.data_path,
    batch_size=args.batch_size,
    num_workers=5,
    log=stats_log,
    drop_m3=args.drop_m3,
    hw=args.hw,
    test_ratio=args.test_ratio,
    split_seed=args.split_seed,
)


print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode == 'asym':
    conf_penalty = NegEntropy()

all_loss = [[], []]  # save the history of losses from two networks

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')

    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)

    else:
        prob1, all_loss[0] = eval_train(net1, all_loss[0])
        prob2, all_loss[1] = eval_train(net2, all_loss[1])

        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

    test(epoch, net1, net2)
