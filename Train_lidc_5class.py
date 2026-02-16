# Train_lidc_5class.py
from __future__ import print_function
import sys
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from PreResNet import ResNet18
import dataloader_lidc as dataloader


parser = argparse.ArgumentParser(description='PyTorch DivideMix LIDC (5-class) Training')

parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)

parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--seed', default=123, type=int)

parser.add_argument('--dataset', default='lidc', type=str)
parser.add_argument('--num_class', default=5, type=int)

parser.add_argument('--csv', required=True, type=str, help='CSV with patient_id, case_number, nodule_number, malignancy')
parser.add_argument('--data_path', required=True, type=str, help='folder containing .npy patches')
parser.add_argument('--test_ratio', default=0.2, type=float)
parser.add_argument('--split_seed', default=123, type=int)
parser.add_argument('--hw', default=32, type=int, help='resize to hw x hw')

parser.add_argument('--num_workers', default=0, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.makedirs("./checkpoint", exist_ok=True)

stats_log = open('./checkpoint/lidc5_stats.txt', 'w')
test_log = open('./checkpoint/lidc5_metrics.txt', 'w')


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


def create_model():
    model = ResNet18(num_classes=args.num_class, in_channels=1).cuda()
    return model


def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)

        labels_x_oh = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).float()

        inputs_x, inputs_x2 = inputs_x.cuda(), inputs_x2.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
        labels_x_oh, w_x = labels_x_oh.cuda(), w_x.cuda()

        with torch.no_grad():
            # co-guessing for unlabeled
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) +
                  torch.softmax(outputs_u12, dim=1) +
                  torch.softmax(outputs_u21, dim=1) +
                  torch.softmax(outputs_u22, dim=1)) / 4.0

            ptu = pu ** (1.0 / args.T)
            targets_u = (ptu / ptu.sum(dim=1, keepdim=True)).detach()

            # label refinement for labeled
            outputs_x1 = net(inputs_x)
            outputs_x2 = net(inputs_x2)
            px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2.0
            px = w_x * labels_x_oh + (1 - w_x) * px
            ptx = px ** (1.0 / args.T)
            targets_x = (ptx / ptx.sum(dim=1, keepdim=True)).detach()

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0), device=all_inputs.device)
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = criterion(
            logits_x, mixed_target[:batch_size * 2],
            logits_u, mixed_target[batch_size * 2:],
            epoch + batch_idx / num_iter,
            warm_up
        )

        # regularization term (same as official)
        prior = (torch.ones(args.num_class, device=logits.device) / args.num_class)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            f'{args.dataset} | Epoch [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter[{batch_idx+1:3d}/{num_iter:3d}]\t '
            f'Labeled loss: {Lx.item():.2f}  Unlabeled loss: {Lu.item():.2f}'
        )
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader_):
    net.train()
    num_iter = (len(dataloader_.dataset) // dataloader_.batch_size) + 1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader_):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            f'{args.dataset} | Epoch [{epoch:3d}/{args.num_epochs:3d}] '
            f'Iter[{batch_idx+1:3d}/{num_iter:3d}]\t CE-loss: {loss.item():.4f}'
        )
        sys.stdout.flush()


def eval_train(model, all_loss):
    model.eval()
    n = len(eval_loader.dataset)
    losses = torch.zeros(n)

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = CE(outputs, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-12)
    all_loss.append(losses)

    input_loss = losses.reshape(-1, 1).cpu().numpy()

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]  # probability of clean component
    return prob, all_loss


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()

    all_probs = []
    all_targets = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net1(inputs) + net2(inputs)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            probs = torch.softmax(outputs, dim=1)  # NxC
            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    acc = 100.0 * correct / total

    y_prob = torch.cat(all_probs).numpy()           # N x 5
    y_true = torch.cat(all_targets).numpy().astype(int)  # N

    # Macro ROC-AUC (OVR). Needs at least 2 classes present in test.
    if len(np.unique(y_true)) < 2:
        auc = float('nan')
    else:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auc = float('nan')

    print(f"\n| Test Epoch #{epoch}\t Accuracy: {acc:.2f}%\t AUC(macro-OVR): {auc:.4f}\n")
    test_log.write(f"Epoch:{epoch} Acc:{acc:.4f} AUC:{auc:.6f}\n")
    test_log.flush()
    return auc


# Warmup length (you can tune; CIFAR10 uses 10, CIFAR100 uses 30)
warm_up = 30

loader = dataloader.lidc_dataloader(
    csv_path=args.csv,
    data_path=args.data_path,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    test_ratio=args.test_ratio,
    split_seed=args.split_seed,
    hw=args.hw,
    num_class=args.num_class,
    log=stats_log,
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

all_loss = [[], []]

best_auc = -1.0
best_epoch = -1

for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for pg in optimizer1.param_groups:
        pg['lr'] = lr
    for pg in optimizer2.param_groups:
        pg['lr'] = lr

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
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)

    auc = test(epoch, net1, net2)

    # Save best by AUC
    if (not np.isnan(auc)) and (auc > best_auc):
        best_auc = auc
        best_epoch = epoch
        print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch} — saving checkpoint")
        torch.save(
            {
                "epoch": epoch,
                "best_auc": float(best_auc),
                "net1": net1.state_dict(),
                "net2": net2.state_dict(),
                "args": vars(args),
            },
            "./checkpoint/best_dividemix_lidc5.pth",
        )
