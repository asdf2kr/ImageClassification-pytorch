import parser
import argparse

from tqdm import tqdm
# import argparser

import torch
import torch.nn
import torch.optim
import torchvision.models as models
import Models.resnet as resnet
import Models.vggnet as vggnet
from utils import prepare_dataloaders
'''
    reference:
        torchvision
        conda install -c conda-forge torchvision
'''
def run_epoch(model, mode, epoch, criterion, optimizer, data_loader, dataset_size):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    total_loss = 0.
    total_correct = 0
    for data, target in tqdm(data_loader, desc='  - (' + mode + ')   ', leave=False):
        # prepare data
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # forward
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        total_correct += torch.sum(output.data.max(1)[1] == target.data)

        acc1, acc5 = accuracy(output, target, topk=(1,5))
        if mode == 'train':
            # backward
            loss.backward()
            optimizer.step()
    print(' - ({}) epoch {} loss {:.4f} acc {:.4f} top1 {:.4f} top5 {:.4f}'.format(mode, epoch, total_loss / dataset_size, total_correct / dataset_size * 100., acc1 / dataset_size * 100. , acc5 / dataset_size * 100.))
    return acc1 / dataset_size * 100.

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='Implement image classification on ImageNet datset using pytorch')
    parser.add_argument('--model', default='resnet50', type=str, help='classification model (resnet(18, 34, 50, 101, 152), vggnet16')
    parser.add_argument('--n_epochs', default=1000, type=int, help='numeber of total epochs to run')
    parser.add_argument('--batch', default=64, type=int, help='mini batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--save_directory', default='trained.chkpt', type=str, help='path to latest checkpoint')
    parser.add_argument('--workers', default=8, type=int, help='num_workers')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    parser.add_argument('--datasets', default='CIFAR10', type=str, help='classification dataset  (CIFAR10, ImageNet)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--save', default='Datas/trained.chkpt', type=str, help='Datas/trained.chkpt')
    parser.add_argument('--save_multi', default='Datas/trained_multi.chkpt', type=str, help='Datas/trained_multi.chkpt')
    args = parser.parse_args()
    args.model = args.model.lower()
    args.datasets = args.datasets.lower()

    # use gpu or multi-gpu or not.
    use_gpu = torch.cuda.is_available()
    if use_gpu :
        use_multi_gpu = torch.cuda.device_count() > 1
    print('[Info] use_gpu:{} use_multi_gpu:{}'.format(use_gpu, use_multi_gpu))

    # load the data.
    print('[Info] Load the data.')
    train_loader, valid_loader, train_size, valid_size = prepare_dataloaders(args)

    # load the model.
    print('[Info] Load the model.')

    if args.datasets == 'cifar10':
        num_classes = 10
    elif args.datasets == 'imagenet':
        num_classes = 1000

    if args.model == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes)
    elif args.model == 'resnet34':
        model = resnet.resnet34(num_classes=num_classes)
    elif args.model == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes)
    elif args.model == 'resnet101':
        model = resnet.resnet101(num_classes=num_classes)
    elif args.model == 'resnet152':
        model = resnet.resnet152(num_classes=num_classes)
    elif args.model == 'vggnet16':
        model = vggnet.vggnet16(num_classes=num_classes)

    # print(count_parameters(model))
    # print('torchvision ', count_parameters(models.resnet50()))

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # define loss function.
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        # Load the checkpoint.
        print('[Info] Resuming from checkpoint.')
        checkpoint = torch.load('ckpt.pth')
    # run epoch.

    best_acc = 0.
    for epoch in range(1, args.n_epochs + 1):
        _ = run_epoch(model, 'train', epoch, criterion, optimizer, train_loader, train_size)
        with torch.no_grad():
            acc1 = run_epoch(model, 'valid', epoch, criterion, optimizer, valid_loader, valid_size)

        # Save checkpoint.
        if acc1 > best_acc:
            best_acc = acc1
            checkpoint = {
                'model': model.state_dict(),
                'epoch': epoch,
                'cnn': args.model}
            torch.save(checkpoint, args.save)

            if torch.cuda.device_count() > 1:
                checkpoint_module = {
                    'model': model.module.state_dict(),
                    'epoch': epoch,
                    'cnn': args.model}
                torch.save(checkpoint_module, args.save_multi)
            print(' - [Info] The checkpoint file has been updated.')

        print('[Info] acc1 {} best {}'.format(acc1, best_acc))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        bsz = target.size(0)
        '''
            https://pytorch.org/docs/stable/torch.html#torch.topk
            torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
        '''
        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
	main()
