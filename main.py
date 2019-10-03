import parser
import argparse
# import argparser

import torch
import torch.nn
import torch.optim
import torchvision.models as models
from Models.resnet import ResNet as resnet
from Models.vggnet import VGGNet as vggnet
from utils import prepare_dataloaders
'''
    reference:
        torchvision
        conda install -c conda-forge torchvision
'''
def run_epoch(model, mode, criterion, optimizer, data_loader):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    for i, (data, target) in enumerate(data_loader):
        # prepare data
        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # forward
        output = model(data)

        loss = criterion(output, target)
        if mode == 'train':
            #  backward
            loss.backward()
            optimizer.step()
            acc1, acc5 = accuracy(output, target, topk=(1,5))

            print(' - ({}) iter {}/{} loss {:.4f} top1 {:.4f} top5 {:.4f}'.format(mode, i+1, len(data_loader), loss, acc1[0], acc5[0]))
            # print(' - ({}) iter {}/{} loss {:.4f}'.format(mode, i+1, len(data_loader), loss))
        else:
            pred = output.data.max(1)[1]
            acc1, acc5 = accuracy(output, target, topk=(1,5))
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            print(' - ({}) iter {}/{} loss {:.4f} top1 {:.4f} top5 {:.4f}'.format(mode, i+1, len(data_loader), loss, acc1[0], acc5[0]))
    return acc1
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser(description='Implement image classification on ImageNet datset using pytorch')
    parser.add_argument('--model', default='resnet50', type=str, help='classification model (resnet101, resnet50, vggnet')
    parser.add_argument('--n_epochs', default=1000, type=int, help='numeber of total epochs to run')
    parser.add_argument('--batch', default=64, type=int, help='mini batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--save_directory', default='trained.chkpt', type=str, help='path to latest checkpoint')
    parser.add_argument('--workers', default=8, type=int, help='num_workers')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    args = parser.parse_args()
    # use gpu or multi-gpu or not.

    use_gpu = torch.cuda.is_available()
    if use_gpu :
        use_multi_gpu = torch.cuda.device_count() > 1
    print('[Info] use_gpu:{} use_multi_gpu:{}'.format(use_gpu, use_multi_gpu))

    # load the data.
    print('[Info] Load the data.')
    train_loader, valid_loader = prepare_dataloaders(args)

    # load the model.
    print('[Info] Load the model.')
    if args.model.find('resnet') != -1:
        model = resnet(args, 10)
    elif args.model.find('vggnet') != -1:
        model = vggnet(args, 10)
    print(count_parameters(model))
    # print('torchvision ', count_parameters(models.resnet50()))
    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # define loss function.
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.resume:
        # Load the checkpoint.
        print('[Info] Resuming from checkpoint.')
        checkpoint = torch.load('ckpt.pth')
    # run epoch.

    best_acc = 0.
    for epoch in range(1, args.n_epochs + 1):
        _ = run_epoch(model, 'train', criterion, optimizer, train_loader)
        with torch.no_grad():
            acc1 = run_epoch(model, 'valid', criterion, optimizer, valid_loader)
        # Save checkpoint.
        if acc > best_acc:
            print('[Info] Save the model.')
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
            res.append(correct_k.mul_(100.0 / bsz))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
	main()
