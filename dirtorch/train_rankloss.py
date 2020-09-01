import os
import torch
import numpy as np
import tensorwatch as tw
import dirtorch.nets as nets
from datetime import datetime
import torch.nn.functional as F
from dirtorch.loss import APLoss
from dirtorch.utils import common
from torch.autograd import Variable
import dirtorch.datasets as datasets
from dirtorch.utils.common import tonumpy, pool
from dirtorch.utils.pytorch_loader import get_loader
from dirtorch.test_dir import eval_model,extract_image_features,expand_descriptors
from tensorboardX import SummaryWriter
model_options = {'arch': 'resnet50_fpn_rmac', 'out_dim': 2048, 'pooling': 'gem', 'gemp': 3}
class ClassLoss(torch.nn.Module):

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    def forward(self, yc, label):
        return self.loss(yc, label)

def time_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def train_model(db,test_db, net, criterion, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
                aqe=None, adba=None, threads=8,batch_size=10, epochs=100, start_epoch=0, dbg=(),optimizer=None,
                scheduler_mul=None,output='',writer=None):
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Training...")
    # same_size = 'Pad' in trfs or 'Crop' in trfs
    # if not same_size:
    torch.backends.cudnn.benchmark = True
    loader = get_loader(db, trf_chain=trfs, preprocess=net.preprocess, iscuda=net.iscuda,output=['img'], batch_size=batch_size, threads=threads, shuffle=False,size=[800,800])
    for i_,epoch in enumerate(range(start_epoch,epochs)):
        net.cuda().train()
        epoch_loss,cls_losses = 0.0,0.0
        for index,inputs in enumerate(loader):
            imgs,targets = inputs[0][0],inputs[1]
            desc_db = []
            for i, img in enumerate(imgs):
                desc_db.append(net(img.unsqueeze(dim=0).cuda()).detach().unsqueeze(dim=0))
            desc_db = torch.cat(desc_db,dim=0)
            desc_db.requires_grad = True
            scores = torch.matmul(desc_db,desc_db.t())
            batch_label = torch.tensor(np.array([int(db.get_label(j)) for j in range(imgs.shape[0])]))
            Y = torch.stack([idx == batch_label for idx in batch_label],dim=0).to(torch.int).cuda()
            rank_loss = criterion(scores, Y)
            rank_loss.backward()
            optimizer.zero_grad()
            # print('\nscores',scores.cpu().detach().numpy()[:5,:5],'\ndesc_db',desc_db.cpu().detach().numpy()[:5,:5],'\ndesc_db.grad',desc_db.grad.cpu().detach().numpy()[:5,:5])
            for i,img in enumerate(imgs):
                img = Variable(img.cuda(),requires_grad=True)
                desc = net(img.unsqueeze(dim=0))
                one_grad = desc_db.grad[i].unsqueeze(0)
                desc.unsqueeze(0).backward(one_grad)
            optimizer.step()
            scheduler_mul.step()
            lr = scheduler_mul.get_lr()[0]
            epoch_loss+=(rank_loss.item()+cls_losses/len(desc_db))
        print(time_now() + '\n[Train Phase][Epoch: %3d/%3d][Batch:%3d][Loss: %3.5f][lr:%.5f]' % (epoch + 1, epochs,index+1, epoch_loss / len(loader), lr))
        writer.add_scalars(os.path.join(output,'log','Train_val_loss') , {output+ 'train_loss': epoch_loss / len(loader)},epoch+1)

        fconv = open(os.path.join(output,'log', 'convergence.csv'), 'a')
        if (epoch + 1) % 5 == 0:
            obj = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, '%smodel_epoch%d.pt' % (output, (epoch + 1)))
            torch.cuda.empty_cache()
            # eval_model
            res = eval_model(test_db, net, trfs, pooling=pooling, gemp=gemp, detailed=detailed,
                             threads=threads, dbg=dbg, whiten=whiten, aqe=aqe, adba=adba,
                             save_feats='./experiments/feats/', load_feats=None,epoch=epoch)
            print(' * ' + '\n * '.join(['%s = %g' % p for p in res.items()]))
            fconv.write('{},{},{},{}\n'.format(epoch+1,lr, epoch_loss / len(loader),res))
            writer.add_scalars(os.path.join(output, 'log', 'top3MAP'),
                               {output + 'top3MAP': res['top3']}, epoch)

        else:
            fconv.write('{},{},{}\n'.format(epoch+1, lr, epoch_loss / len(loader)))

        fconv.close()


def load_model(path=None, iscuda=''):
    checkpoint = common.load_checkpoint('./pretrained_model/Resnet50-AP-GeM.pt', iscuda)
    net = nets.create_model(pretrained="", **model_options)#**checkpoint['model_options']
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    if path:
        checkpoint_2 = common.load_checkpoint(path, iscuda)
        checkpoint_state_dict = checkpoint_2['state_dict']
        start_epoch = checkpoint_2['epoch']
    else:
        checkpoint_state_dict = checkpoint['state_dict']
        start_epoch = 0
    # net.load_state_dict(checkpoint_state_dict,False)
    load_param(net,checkpoint_state_dict)
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')

    return net,start_epoch

def load_param(model,checkpoint_state_dict):
    model_pretrained = checkpoint_state_dict
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_pretrained.items() if (k in model_dict) and (v.shape==model_dict[k].shape)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--test_dataset', type=str, required=True, help='Command to load dataset')

    parser.add_argument('--checkpoint', type=str, default=None, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='ToTensor(),Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', default='0,1', nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default='Landmarks_clean', help='applies whitening')

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')


    parser.add_argument('--batch_size', type=int, default=320, help='size of batch imags')
    parser.add_argument('--epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--saved', type=str, default='./experiments/ICIAR+no_pretrained/', help='train epochs')

    args = parser.parse_args()
    gpu_ids = args.gpu[0].split(',')
    gpu_ids = [int(i) for i in gpu_ids]
    args.iscuda = common.torch_set_gpu(gpu_ids)
    if args.aqe is not None:
        args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None:
        args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

    #load dataset
    dataset = datasets.create(args.dataset)
    test_dataset = datasets.create(args.test_dataset)

    #load model
    print("With %s Train Model:" %(args.dataset))
    net, start_epoch = load_model(args.checkpoint, args.iscuda)
    net.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.nn.DataParallel(net,device_ids = [0])
    net.to(device)

    #define optimizer parameter
    net.pca = net.module.pca
    net.preprocess = net.module.preprocess
    net.iscuda = net.module.iscuda
    criterion = APLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=5*1e-3, weight_decay=1e-4)
    scheduler_mul = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(np.arange(int(start_epoch)+50, int(args.epochs), 100)),0.9)
    writer = SummaryWriter(os.path.join(args.saved,'log'))
    #whiten and pca
    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    #record
    fconv = open(os.path.join(args.saved,'log','convergence.csv'), 'w')
    fconv.write('epoch,lr,loss,MAP\n')
    fconv.close()

    # Train
    kw = dict(pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
              threads=args.threads, dbg=args.dbg, whiten=args.whiten, aqe=args.aqe,
              adba=args.adba,batch_size=args.batch_size,epochs=args.epochs,
              start_epoch=start_epoch,optimizer=optimizer,scheduler_mul=scheduler_mul,output=args.saved,writer=writer)
    train_model(dataset,test_dataset, net,criterion, args.trfs, **kw)
    writer.close()
