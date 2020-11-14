import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

import nndt


def load_synthetic_dataset(n_examples=1000):
    X = np.random.randint(0, 2, size=(n_examples, 4))
    y = X[:, 0]

    dataset = Bunch()
    dataset.data = X.astype(np.float32)
    dataset.target = y
    dataset.feature_names = [f'input{i}' for i in range(X.shape[1])]
    dataset.target_names = ['neg', 'pos']

    return dataset



DATASETS = {
    'synthetic': load_synthetic_dataset,
}


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    group = parser.add_argument_group('Data')
    group.add_argument('dataset', choices=sorted(DATASETS.keys()),
                       help='dataset to be used')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    group = parser.add_argument_group('Semantic Loss')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='trade-off between losses')

    group = parser.add_argument_group('Neural Net')
    parser.add_argument('--n-epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Load the data
    dataset = DATASETS[args.dataset]()
    indices = list(range(dataset.data.shape[0]))
    tr, ts = train_test_split(indices, test_size=0.2)
    tr_loader, ts_loader = nndt.dataset_to_loaders(dataset, tr, ts, device,
                                                   batch_size=args.batch_size)

    # Build the semantic loss
    sl = nndt.DecisionTreeLoss(dataset).fit(dataset.data[tr], dataset.target[tr])
    sl.sync()

    # Build the neural net
    n_inputs = dataset.data.shape[1]
    net = nndt.FeedForwardNetwork(dataset, n_inputs).to(device)

    # Evaluate the NN+DT combo
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.n_epochs + 1):
        nndt.train(net, device, tr_loader, optimizer, sl, args.alpha)
        label_loss, distillation_loss, n_correct = nndt.test(net, device, ts_loader, sl)
        print(f'{epoch} : ll={label_loss:5.3f} dl={distillation_loss:5.3f} acc={n_correct}')
        scheduler.step()


if __name__ == '__main__':
    main()
