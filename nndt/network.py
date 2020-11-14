import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import StandardScaler


class _TorchifiedDataset(Dataset):
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def dataset_to_loaders(dataset, tr, ts, device, **kwargs):
    if device != 'cpu':
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    torch_dataset = _TorchifiedDataset(dataset.data, dataset.target)
    tr_loader = DataLoader(torch_dataset,
                           sampler=SubsetRandomSampler(tr),
                           **kwargs)
    ts_loader = DataLoader(torch_dataset,
                           sampler=SubsetRandomSampler(ts),
                           **kwargs)

    return tr_loader, ts_loader


class FeedForwardNetwork(nn.Module):
    """A feedforward neural net with a single hidden layer."""
    def __init__(self, dataset, n_hidden):
        super(FeedForwardNetwork, self).__init__()

        n_inputs = dataset.data.shape[1]
        n_targets = len(dataset.target_names)

        self.linear1 = nn.Linear(n_inputs, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_targets)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


def train(net, device, loader, optimizer, sl, alpha):
    net.train()
    for b, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logprobs = net(data)
        # TODO extract relevance
        label_loss = F.nll_loss(logprobs, target)
        semantic_loss = sl.loss(data, logprobs)
        loss = alpha * label_loss + (1 - alpha) * semantic_loss
        loss.backward()
        optimizer.step()


def test(net, device, loader, sl):
    net.eval()
    label_loss, semantic_loss, n_correct, n = 0, 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logprobs = net(data)
            label_loss += F.nll_loss(logprobs, target, reduction='sum').item()
            # TODO extract relevance
            semantic_loss += sl.loss(data, logprobs)
            pred = logprobs.argmax(dim=1, keepdim=True)
            n_correct += pred.eq(target.view_as(pred)).sum().item()
            n += len(data)
    return label_loss / n, semantic_loss / n, n_correct / n
