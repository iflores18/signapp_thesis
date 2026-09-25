"""Training loop."""


def train_one_epoch(model, loader, optimizer, criterion, device):
    raise NotImplementedError


def train(model, train_loader, val_loader, config, device):
    raise NotImplementedError
