import torch
import torch.utils.data

from acclib.models.utils import process_data_train


def get_train_sample(name, trend, window_size, step, correction, storage, frequency):
    acc, gyro, trajectory = storage[name, 'acc'], storage[name, 'gyro'], storage[name, 'trajectory']
    x_t, v_t = process_data_train(acc, gyro, trajectory, trend, window_size, step, correction, frequency)
    return x_t, v_t


def get_x_from_names(names, trend, window_size, step, correction, storage, frequency):
    res = []
    for name in names:
        res.append(get_train_sample(name, trend, window_size, step, correction, storage, frequency))
    res = list(zip(*res))
    x = torch.cat(list(res[0]))
    v = torch.cat(list(res[1]))
    return x, v


def get_loader(x, y, batch_size=128):
    data = torch.utils.data.TensorDataset(x, y)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader


def get_loaders(train_ind, valid_ind, trend, window_size, step, train_correction, val_correction,
                storage, frequency, batch_size=128):
    x_train, v_train = get_x_from_names(train_ind, trend, window_size, step, train_correction,
                                        storage, frequency)
    x_valid, v_valid = get_x_from_names(valid_ind, trend, window_size, step, val_correction,
                                        storage, frequency)
    train_loader = get_loader(x_train, v_train[:, :2], batch_size=batch_size)
    valid_loader = get_loader(x_valid, v_valid[:, :2], batch_size=batch_size)
    return train_loader, valid_loader


def train_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    losses = []
    for x, y in train_loader:
        out = model(x.to(device))
        loss = criterion(out, y.to(device))
        losses.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(losses) / len(losses)


def evaluate_loss(loader, model, criterion, device):
    with torch.no_grad():
        model.eval()
        losses = []
        for X, y in loader:
            out = model(X.to(device))
            losses.append(criterion(out, y.to(device)).cpu().numpy())
    return sum(losses) / len(losses)


def train(model, opt, train_loader, test_loader, criterion, n_epochs, device, writer=False,
          verbose=True, scheduler=False):
    train_losses = []
    valid_losses = []
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, train_loader, criterion, device)
        train_losses.append(train_loss)
        valid_loss = evaluate_loss(test_loader, model, criterion=criterion, device=device)
        valid_losses.append(valid_loss)
        if writer:
            writer.add_scalars('loss', {'validation': valid_loss, 'train': train_loss},
                               global_step=epoch)
        if verbose:
            print('Epoch [%d/%d], Loss (train/test): %.6f/%.6f,' % (
                epoch + 1, n_epochs, train_loss, valid_loss))
        if scheduler:
            scheduler.step()
    return train_losses, valid_losses
