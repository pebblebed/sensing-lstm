import torch
from torch.autograd import Variable
import torch.nn as nn
import lightning as L

class LSTMAutoEncoder(nn.Module):
  def __init__(self, input_size=4, offset=32, hidden=1024, output_size=4):
    super().__init__()
    self.input_size = input_size
    self.offset = 1
    self.hidden_size = hidden
    self.encoder = nn.LSTM(input_size, hidden, batch_first=True).to(self.device())
    self.label = nn.Linear(hidden, output_size).to(self.device())

  def device(self):
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def forward(self, x):
    (b, t, d) = x.shape
    h_0 = Variable(torch.zeros(b, 1, self.hidden_size)).to(self.device())
    out, _ = self.encoder(x)
    assert out.shape == (b, t, self.hidden_size)
    out = self.label(out)
    assert out.shape == x.shape
    return out

class LightningAutoEncode(L.LightningModule):
  def __init__(self, input_size=4, offset=32, hidden=1024, output_size=4):
    super().__init__()
    self.model = LSTMAutoEncoder(input_size, offset, hidden, output_size)
    self.offset = offset

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters())

  def forward(self, x):
    return self.model(x)

  def criterion(self, y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

  def training_step(self, batch, batch_idx):
    x, y = batch
    assert x.shape == y.shape
    y_hat = self(x)
    assert y_hat.shape == y.shape
    loss = self.criterion(y_hat, y)
    self.log('train_loss', loss, on_epoch=True, prog_bar=True)
    return loss
  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.criterion(y_hat, y)
    self.log('test_loss', loss, on_epoch=True, prog_bar=True)
    return loss
