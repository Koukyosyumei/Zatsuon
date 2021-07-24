import torch
from torch.utils.data import DataLoader


class DenoisedAutoEncoder:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        sampling_rate=16e3,
        batch_size=1,
        epoch=1,
        log_interval=1,
        device=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch
        self.log_interval = log_interval
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        else:
            self._set_device()

    def train(self, train_dataset, val_dataset):
        train_log = []
        eval_log = []

        train_datasize = train_dataset.tensors[0].shape[0]
        val_datasize = val_dataset.tensors[0].shape[0]

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            eval_loss = 0.0

            for data in train_dataloader:
                x_batch, y_batch = data
                self.optimizer.zero_grad()
                output = self.model(
                    x_batch.to(self.device).reshape(-1, 1, self.sampling_rate)
                )
                output = output.reshape(-1, self.sampling_rate)
                loss = self.criterion(output, y_batch.to(self.device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() / (train_datasize / self.batch_size)
            train_log.append(train_loss)

            for data in val_dataloader:
                x_batch, y_batch = data
                with torch.no_grad():
                    output = self.model(
                        x_batch.to(self.device).reshape(-1, 1, self.sampling_rate)
                    )
                    output = output.reshape(-1, self.sampling_rate)
                    loss = self.criterion(output, y_batch.to(self.device))
                    eval_loss += loss.item() / (val_datasize / self.batch_size)
            eval_log.append(eval_loss)

            if epoch % self.log_interval == 0:
                print(epoch, train_loss, eval_loss)

    def denoise(self, path_to_wav, path_to_output=None):
        pass

    def load_model(self, path_to_state_dict):
        pass

    def save_model(self, path_to_state_dict):
        torch.save(self.model.to("cpu").state_dict(), path_to_state_dict)

    def _set_device(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
