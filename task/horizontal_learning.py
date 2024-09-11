import logging
from typing import Any, Dict, Iterable, List, Tuple

import delta.dataset
import numpy as np
import torch
from delta.delta_node import DeltaNode
from delta.task.learning import FaultTolerantFedAvg, HorizontalLearning
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset


def default_transform_data(data: List[Tuple[Image, str]]):
    """
    english edtion:
    A collate_fn function for DataLoader, used for preprocessing.
    Resize the input image, normalize it, and return it as a torch.Tensor.
    """
    xs, ys = [], []
    for x, y in data:
        xs.append(np.array(x).reshape((1, 28, 28)))
        ys.append(int(y))

    imgs = torch.tensor(xs)
    label = torch.tensor(ys)
    imgs = imgs / 255 - 0.5
    return imgs, label


class LeNet(torch.nn.Module):
    # Example LeNet model, you can replace it with your own model
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, padding=2)
        self.pool1 = torch.nn.AvgPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5)
        self.pool2 = torch.nn.AvgPool2d(2, stride=2)
        self.dense1 = torch.nn.Linear(400, 100)
        self.dense2 = torch.nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 400)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


class HorizontalTask(HorizontalLearning):
    def __init__(self, **kwargs) -> None:
        if "dataset" in kwargs:
            self.dataset_name = kwargs["dataset"]
            del kwargs["dataset"]
        else:
            print("Using default dataset: mnist")
            self.dataset_name = "mnist"
        self.model = None
        self.loss_func = None
        self.optimizer = None
        for k, v in kwargs.items():
            if k == "model":
                self.model = v
            elif k == "loss_func":
                self.loss_func = v
            elif k == "optimizer":
                self.optimizer = v
        if self.model is None:
            self.model = LeNet()
        if self.loss_func is None:
            self.loss_func = torch.nn.CrossEntropyLoss()
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        super_kwargs = {
            "name": "HorizontalTask",
            "max_rounds": 2,
            "validate_interval": 1,
            "validate_frac": 0.1,
            "strategy": FaultTolerantFedAvg(
                min_clients=kwargs.get("min_clients", 2),
                max_clients=kwargs.get("max_clients", 3),
                merge_epoch=kwargs.get("merge_epoch", 1),
                wait_timeout=kwargs.get("wait_timeout", 60),
                connection_timeout=kwargs.get("connection_timeout", 10),
            ),
        }
        for k, v in super_kwargs.items():
            if k in kwargs:
                super_kwargs[k] = kwargs[k]
        super().__init__(**super_kwargs)

    def dataset(self) -> delta.dataset.Dataset:
        return delta.dataset.Dataset(dataset=self.dataset_name)

    def make_train_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            collate_fn=default_transform_data,
        )

    def make_validate_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            drop_last=False,
            collate_fn=default_transform_data,
        )

    def train(self, dataloader: Iterable):
        """
        Training step, update the model with the training data.
        dataloader: The dataloader of the training set.
        return: None
        """
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader: Iterable) -> Dict[str, Any]:
        """
        Validation step, output the validation metrics.
        dataloader: The dataloader of the validation set.
        return: Dict[str, float], a dictionary, the key is the name of the metric (str), the value is the corresponding metric value (float)
        """
        total_loss = 0
        count = 0
        ys = []
        y_s = []
        for batch in dataloader:
            x, y = batch
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            total_loss += loss.item()
            count += 1

            y_ = torch.argmax(y_pred, dim=1)
            y_s.extend(y_.tolist())
            ys.extend(y.tolist())
        avg_loss = total_loss / count
        tp = len([1 for i in range(len(ys)) if ys[i] == y_s[i]])
        precision = tp / len(ys)

        return {"loss": avg_loss, "precision": precision}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        english:
        Return the model parameters that need to be trained and updated.
        When aggregating and saving the results, only the parameters returned by get_params will be updated and saved.
        return: List[torch.Tensor], a list of model parameters
        """
        return self.model.state_dict()
