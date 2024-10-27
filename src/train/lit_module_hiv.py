# pylint: disable=arguments-differ,unused-argument
import torch
import lightning as L
from typing import Dict

from src.model import Model
from src.optim import OptimizerConfig, LRSchedulerConfig

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import Tensor

def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''
    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and one negative sample
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            rocauc_list.append(rocauc)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

class LitModuleHIV(L.LightningModule):
    def __init__(
            self,
            model: Model,
            optimizer_config: OptimizerConfig,
            lr_scheduler_config: LRSchedulerConfig,
        ):
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler = None

        self.criterion = self.criterion_function  # 使用自定义的损失函数

        # 初始化用于收集验证和测试阶段的预测和目标
        self.validation_preds = []
        self.validation_targets = []

        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = self.optimizer_config.setup(self.model)
        self.lr_scheduler = self.lr_scheduler_config.setup(optimizer)
        return optimizer

    def forward(self, batch):
        return self.model(batch)

    @torch._dynamo.disable  # 使用 @torch._dynamo.disable 禁用编译
    def log_training_metrics(self, loss, batch_size):
        self.log('training/loss', loss, batch_size=batch_size,
                 on_step=True, on_epoch=True, logger=True, rank_zero_only=True)
        self.log('training/lr', self.lr_scheduler.lr,
                 on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
        self.log('training/step', float(self.global_step),
                 on_step=True, on_epoch=False, logger=True, rank_zero_only=True)

    def training_step(self, batch, batch_idx):
        assert self.model.training
        y_hat, y = self.forward(batch)
        loss = self.criterion(y_hat, y)
        self.lr_scheduler.step(self.global_step)
        self.log_training_metrics(loss.detach(), y_hat.shape[0])
        return loss

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, batch):
        assert not self.model.training
        return self.forward(batch)

    @torch._dynamo.disable  # 禁用编译
    def log_validation_metrics(self, loss):
        self.log('validation/loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        # print("\ny_hat.shape", y_hat.shape)
        # print("\ny.shape", y.shape)
        loss = self.criterion(y_hat, y)

        # 获取预测概率并调整形状
        if y_hat.shape[1] == 2:
            preds = torch.softmax(y_hat, dim=-1)[:, 1].unsqueeze(1)  # 形状变为 [batch_size, 1]
        else:
            preds = torch.softmax(y_hat, dim=-1)  # 多分类保持形状 [batch_size, num_classes]
            # 将 y 转换为 one-hot 编码
            y = torch.nn.functional.one_hot(y.squeeze(1), num_classes=y_hat.shape[1]).float()

        # 收集预测和目标
        self.validation_preds.append(preds.detach().cpu())
        self.validation_targets.append(y.detach().cpu())

        # 记录损失
        self.log_validation_metrics(loss)

    @torch._dynamo.disable  # 禁用编译
    def log_test_metrics(self, loss):
        self.log('test/loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        loss = self.criterion(y_hat, y)

        # 获取预测概率并调整形状
        if y_hat.shape[1] == 2:
            preds = torch.softmax(y_hat, dim=-1)[:, 1].unsqueeze(1)  # 形状变为 [batch_size, 1]
        else:
            preds = torch.softmax(y_hat, dim=-1)  # 多分类保持形状 [batch_size, num_classes]
            # 将 y 转换为 one-hot 编码
            y = torch.nn.functional.one_hot(y.squeeze(1), num_classes=y_hat.shape[1]).float()

        # 收集预测和目标
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        # 记录损失
        self.log_test_metrics(loss)

    @torch._dynamo.disable  # 禁用编译
    def log_epoch_metrics(self, mode, roc_auc):
        self.log(f'{mode}/roc_auc', roc_auc, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        if len(self.validation_preds) == 0:
            self.log_epoch_metrics('validation', float('nan'))
            return

        # 将所有批次的预测和目标拼接起来
        all_preds = torch.cat(self.validation_preds).numpy()
        all_targets = torch.cat(self.validation_targets).numpy()

        # 计算 ROC AUC
        try:
            perf = eval_rocauc(all_targets, all_preds)
        except RuntimeError as e:
            perf = {'rocauc': float('nan')}
            self.print(str(e))

        # 记录 ROC AUC
        self.log_epoch_metrics('validation', perf['rocauc'])

        # 清空列表以准备下一个 epoch
        self.validation_preds.clear()
        self.validation_targets.clear()

    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            self.log_epoch_metrics('test', float('nan'))
            return

        # 将所有批次的预测和目标拼接起来
        all_preds = torch.cat(self.test_preds).numpy()
        all_targets = torch.cat(self.test_targets).numpy()

        # 计算 ROC AUC
        try:
            perf = eval_rocauc(all_targets, all_preds)
        except RuntimeError as e:
            perf = {'rocauc': float('nan')}
            self.print(str(e))

        # 记录 ROC AUC
        self.log_epoch_metrics('test', perf['rocauc'])

        # 清空列表以准备下一个 epoch
        self.test_preds.clear()
        self.test_targets.clear()

    @torch._dynamo.disable  # 禁用编译
    def log_predict_metrics(self, batch, y_hat, y):
        # 你可以在这里添加任何预测相关的日志记录
        pass

    def predict_step(self, batch, batch_idx):
        y_hat, y = self.inference(batch)
        self.predict_step_outputs.append((batch, y_hat, y))
        self.log_predict_metrics(batch, y_hat, y)

    def on_predict_epoch_end(self):
        self.predict_step_outputs.clear()

    def criterion_function(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y = y.view(-1).long()  # 确保 y 的形状为 [batch_size] 且类型为 long
        return torch.nn.functional.cross_entropy(y_hat, y)
# import torch
# import lightning as L
# from typing import Dict

# from src.model import Model
# from src.optim import OptimizerConfig, LRSchedulerConfig

# import numpy as np
# from sklearn.metrics import roc_auc_score
# from torch import Tensor

# def eval_rocauc(y_true, y_pred):
#     '''
#         compute ROC-AUC averaged across tasks
#     '''
#     rocauc_list = []

#     for i in range(y_true.shape[1]):
#         # AUC is only defined when there is at least one positive and one negative sample
#         if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#             # ignore nan values
#             is_labeled = y_true[:, i] == y_true[:, i]
#             rocauc = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
#             rocauc_list.append(rocauc)

#     if len(rocauc_list) == 0:
#         raise RuntimeError(
#             'No positively labeled data available. Cannot compute ROC-AUC.')

#     return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

# class LitModuleHIV(L.LightningModule):
#     def __init__(
#             self,
#             model: Model,
#             optimizer_config: OptimizerConfig,
#             lr_scheduler_config: LRSchedulerConfig,
#         ):
#         super().__init__()
#         self.model = model
#         self.optimizer_config = optimizer_config
#         self.lr_scheduler_config = lr_scheduler_config
#         self.lr_scheduler = None

#         self.criterion = model.walker.criterion
#         # 移除旧的 evaluator 和 metric_name
#         # self.evaluator = model.walker.evaluator
#         # self.metric_name = model.walker.metric_name

#         self.predict_step_outputs = []

#         # 初始化用于收集验证和测试阶段的预测和目标
#         self.validation_preds = []
#         self.validation_targets = []

#         self.test_preds = []
#         self.test_targets = []

#     def configure_optimizers(self):
#         optimizer = self.optimizer_config.setup(self.model)
#         self.lr_scheduler = self.lr_scheduler_config.setup(optimizer)
#         return optimizer

#     def forward(self, batch):
#         return self.model(batch)

#     @torch._dynamo.disable  # 使用 @torch._dynamo.disable 禁用编译
#     def loss_log(self, loss, batch_size):
#         self.log('training/loss', loss, batch_size=batch_size,
#                  on_step=True, on_epoch=True, logger=True, rank_zero_only=True)
#         self.log('training/lr', self.lr_scheduler.lr,
#                  on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
#         self.log('training/step', float(self.global_step),
#                  on_step=True, on_epoch=False, logger=True, rank_zero_only=True)
        
#     @torch.compiler.disable
#     def inference_loss_log(self, mode, loss, perf):
#         self.log(f'{mode}/loss', loss,
#                  on_step=False, on_epoch=True, logger=True, sync_dist=True)
#         if isinstance(self.metric_name, str):
#             metric_mean, metric_count = self.parse_perf(perf)
#             self.log(f'{mode}/{self.metric_name}', metric_mean, batch_size=metric_count,
#                      on_step=False, on_epoch=True, logger=True, sync_dist=True)
#         else:
#             assert isinstance(self.metric_name, list)
#             for metric_name in self.metric_name:
#                 metric_mean, metric_count = self.parse_perf(perf[metric_name])
#                 self.log(f'{mode}/{metric_name}', metric_mean, batch_size=metric_count,
#                          on_step=False, on_epoch=True, logger=True, sync_dist=True)

#     def training_step(self, batch, batch_idx):
#         assert self.model.training
#         y_hat, y = self.forward(batch)
#         loss = self.criterion(y_hat, y)
#         self.lr_scheduler.step(self.global_step)
#         self.training_log(loss.detach(), y_hat.shape[0])
#         return loss

#     @torch.autocast(device_type='cuda', dtype=torch.float32)
#     def inference(self, batch):
#         assert not self.model.training
#         return self.forward(batch)

#     def validation_step(self, batch, batch_idx):
#         y_hat, y = self.inference(batch)
#         # print("\ny_hat.shape", y_hat.shape)
#         # print("\ny.shape", y.shape)
#         loss = self.criterion(y_hat, y)

#         # 获取预测概率并调整形状
#         if y_hat.shape[1] == 2:
#             preds = torch.softmax(y_hat, dim=-1)[:, 1].unsqueeze(1)  # 形状变为 [batch_size, 1]
#         else:
#             preds = torch.softmax(y_hat, dim=-1)  # 多分类保持形状 [batch_size, num_classes]

#         # 收集预测和目标
#         self.validation_preds.append(preds.detach().cpu())
#         self.validation_targets.append(y.detach().cpu())

#         # 记录损失
#         self.log('validation/loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

#     def on_validation_epoch_end(self):
#         if len(self.validation_preds) == 0:
#             self.log('validation/roc_auc', float('nan'), prog_bar=True, sync_dist=True)
#             return

#         # 将所有批次的预测和目标拼接起来
#         all_preds = torch.cat(self.validation_preds).numpy()
#         all_targets = torch.cat(self.validation_targets).numpy()

#         # 计算 ROC AUC
#         try:
#             perf = eval_rocauc(all_targets, all_preds)
#         except RuntimeError as e:
#             perf = {'rocauc': float('nan')}
#             self.print(str(e))

#         # 记录 ROC AUC
#         self.log('validation/roc_auc', perf['rocauc'], prog_bar=True, sync_dist=True)

#         # 清空列表以准备下一个 epoch
#         self.validation_preds.clear()
#         self.validation_targets.clear()

#     def test_step(self, batch, batch_idx):
#         y_hat, y = self.inference(batch)
#         loss = self.criterion(y_hat, y)

#         # 获取预测概率并调整形状
#         if y_hat.shape[1] == 2:
#             preds = torch.softmax(y_hat, dim=-1)[:, 1].unsqueeze(1)  # 形状变为 [batch_size, 1]
#         else:
#             preds = torch.softmax(y_hat, dim=-1)  # 多分类保持形状 [batch_size, num_classes]

#         # 收集预测和目标
#         self.test_preds.append(preds.detach().cpu())
#         self.test_targets.append(y.detach().cpu())

#         # 记录损失
#         self.log('test/loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

#     def on_test_epoch_end(self):
#         if len(self.test_preds) == 0:
#             self.log('test/roc_auc', float('nan'), prog_bar=True, sync_dist=True)
#             return

#         # 将所有批次的预测和目标拼接起来
#         all_preds = torch.cat(self.test_preds).numpy()
#         all_targets = torch.cat(self.test_targets).numpy()

#         # 计算 ROC AUC
#         try:
#             perf = eval_rocauc(all_targets, all_preds)
#         except RuntimeError as e:
#             perf = {'rocauc': float('nan')}
#             self.print(str(e))

#         # 记录 ROC AUC
#         self.log('test/roc_auc', perf['rocauc'], prog_bar=True, sync_dist=True)

#         # 清空列表以准备下一个 epoch
#         self.test_preds.clear()
#         self.test_targets.clear()

#     @torch._dynamo.disable 
#     def predict_step(self, batch, batch_idx):
#         y_hat, y = self.inference(batch)
#         self.predict_step_outputs.append((batch, y_hat, y))

#     def on_predict_epoch_end(self):
#         self.predict_step_outputs.clear()