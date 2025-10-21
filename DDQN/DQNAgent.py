import copy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 更新目标网络的操作函数，在MyDQNAgent.learn()函数中调用
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if tau >= 1.0:
        target.load_state_dict(source.state_dict())
        return

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * param.data)


class MyDQNAgent:

    def __init__(
        self,
        model,
        action_size,
        gamma=None,
        lr=None,
        e_greed=0.1,
        e_greed_decrement=0,
        update_target_steps=15,
        max_grad_norm: float = 10.0,
        target_tau: float = 1.0,
    ):

        self.action_size = action_size
        self.global_step = 0
        self.update_target_steps = max(1, int(update_target_steps))
        self.e_greed = e_greed  # ϵ-greedy中的ϵ
        self.e_greed_decrement = e_greed_decrement  # ϵ的动态更新因子
        self.model = model.to(device)
        self.target_model = copy.deepcopy(model).to(device)
        self.gamma = gamma  # 回报折扣因子
        self.lr = lr
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.optimizer = optim.Adam(lr=lr, params=self.model.parameters())
        self.max_grad_norm = max_grad_norm
        self.target_tau = float(target_tau)

    def _update_target_model(self):
        soft_update(self.target_model, self.model, self.target_tau)

    # 使用行为策略生成动作
    def sample(self, state, valid_actions: Optional[np.ndarray] = None):
        valid_indices = None
        if valid_actions is not None:
            valid_mask = np.asarray(valid_actions, dtype=bool)
            valid_indices = np.flatnonzero(valid_mask)
            if valid_indices.size == 0:
                valid_indices = np.arange(self.action_size)

        sample = np.random.random()  # [0.0, 1.0)
        if sample < self.e_greed:
            if valid_indices is None:
                act = int(np.random.randint(self.action_size))
            else:
                act = int(np.random.choice(valid_indices))
        else:
            act = self.predict(state, valid_actions)

        # 动态更改e_greed,但不小于0.1
        self.e_greed = max(0.1, self.e_greed - self.e_greed_decrement)

        return act

    # DQN网络做预测
    def predict(self, state, valid_actions: Optional[np.ndarray] = None):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        if valid_actions is not None:
            mask_np = np.asarray(valid_actions, dtype=bool)
            if not np.any(mask_np):
                return 0
            mask_tensor = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            min_value = torch.finfo(q_values.dtype).min
            q_values = q_values.masked_fill(~mask_tensor, min_value)

        act = int(q_values.argmax(dim=1).item())
        return act

    # 更新DQN网络
    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        terminal,
        weights: Optional[Sequence[float]] = None,
        next_valid_actions: Optional[Sequence[Sequence[bool]]] = None,
        n_steps: Optional[Sequence[int]] = None,
    ):
        """Update model with an episode data

        Args:
            state(np.float32): shape of (batch_size, state_size)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
            next_state(np.float32): shape of (batch_size, state_size)
            terminal(np.float32): shape of (batch_size)

        Returns:
            loss(float)
        """

        state_tensor = torch.as_tensor(np.array(state), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(np.array(action), dtype=torch.int64, device=device).unsqueeze(-1)
        reward_tensor = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device).unsqueeze(-1)
        next_state_tensor = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(
            (np.array(terminal) != -1).astype(np.float32),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        if weights is None:
            weight_tensor = torch.ones_like(reward_tensor)
        else:
            weight_tensor = torch.as_tensor(np.array(weights), dtype=torch.float32, device=device).unsqueeze(-1)

        if n_steps is None:
            gamma_tensor = torch.full_like(reward_tensor, self.gamma)
        else:
            gamma_array = np.array(n_steps, dtype=np.float32)
            gamma_tensor = torch.as_tensor(np.power(self.gamma, gamma_array), dtype=torch.float32, device=device).unsqueeze(-1)

        # 1. DQN网络做正向传播
        pred_values = self.model(state_tensor)
        pred_value = pred_values.gather(1, action_tensor)

        # target Q
        with torch.no_grad():
            next_q_online = self.model(next_state_tensor)
            next_q_target = self.target_model(next_state_tensor)

            valid_any = None
            if next_valid_actions is not None:
                processed_masks: List[np.ndarray] = []
                for mask in next_valid_actions:
                    if mask is None:
                        processed_masks.append(np.zeros(self.action_size, dtype=bool))
                    else:
                        processed_masks.append(np.asarray(mask, dtype=bool))
                mask_array = np.stack(processed_masks, axis=0)
                mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool, device=device)
                if mask_tensor.dim() == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                invalid_mask = ~mask_tensor
                min_value = torch.finfo(next_q_online.dtype).min
                next_q_online = next_q_online.masked_fill(invalid_mask, min_value)
                next_q_target = next_q_target.masked_fill(invalid_mask, min_value)
                valid_any = mask_tensor.any(dim=1, keepdim=True)

            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            max_next_q = next_q_target.gather(1, best_next_actions)
            if valid_any is not None:
                max_next_q = torch.where(valid_any, max_next_q, torch.zeros_like(max_next_q))
            target = reward_tensor + (1.0 - done_tensor) * gamma_tensor * max_next_q

        # 4. TD 误差
        td_errors = pred_value - target
        element_loss = self.loss_fn(pred_value, target)
        loss = (weight_tensor * element_loss).mean()

        # 5. 更新DQN的参数
        # 梯度清零
        self.optimizer.zero_grad()
        # 反向计算梯度
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        # 梯度更新
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.update_target_steps == 0:
            self._update_target_model()

        return loss.item(), td_errors.detach().cpu().numpy().squeeze(-1)
