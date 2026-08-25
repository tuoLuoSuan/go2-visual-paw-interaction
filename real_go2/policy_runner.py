"""纯 numpy 策略推理器（部署版：VM 控制环境无需安装 torch）。

支持 MLP 与 GRU 主干（GRUPolicy 结构：gru -> net(2层) -> mean）。
GRU 使用与 PyTorch 一致的更新式，隐藏态由 runner 维护（reset() 清零）。

用法：
    runner = PolicyRunner("xxx.npz")
    actions = runner.act(obs)   # obs: (obs_dim,) -> 动作增量（确定性）
    runner.reset()              # GRU：回合/会话开始时清隐藏态
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PolicyRunner:
    def __init__(self, checkpoint_path):
        """加载 checkpoint：优先 .npz（纯 numpy）；.pt 需要 torch。"""
        state = None
        meta = {}
        if str(checkpoint_path).endswith(".npz"):
            data = np.load(checkpoint_path, allow_pickle=False)
            meta = {"obs_dim": int(data["obs_dim"]), "act_dim":
                    int(data["act_dim"]), "task": str(data["task"]),
                    "backbone": str(data["backbone"])}
            state = {key: data[key] for key in data.files
                     if key not in meta}
        else:
            import torch  # 仅 .pt 加载需要
            ckpt = torch.load(checkpoint_path, map_location="cpu",
                              weights_only=False)
            meta = {"obs_dim": int(ckpt.get("obs_dim", 27)), "act_dim":
                    int(ckpt.get("act_dim", 6)),
                    "task": ckpt.get("task", "prone_px"),
                    "backbone": ckpt.get("backbone", "mlp")}
            state = {k: v.numpy().astype(np.float64)
                     for k, v in ckpt["policy"].items()}
        self.obs_dim = meta["obs_dim"]
        self.act_dim = meta["act_dim"]
        self.task = meta["task"]
        self.backbone = meta["backbone"]
        if self.backbone not in ("mlp", "gru"):
            raise ValueError(f"PolicyRunner 仅支持 mlp/gru，收到 "
                             f"{self.backbone}")
        if self.backbone == "gru":
            self.hidden_size = int(state["gru.weight_hh_l0"].shape[1])
            self._w_ih = np.asarray(state["gru.weight_ih_l0"],
                                    dtype=np.float64)   # (3H, obs)
            self._w_hh = np.asarray(state["gru.weight_hh_l0"],
                                    dtype=np.float64)   # (3H, H)
            self._b_ih = np.asarray(state["gru.bias_ih_l0"],
                                    dtype=np.float64)   # (3H,)
            self._b_hh = np.asarray(state["gru.bias_hh_l0"],
                                    dtype=np.float64)   # (3H,)
            self.hidden = np.zeros((1, self.hidden_size), dtype=np.float64)
            layers = []
            for key in ("net.0", "net.2"):
                w = np.asarray(state[f"{key}.weight"], dtype=np.float64)
                b = np.asarray(state[f"{key}.bias"], dtype=np.float64)
                layers.append((w, b))
            self.layers = layers
        else:
            self.hidden_size = 0
            self.hidden = None
            layers = []
            for key in ("net.0", "net.2", "net.4"):
                w = np.asarray(state[f"{key}.weight"], dtype=np.float64)
                b = np.asarray(state[f"{key}.bias"], dtype=np.float64)
                layers.append((w, b))
            self.layers = layers
        self.w_mean = np.asarray(state["mean.weight"], dtype=np.float64)
        self.b_mean = np.asarray(state["mean.bias"], dtype=np.float64)
        if self.w_mean.shape[0] != self.act_dim:
            raise ValueError("checkpoint act_dim 与 mean 头不匹配")

    def reset(self):
        if self.backbone == "gru":
            self.hidden = np.zeros((1, self.hidden_size), dtype=np.float64)

    def _gru_step(self, x):
        """单步 GRU（PyTorch 语义）：x=(1,obs) -> (1,H)。门序 [r, z, n]。"""
        h = self.hidden
        x_gates = x @ self._w_ih.T + self._b_ih        # (1, 3H)
        h_gates = h @ self._w_hh.T + self._b_hh        # (1, 3H)
        r = _sigmoid(x_gates[:, :self.hidden_size]
                     + h_gates[:, :self.hidden_size])
        z = _sigmoid(x_gates[:, self.hidden_size:2 * self.hidden_size]
                     + h_gates[:, self.hidden_size:2 * self.hidden_size])
        n = np.tanh(x_gates[:, 2 * self.hidden_size:]
                    + r * h_gates[:, 2 * self.hidden_size:])
        h_new = (1.0 - z) * n + z * h
        self.hidden = h_new
        return h_new

    def act(self, obs):
        """输入观测（(obs_dim,) 或 (batch, obs_dim)），返回动作增量（确定性）。"""
        x = np.asarray(obs, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[1] != self.obs_dim:
            raise ValueError(f"obs 维度 {x.shape[1]} != 期望 {self.obs_dim}")
        if self.backbone == "gru":
            if x.shape[0] != 1:
                raise ValueError("GRU 推理仅支持单样本")
            x = self._gru_step(x)
        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)
        out = x @ self.w_mean.T + self.b_mean
        if single:
            return out[0]
        return out
