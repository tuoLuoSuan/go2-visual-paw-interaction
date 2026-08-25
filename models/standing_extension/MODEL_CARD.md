# Standing extension model card

- 文件：`best_gru_standing_px_v4_mask.npz`
- SHA-256：`5a6cdb1a2bc86a99cb223196d9eac4ebf748e07aef1e32f08f3ca4cebdb822d5`
- 任务标识：`standing_px`
- 主干：GRU
- 观测维度：29
- 动作维度：12
- 部署角色：`v4_mask` 站姿扩展

该模型只对应 FORMAL-03 的站姿定性扩展。三个录制段得到执行层 `STAND_HS_OK`，但没有与 FORMAL-02 等价的独立接触/保持端点标注，因此本模型及其记录不能与趴姿主实验合并计算成功率，也不能被描述为经过充分验证的跨姿态通用方法。
