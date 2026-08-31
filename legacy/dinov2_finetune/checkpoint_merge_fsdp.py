import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def consolidate_model(model, save_path):
    full_state_dict = FSDP.full_state_dict(model, rank0_only=True)
    if dist.get_rank() == 0:
        torch.save(full_state_dict, save_path)
        print(f"✅ Saved consolidated checkpoint to {save_path}")

def main():
    dist.init_process_group("nccl")

    model = build_your_model()
    model = FSDP(model)

    # 每个 rank 加载自己 shard
    rank = dist.get_rank()
    ckpt_path = f"/home/li_yu/Proj04_he/done_work/dinov2/dinov2/results/dinov2_he0810_pretrain_imagenet22k/model_0116249.rank_{rank}.pth"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    # 合并并保存
    consolidate_model(model, "final_model.pth")

if __name__ == "__main__":
    main()
