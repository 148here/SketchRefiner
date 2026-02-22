import argparse
import multiprocessing

# 设置 multiprocessing start method 为 'spawn'，避免 CUDA 在 fork 子进程中重新初始化的问题
# 必须在导入 torch 之前设置
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass


# initialize configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Configuration of sketch refiner network')

    # data configuration
    ## training data
    parser.add_argument('--images', type=str, default='', help='paths of ground truth images; multiple dirs separated by comma, each recursively scanned')
    parser.add_argument('--edges_prefix', type=str, default='', help='path prefix of input edges')
    parser.add_argument('--output', type=str, default='', help='path of output')
    parser.add_argument('--max_move_lower_bound', type=int, default=30, help='lower bound of the randomize interval of deforming algorithm')
    parser.add_argument('--max_move_upper_bound', type=int, default=100, help='upper bound of the randomize interval of deforming algorithm')

    ## validation data
    parser.add_argument('--images_val', type=str, default='', help='(legacy) path of ground truth images for validation (txt file)')
    parser.add_argument('--masks_val', type=str, default='', help='(legacy) path of free-form masks for validation (txt file)')
    parser.add_argument('--sketches_prefix_val', type=str, default='', help='(legacy) path prefix of deformed sketches for validation')
    parser.add_argument('--edges_prefix_val', type=str, default='', help='(legacy) path prefix of edges for validation')
    parser.add_argument('--val_root_dir', type=str, default='', help='root directory of validation images with *_edge/_sketch/_mask companions')

    # training configuration
    ## loss function configuration
    parser.add_argument('--rm_l1_weight', default=1.0, type=float, help='the weight of l1 loss of RM')
    parser.add_argument('--rm_cc_weight', default=0.4, type=float, help='the weight of cc loss of RM')
    parser.add_argument('--em_l1_weight', default=1.0, type=float, help='the weight of l1 loss of EM')
    parser.add_argument('--em_cc_weight', default=0.9, type=float, help='the weight of l1 loss of EM')

    ## network configuration
    parser.add_argument('--train_EM', action='store_true', help='train enhancement module, otherwise train registration module')
    parser.add_argument('--RM_checkpoint', default='', type=str, help='checkpoint path of fixed RM')

    # resume configuration
    parser.add_argument('--resume_checkpoint', default='', type=str, help='optional checkpoint path to resume training (RM or EM weights only)')

    ## training configuration
    parser.add_argument('--max_iters', default=500003, type=int, help='max iterations of training')
    parser.add_argument('--epochs', default=10, type=int, help='epochs of training')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number of data loader')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--val_interval', default=0, type=int, help='the interval of validation, set to 0 for no validation')
    parser.add_argument('--sample_interval', default=10000, type=int, help='the interval of saving training samples')
    parser.add_argument('--checkpoint_interval', default=50000, type=int, help='the interval of saving checkpoints')
    parser.add_argument('--size', default=256, type=int, help='resolution of sketches and edges')

    # edge cache
    parser.add_argument('--clear_edge_cache', action='store_true', help='clear edge cache once and exit (for pre-training cleanup)')
    parser.add_argument('--cache_clear_interval', default=5, type=int, help='clear edge cache every N epochs (0 to disable)')

    # learning rate scheduler
    parser.add_argument('--use_cosine_lr', action='store_true', help='use cosine annealing learning rate scheduler')
    parser.add_argument('--cosine_eta_min', default=0.0, type=float, help='minimum lr for cosine scheduler')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    configs = parse_args()

    if configs.clear_edge_cache:
        try:
            from YZA_patch.generator import clear_edge_cache
            clear_edge_cache()
            print('[SRN] Edge cache cleared.')
        except Exception as e:
            print(f'[SRN] Failed to clear edge cache: {e}')
        exit(0)

    from SRN_src.SRN_trainer import *

    model = SRNTrainer(configs)
    
    if configs.train_EM:
        model.train_EM()
    else:
        model.train_RM()
