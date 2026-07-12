import argparse
import os
import sys

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datasets
import models
from models.sam import perforate_model
import utils
from statistics import mean
import torch
import torch.distributed as dist

# Attempt to import PerforatedAI
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    pai_available = True
except ImportError:
    pai_available = False

# Attempt to import wandb
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'dice_iou':
        # Metric used in the Medical-SAM paper
        metric_fn = utils.calc_dice_iou
        metric1, metric2, metric3, metric4 = 'dice', 'iou', 'none', 'none'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    # See if we can activate PAI
    pai_perforate_image_encoder = config.get('pai_perforate_image_encoder', False)
    pai_perforate_adapter       = config.get('pai_perforate_adapter', False)
    pai_perforate_mask_decoder  = config.get('pai_perforate_mask_decoder', False)
    pai_active                  = pai_available and config.get('pai_enable', False)
    if config.get('pai_enable', False) and not pai_available:
        if local_rank == 0:
            log('config pai_enable=True but perforatedai is not installed; training without dendrites.')
    if pai_active and not (pai_perforate_image_encoder or pai_perforate_adapter or pai_perforate_mask_decoder):
        if local_rank == 0:
            log('Warning: pai_enable=True but none of pai_perforate_image_encoder / '
                'pai_perforate_adapter / pai_perforate_mask_decoder is set; nothing to '
                'perforate, training without dendrites.')
        pai_active = False
    if pai_active and config.get('epoch_val') != 1:
        if local_rank == 0:
            log('Warning: pai_enable=True but epoch_val != 1; PAI only sees a validation '
                'score on epochs where eval runs, so dendrite switch timing will be skewed.')

    # See if we can activate wandb
    wandb_active = wandb_available and config.get('wandb_enable', False) and local_rank == 0
    if config.get('wandb_enable', False) and not wandb_available:
        if local_rank == 0:
            log('config wandb_enable=True but wandb is not installed; skipping wandb logging.')
    if wandb_active:
        # Dendrite restructure relaunches the process so use a stable id tied to save_path
        wandb_run_id = os.path.basename(save_path.rstrip('/'))
        try:
            wandb.init(
                project=config.get('wandb_project', 'sam-adapter-drive'),
                entity=config.get('wandb_entity') or None,
                id=wandb_run_id,
                resume='allow',
                name=wandb_run_id,
                config=config,
            )
        except Exception as e:
            # The deterministic id can collide with a run that was deleted
            # server-side (deleted ids can't be resumed/reused)
            log(f'wandb.init failed to reuse run id "{wandb_run_id}" ({e}); '
                'starting a new run with a fresh id instead.')
            wandb.init(
                project=config.get('wandb_project', 'sam-adapter-drive'),
                entity=config.get('wandb_entity') or None,
                name=wandb_run_id,
                config=config,
            )

    epoch_start = config.get('resume') + 1 if config.get('resume') is not None else 1
    model = models.make(config['model']).cuda()

    # Load SAM Weights before Perforating
    sam_checkpoint = torch.load(config['sam_checkpoint'], map_location='cuda:{}'.format(local_rank))
    model.load_state_dict(sam_checkpoint, strict=False)

    # Choose which modules are trained from the config
    ft_image_encoder = config.get('ft_image_encoder', False)
    ft_image_adapter = config.get('ft_image_adapter', True)
    ft_mask_decoder  = config.get('ft_mask_decoder', True)
    for name, para in model.named_parameters():
        if "dendrite" in name:
            # PAI dendrites always train
            continue
        if "image_encoder.prompt_generator" in name:
            para.requires_grad_(ft_image_adapter)
        elif "image_encoder" in name:
            para.requires_grad_(ft_image_encoder)
        elif "mask_decoder" in name:
            para.requires_grad_(ft_mask_decoder)

    # Perforate the model
    if pai_active:
        # Fix the save name to work with wandb and PAI
        pai_save_name = os.path.basename(save_path.rstrip('/'))
        try:
            os.symlink(os.path.abspath(save_path), pai_save_name)
        except FileExistsError:
            pass
        model = perforate_model(model, save_name=pai_save_name,
                                 perforate_image_encoder=pai_perforate_image_encoder,
                                 perforate_adapter=pai_perforate_adapter,
                                 perforate_mask_decoder=pai_perforate_mask_decoder)
        if args.pai_load_folder is not None:
            model = UPA.load_system(model, args.pai_load_folder, 'latest', True)
            # Get the correct epoch to start from on resume
            epoch_start = GPA.pai_tracker.member_vars['num_epochs_run'] + 1
            if local_rank == 0:
                log('Loaded PAI dendrite structure and tracker state from {}/latest.pt'.format(
                    args.pai_load_folder))

    model = model.cuda()

    # Setup the PAI Optimizer + Scheduler
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    if pai_active:
        GPA.pai_tracker.set_optimizer_instance(optimizer)
    model.optimizer = optimizer
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.get('lr_factor', 0.5),
        patience=config.get('lr_patience', 3),
        min_lr=config.get('lr_min', 0),
    )

    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    # DDP wrap is immediately unwrapped via `.module`, matching this repo's
    # existing (pre-PAI) behavior; the raw module is what gets trained below.
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()

    epoch = epoch_start - 1
    while pai_active or epoch < epoch_max:
        epoch += 1
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)

        if local_rank == 0:
            log_info = ['epoch {}'.format(epoch) if pai_active else 'epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)
            if wandb_active:
                wandb.log({
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': train_loss_G,
                }, step=epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))
            lr_scheduler.step(result1)

            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                writer.add_scalars(metric4, {'val': result4}, epoch)

                if wandb_active:
                    val_log = {'val_{}'.format(metric1): result1, 'val_{}'.format(metric2): result2}
                    if metric3 != 'none':
                        val_log['val_{}'.format(metric3)] = result3
                    if metric4 != 'none':
                        val_log['val_{}'.format(metric4)] = result4
                    wandb.log(val_log, step=epoch)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / max(epoch_max - epoch_start + 1, 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()

            # Track validation score via PAI
            if pai_active:
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(result1, model)
                model = model.cuda()

                # End training once PAI is done
                if training_complete:
                    if local_rank == 0:
                        log('PAI training complete!')
                        os.makedirs(save_path, exist_ok=True)
                        with open(os.path.join(save_path, '.training_complete'), 'w') as f:
                            f.write('complete')
                        if wandb_active:
                            wandb.finish()
                    dist.barrier()
                    dist.destroy_process_group()
                    sys.exit(0)

                # Dendrites need to be integrated
                if restructured:
                    if local_rank == 0:
                        log('Dendrites restructured; exiting so the DDP process group can be '
                            'rebuilt for the new architecture. Relaunch with --pai_load_folder '
                            '{} to resume (train_distributed.sh does this automatically).'.format(save_path)
                            )
                    # New dendrites -> DDP + Optimizer are stale
                    # train_distributed.sh handles restarts automatically
                    # wandb.finish() is deliberately skipped here (not just
                    # a normal process exit): the resumed process reuses
                    # this same wandb run id, so leave it open to resume.
                    dist.barrier()
                    dist.destroy_process_group()
                    sys.exit(0)

    if wandb_active:
        wandb.finish()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--pai_load_folder', type=str, default=None,
                         help='Folder to load PAI dendrite/tracker state from (for resuming '
                              'after a dendrite restructure exits the process).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
