import copy
import os
import tempfile
import torch
import torch.nn as nn
from tqdm import tqdm
from src.benchmark.runner import run_full_benchmark
from src.inference.baseline import evaluate_model
from src.models.unet import DoubleConv
from src.training.trainer import DiceBCELoss
from src.utils.logging_utils import get_logger, save_json

class QuantWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

class _QATTrainer:

    def __init__(self, model, device, lr=1e-05, weight_decay=1e-05):
        self.model = model.to(device)
        self.device = device
        self.criterion = DiceBCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def fit(self, train_loader, epochs=1, max_batches=None, logger=None):
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            seen = 0
            for (step, (imgs, masks)) in enumerate(tqdm(train_loader, desc=f'QAT {epoch}/{epochs}', leave=False)):
                if max_batches is not None and step >= max_batches:
                    break
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                seen += imgs.size(0)
            mean_loss = total_loss / max(1, seen)
            if logger:
                logger.info('QAT epoch %d/%d | loss=%.4f', epoch, epochs, mean_loss)

def _fuse_unet_blocks(model):
    for module in model.modules():
        if isinstance(module, DoubleConv):
            if hasattr(torch.ao.quantization, 'fuse_modules_qat'):
                torch.ao.quantization.fuse_modules_qat(module.net, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
            else:
                was_training = module.net.training
                module.net.eval()
                torch.ao.quantization.fuse_modules(module.net, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
                module.net.train(was_training)

def _serialized_model_size_mb(model):
    (fd, temp_path) = tempfile.mkstemp(suffix='.pt')
    os.close(fd)
    try:
        torch.save(model.state_dict(), temp_path)
        return os.path.getsize(temp_path) / 1000000.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _select_quant_engine():
    available = torch.backends.quantized.supported_engines
    if 'fbgemm' in available:
        return 'fbgemm'
    if 'qnnpack' in available:
        return 'qnnpack'
    return available[0] if available else 'fbgemm'

def run_qat_benchmark(model, train_loader, test_loader, output_dir='results', train_device='cpu', qat_epochs=2, qat_lr=1e-05, qat_weight_decay=1e-05, max_qat_batches=None, warmup=20, latency_runs=100, batch_size_throughput=16, input_size=(256, 256)):
    logger = get_logger('qat_benchmark')
    logger.info('Running QAT benchmark | train_device=%s | epochs=%d', train_device, qat_epochs)
    fp_model = copy.deepcopy(model).float().cpu()
    quant_model = QuantWrapper(fp_model)
    quant_model.train()
    _fuse_unet_blocks(quant_model.model)
    engine = _select_quant_engine()
    torch.backends.quantized.engine = engine
    logger.info('Using quantized engine: %s', engine)
    quant_model.qconfig = torch.ao.quantization.get_default_qat_qconfig(engine)
    torch.ao.quantization.prepare_qat(quant_model, inplace=True)
    trainer = _QATTrainer(quant_model, device=train_device, lr=qat_lr, weight_decay=qat_weight_decay)
    trainer.fit(train_loader=train_loader, epochs=qat_epochs, max_batches=max_qat_batches, logger=logger)
    quant_model.eval().cpu()
    int8_model = torch.ao.quantization.convert(quant_model, inplace=False)
    int8_model.eval()
    quality = evaluate_model(int8_model, test_loader, device='cpu', precision='fp32')
    benchmark = run_full_benchmark(model_fn=lambda x: int8_model(x), data_loader=test_loader, device='cpu', precision='fp32', warmup=warmup, latency_runs=latency_runs, batch_size_throughput=batch_size_throughput, input_size=input_size)
    result = {'method': 'QAT (int8)', 'precision': 'int8', 'device': train_device, 'runtime_device': 'cpu', 'quant_engine': engine, 'model_size_mb': _serialized_model_size_mb(int8_model), 'p50_latency_ms': benchmark['latency']['median_ms'], 'p95_latency_ms': benchmark['latency']['p95_ms'], 'mean_latency_ms': benchmark['latency']['mean_ms'], 'throughput_img_sec': benchmark['throughput']['throughput_img_sec'], 'batch_size': batch_size_throughput, **quality, **{f'memory_{k}': v for (k, v) in benchmark['memory'].items()}, 'status': 'ok'}
    save_path = os.path.join(output_dir, 'logs', 'qat_result.json')
    save_json(result, save_path)
    logger.info('Saved QAT result -> %s', save_path)
    return result
