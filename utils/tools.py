import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 1 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == "decay":
        lr_adjust = {epoch: args.learning_rate * (args.lr_decay ** ((epoch - 1) // 1)) }
    elif args.lradj == "step":
        # OneCycleLR already updates every parameter group. Reassigning the
        # first group's LR to all groups destroys differential fine-tuning.
        lr_adjust = {}
    elif args.lradj == "exp":
        lr_adjust = {}
        scheduler.step()

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        base_lr = max(float(args.learning_rate), np.finfo(float).eps)
        for param_group in optimizer.param_groups:
            target_lr = float(param_group.get("target_lr", base_lr))
            param_group['lr'] = lr * target_lr / base_lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """Save the best checkpoint and stop after a metric stops improving.

    Forecasting keeps the default ``mode="min"`` for validation loss.
    Classification uses ``mode="max"`` so metrics such as Macro F1 can be
    passed directly instead of being negated and reported as a loss.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        mode="min",
        metric_name="Validation loss",
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.metric_name = metric_name
        self.best_value = np.inf if mode == "min" else -np.inf

    def __call__(self, value, model, path):
        score = -value if self.mode == "min" else value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(value, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(value, model, path)
            self.counter = 0

    def save_checkpoint(self, value, model, path):
        if self.verbose:
            direction = "decreased" if self.mode == "min" else "increased"
            print(
                f'{self.metric_name} {direction} '
                f'({self.best_value:.6f} --> {value:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.best_value = value


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def compare_tensors(tensor1, tensor2):
    """
    Compares two PyTorch tensors element-wise and returns a tensor with 1 where tensor1 is greater than or
    equal to tensor2, and 0 where tensor1 is less than tensor2.

    Args:
        tensor1 (torch.Tensor): A PyTorch tensor.
        tensor2 (torch.Tensor): A PyTorch tensor with the same shape as tensor1.

    Returns:
        A PyTorch tensor with the same shape as tensor1, containing 1s where tensor1 is greater than or equal to
        tensor2, and 0s where tensor1 is less than tensor2.
    """
    # Use PyTorch's element-wise comparison function to create a tensor of 1s and 0s
    comparison = torch.ge(tensor1, tensor2)

    # Convert the boolean tensor to a tensor of 1s and 0s
    result = comparison.int()

    return result.type_as(torch.LongTensor())

def transfer_weights(weights_path, model, exclude_head=True, device="cpu", strict=True):
    """Transfer a pre-trained backbone and fail on silent incompatibilities.

    Checkpoints may contain a plain state dict or a ``model_state_dict`` entry.
    ``module.`` prefixes from DataParallel are normalised.  Forecast/rebuild
    heads are intentionally ignored, while encoder and prompt parameters that
    exist in the target model are required when ``strict`` is true.
    """

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        source_state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)!r}")

    source_state = {
        (name[7:] if name.startswith("module.") else name): value
        for name, value in source_state.items()
    }
    target_state = model.state_dict()
    ignored_tokens = (
        "head.",
        "projection.",
        "denoising_patch_decoder.",
        "diffusion.",
    )

    matched = {}
    shape_mismatches = {}
    for name, value in source_state.items():
        if exclude_head and any(token in name for token in ignored_tokens):
            continue
        if name not in target_state:
            continue
        if target_state[name].shape != value.shape:
            shape_mismatches[name] = (tuple(value.shape), tuple(target_state[name].shape))
            continue
        matched[name] = value

    if shape_mismatches:
        details = ", ".join(
            f"{name}: checkpoint{src} != model{dst}"
            for name, (src, dst) in shape_mismatches.items()
        )
        raise RuntimeError(f"Incompatible pre-trained parameter shapes: {details}")
    if not matched:
        raise RuntimeError("No compatible parameters were found in the pre-trained checkpoint")

    required_state_names = {"sos_token"}
    required_prefixes = ["enc_embedding.", "encoder."]
    if any(name.startswith("regime_predictor.") for name in target_state):
        required_prefixes.extend(["regime_predictor.", "soft_prompt_generator."])
        if getattr(model, "regime_label_method", None) == "trend_quantile":
            required_state_names.update(
                {"regime_down_thresh", "regime_up_thresh"}
            )
    if any(name.startswith("scene_wiki_router.") for name in target_state):
        required_prefixes.append("scene_wiki_router.")
        required_state_names.update({"scene_scaler_mean", "scene_scaler_scale"})
    if strict:
        missing_required = [
            name
            for name in target_state
            if (
                name in required_state_names
                or any(name.startswith(prefix) for prefix in required_prefixes)
            )
            and name not in matched
        ]
        if missing_required:
            preview = ", ".join(missing_required[:20])
            suffix = " ..." if len(missing_required) > 20 else ""
            raise RuntimeError(
                "The pre-trained checkpoint is incomplete for this model. "
                f"Missing {len(missing_required)} required parameters: {preview}{suffix}"
            )

    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    target_parameters = dict(model.named_parameters())
    matched_parameter_elements = sum(
        int(target_parameters[name].numel())
        for name in matched
        if name in target_parameters
    )
    target_parameter_elements = sum(
        int(parameter.numel()) for parameter in target_parameters.values()
    )
    required_parameter_names = [
        name
        for name in target_parameters
        if name in required_state_names
        or any(name.startswith(prefix) for prefix in required_prefixes)
    ]
    required_parameter_elements = sum(
        int(target_parameters[name].numel()) for name in required_parameter_names
    )
    matched_required_elements = sum(
        int(target_parameters[name].numel())
        for name in required_parameter_names
        if name in matched
    )
    transfer_audit = {
        "matched_state_tensors": int(len(matched)),
        "matched_parameter_elements": matched_parameter_elements,
        "target_parameter_elements": target_parameter_elements,
        "target_parameter_coverage_pct": 100.0
        * matched_parameter_elements
        / max(1, target_parameter_elements),
        "required_backbone_parameter_elements": required_parameter_elements,
        "matched_required_backbone_elements": matched_required_elements,
        "required_backbone_coverage_pct": 100.0
        * matched_required_elements
        / max(1, required_parameter_elements),
    }
    model.pretrain_transfer_audit = transfer_audit
    print(
        "Transferred "
        f"{transfer_audit['matched_state_tensors']} state tensors / "
        f"{transfer_audit['matched_parameter_elements']:,} parameter elements "
        f"({transfer_audit['target_parameter_coverage_pct']:.2f}% of target; "
        f"required backbone {transfer_audit['required_backbone_coverage_pct']:.2f}%) "
        f"from {weights_path}"
    )
    return model.to(device)

def show_series(batch_x, batch_x_m, pred_batch_x, idx, time_points=336):

    batch_x = batch_x.permute(0, 2, 1).reshape(batch_x.shape[0], -1)
    batch_x_m = batch_x_m.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)
    pred_batch_x = pred_batch_x.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)

    bs = batch_x.shape[0]

    if time_points is None:
        time_points = batch_x.shape[1]

    positive_numbers = batch_x_m.shape[0] // bs

    batch_x = batch_x.numpy()
    batch_x_m = batch_x_m.numpy()

    x = list(range(time_points))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for t_i in range(time_points - 1):
        for pn in range(positive_numbers):
            s_i = pn * bs + idx
            if batch_x_m[s_i][t_i] == 0:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], ':', color='grey', alpha=0.5, label='masked')
            else:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color=colors[pn], label='unmasked')

        axs[1].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color='blue', label='original')
        axs[1].plot([x[t_i], x[t_i + 1]], [pred_batch_x[idx][t_i], pred_batch_x[idx][t_i + 1]], '-', color='orange', label='prediction')

    axs[0].set_title('Multi-masked time series')
    axs[0].set_xlabel('X - time points')
    axs[0].set_ylabel('Y - time values')

    axs[1].set_title('Original vs Reconstruction')
    axs[1].set_xlabel('X - time points')
    axs[1].set_ylabel('Y - time values')

    return fig

def show_matrix(logits, positive_matrix, rebuild_weight_matrix):

    logits = logits.cpu().numpy()
    fig_logits = plt.figure(figsize=(80, 80))
    sns.heatmap(logits, cmap='coolwarm', vmin=np.min(logits), vmax=np.max(logits), annot=False, fmt='.2f', square=False)

    positive_matrix = positive_matrix.cpu().numpy()
    fig_positive_matrix = plt.figure(figsize=(80, 80))
    sns.heatmap(positive_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.1f', square=False)

    rebuild_weight_matrix = rebuild_weight_matrix.cpu().numpy()
    fig_rebuild_weight_matrix = plt.figure(figsize=(100, 100))
    sns.heatmap(rebuild_weight_matrix, cmap='coolwarm', vmin=0, vmax=1, annot=False, fmt='.3f', square=False)

    return fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix

class ContrastiveWeight(nn.Module):

    def __init__(self, args):
        super(ContrastiveWeight, self).__init__()
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.positive_nums = args.positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0])

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat((torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])), dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(torch.nn.Module):

    def __init__(self, args):
        super(AggregationRebuild, self).__init__()
        self.args = args
        self.temperature = args.temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.positive_nums = args.positive_nums

    def forward(self, similarity_matrix, batch_emb_om):

        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb
