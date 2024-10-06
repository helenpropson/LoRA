import time 
import math
import torch
import random

import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from torch import Tensor, device, dtype, nn

import inspect
from transformers import GPT2LMHeadModel

from model import GPT2Config, GPT2LMModel
import loralib as lora
from torch.nn import functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
# from optimizer import (
#     create_adam_optimizer, 
#     create_optimizer_scheduler, 
#     add_optimizer_params, 
#     create_adam_optimizer_from_args
# )
from torch.optim import Optimizer

from data_utils import FT_Dataset

# PARAMS
platform = "local"
local_rank = 0
rank = 0
device = 0 # cuda:0
world_size = 1
random_seed = 110
lr = 0.0002
weight_decay = 0.01
correct_bias = True
adam_epislon = 1e-06
no_decay_bias = False
adam_beta1 = 0.9
adam_beta2 = 0.999
scheduler = 'linear'

max_epoch = 5
warmup_step = 500
i_steps = 0
i_lrs = 0.00025
train_data = './data/e2e/train.jsonl'
valid_data = './data/e2e/valid.jsonl'
train_batch_size = 8
valid_batch_size = 4

grad_acc = 1
clip = 0
seq_len = 512

# train
log_interval = 100
eval_interval = 2000
save_interval = 1000
work_dir = "./trained_models/GPT2_S/e2e"
lora_dim = 4
lora_alpha = 32
lora_dropout = 0.1
label_smooth = 0.1
roll_interval = -1
roll_lr = 1e-05
roll_step = 100
eval_epoch = 1

# beam
eval_len = 64
min_length = 0
beam = 10
length_penalty = 0.8
no_repeat_ngram_size = 4
repetition_penalty = 1.0
eos_token_id = [50256, 628]
# output_file = 'predict.26290.b10p08r4.jsonl'

torch.manual_seed(random_seed)
random.seed(random_seed)


# LOAD DATA
train_data = FT_Dataset(
    "./examples/NLG/data/e2e/train.jsonl", 8, 512,
    joint_lm="clm"=='jlm'
)     
train_loader = DataLoader(
    train_data, batch_size=8, num_workers=0, 
    # shuffle=False, 
    pin_memory=False, drop_last=True,
    # sampler=torch.utils.data.sampler.DistributedSampler(train_data, seed=110)
    shuffle=True
)


# LOAD MODEL
config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=lora_dim, 
            lora_attn_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
        )

lm_net = GPT2LMModel(config)
print('loading model pretrained weight.')
lm_net.load_weight(torch.load("./examples/NLG/pretrained_checkpoints/gpt2-pytorch_model.bin"))
# try:
#     lm_net.load_weight(torch.load('./examples/NLG/trained_models/GPT2_S_LL/e2e/model.26290.pt'))
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# lm_net = lm_net.cuda()

lora.mark_only_lora_as_trainable(lm_net) # /data/healthy-ml/scratch/hpropson/.local/lib/python3.10/site-packages/loralib/utils.py
# optimizer = create_adam_optimizer_from_args(lm_net, args)
max_step = (max_epoch * train_data.num_batches + world_size - 1) // world_size
# scheduler = create_optimizer_scheduler(optimizer, args)
# lm_net = torch.nn.parallel.DistributedDataParallel(
#             lm_net, device_ids=[local_rank], output_device=local_rank, 
#             find_unused_parameters=False, broadcast_buffers=False
#         ) # , optimizer = distributed_opt(args, lm_net, optimizer, grad_acc=args.grad_acc)


# hf_model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation = "eager")
# # hf_model.eval()

# # compare weights
# print(lm_net)
# state_dict_1 = lm_net.state_dict()
# state_dict_2 = hf_model.state_dict()
# weights_identical = True
# for key in state_dict_1:
#     if not torch.equal(state_dict_1[key], state_dict_2[key]):
#         print(f"Weights for {key} differ between the two models.")
#         weights_identical = False


# HELPER CODE
class AverageMeter(object):
    """Computes and stores the average and current value
         Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def optimizer_step(_loss, _optimizer, _model, _schedule, args, is_update=True):
#     _loss.backward()

#     if is_update:
#         if args.clip > 0:
#             torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)

#         _optimizer.step()        
#         _optimizer.zero_grad()

#     if _schedule is not None:
#         _schedule.step()


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.98)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)


    def reset_state(self):
        for group in param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


# TRAIN

model = lm_net
# optimizer = AdamW(
#         grouped_parameters, 
#         lr=args.lr, 
#         betas=(args.adam_beta1, args.adam_beta2), 
#         eps=args.adam_epislon, 
#         weight_decay=args.weight_decay, 
#         correct_bias=args.correct_bias
#     )
def _train():
    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            
            # train_validate
            model.train()
            avg_lm_loss = AverageMeter()
            print('start to train the model................', epoch)
            log_start_time = time.time()
            best_val_ppl = None

            # train_loader.sampler.set_epoch(epoch) # dont need to do this for non-distributed; data is shuffled in a different way each epoch

            for idx, data in enumerate(train_loader):   # RandomSampler will shuffle the data randomly each epoch 
                data = {key: value for key, value in data.items()}

                _input = data['input']
                _target = data['target']
                _msk = data['mask']


                _lm_logits, _lm_loss = model(
                    _input, lm_labels=_target, lm_mask=_msk, label_smooth=0.1)

                _lm_loss = _lm_loss.mean() # average loss over the entire batch
                
                train_step += 1
                is_update = True if train_step % grad_acc == 0 else False
                avg_lm_loss.update(_lm_loss.item())
                # optimizer_step(
                #     _lm_loss/(grad_acc), optimizer, model, scheduler, args, is_update=is_update
                # )
                
                # optimizer step code
                _loss = _lm_loss/(grad_acc)
                _loss.backward()

                if is_update:
                    if clip > 0: # clip is zero so this step code is not executed
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                    optimizer.step()        
                    optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if train_step == max_step:
                    break
            
            if train_step >= max_step or (max_epoch is not None and epoch >= max_epoch):
                if rank == 0:
                    print('-' * 100)
                    print('End of training')
                break

    except KeyboardInterrupt:
        if rank == 0:
            print('-' * 100)
            print('Exiting from training early')


# BEAM SEARCH

def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx).contiguous().detach() for layer_past in past)


def _calc_banned_ngram_tokens(
    prev_input_ids: Tensor, 
    num_hypos: int, 
    no_repeat_ngram_size: int, 
    cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs, 
    batch_size, 
    num_beams, 
    prev_output_tokens, 
    repetition_penalty
):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)

        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,                                
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
                history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(
    best_score, 
    best_sequence, 
    batch_size, 
    num_beams, 
    beam_scores, 
    history, 
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.view(-1)[_i] / cur_len ** length_penalty

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.view(-1)[_i] = -float("inf")


def _beam(model, data_iter):
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    with torch.no_grad():
        for idx, data in enumerate(data_iter):
            data = {key: value for key, value in data.items()}

            _id = data['id']
            _query = data['query']
            _query_len = data['query_len']

            ## local adaptation start.

            ## local adaptation end.


            output = None
            score = None

            batch_size = _id.size(0)
            num_beams = beam
            # length_penalty = length_penalty

            _batch = torch.arange(0, _id.size(0), dtype=torch.long)
            
            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).view(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).view(-1)

            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).view(-1)
            
            # scores for each sentence in the beam
            beam_scores = torch.zeros(
                (batch_size, num_beams), dtype=torch.float, device=_query.device
            )

            best_sequence = torch.zeros(
                (batch_size, eval_len), dtype=torch.long, device=_query.device
            )
            best_score = {}

            history = None
            with torch.no_grad():
                for i in range(0, eval_len):
                    print(i)
                    if i == 0:
                        logits, past = model(_query) 
                        logits = logits[_bbatch, (_query_len-1).long(), :] # batch_size * beam, vocab
                    else:
                        #print('token_id.shape', token_id.shape, token_id)
                        #print('past.shape', past[0].shape)
                        #print('len_past.shape', len_past.shape, len_past)
                        
                        logits, past = model(token_id, past=past, len_past=len_past) 
                        logits = logits[:, -1, :]    # batch_size * beam, vocab

                    logits = _postprocess_next_token_scores(           
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=repetition_penalty,                                
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        min_length=min_length,
                        eos_token_id=eos_token_id,
                    )

                    softmax_probs = F.softmax(logits, dim=-1)
                    ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                    vocab_size = softmax_probs.shape[-1] 
                    

                    _logprob = torch.log(softmax_probs) # batch_size * beam, vocab
                    if i == 0:
                        next_scores = _logprob.view(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab
                        
                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.view(batch_size, num_beams, -1)
                        next_scores = next_scores.view(batch_size, -1) # batch_size, beam * vocab

                    next_scores, next_tokens = torch.topk(
                        next_scores, num_beams, dim=1, largest=True, sorted=True
                    )     # batch_size, num_beams
                    
                    beam_id = (next_tokens // vocab_size).view(-1)    # batch_size * num_beams
                    token_id = (next_tokens % vocab_size).view(-1).unsqueeze(-1) # batch_size, num_beams

                    beam_idx = beam_id.view(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                    past = _reorder_cache(past, beam_idx.view(-1))                
                    beam_scores = next_scores # batch_size, num_beams
                    len_past = (_query_len + i).long()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = torch.cat((history[beam_idx.view(-1)], token_id.detach()), dim=1).detach()

                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history, 
                        eos_token_id=eos_token_id
                    )
                
                _add_beam_candidate(
                    best_score, best_sequence, batch_size, num_beams, beam_scores, history
                )


            # with torch.no_grad():
            #     _id = distributed_gather(args, _id)
            #     output = distributed_gather(args, best_sequence)
            #     #score = distributed_gather(args, score)
            #     distributed_sync(args)

            if rank == 0:
                _id = _id.view(-1).cpu()
                output = output.view(-1, output.shape[-1]).cpu()
                #score = score.view(-1, score.shape[-1]).cpu()

                for _b in range(0, _id.shape[-1]):
                    _i = int(_id[_b].item())
                    all_predictions[_i] = {}
                    all_predictions[_i]['id'] = _i
                    all_predictions[_i]['predict'] = output[_b].tolist()
                    #all_predictions[_i]['score'] = score[_b].tolist()

                if idx % 10 == 0:
                    print('inference samples', idx)

    # if rank == 0:
    #     pred_file = os.path.join(work_dir, args.output_file) 
    #     print('saving prediction file', pred_file)
    #     with open(pred_file, 'w') as writer:
    #         for _i in all_predictions:
    #             writer.write(json.dumps(all_predictions[_i]) + '\n')


if __name__ == '__main__':
     _train()
    # _beam(model, train_loader)