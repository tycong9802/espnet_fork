import torch
import torch.nn.functional as F


def padding_audio_vanilla(speech, target_length):
    audio_input = torch.zeros([1, target_length])
    for i in range(len(speech[0])):
        audio_input[0,i] = speech[0,i]
    speech = audio_input
    return speech, torch.tensor([speech.size(1)])


def audio_cutoff(speech, target_length):
    audio_input = torch.zeros([1, target_length])
    for i in range(target_length):
        audio_input[0,i] = speech[0,i]
    speech = audio_input
    return speech, torch.tensor([speech.size(1)])


def padding_audio(input_tensor, desired_shape):
    # padding = desired_shape[1] - input_tensor.size(1)
    padding_needed = [max(desired_shape[i] - input_tensor[i].shape, 0)
                      for i in range(len(desired_shape))]
    padded_tensor = F.pad(input_tensor, pad=padding_needed)
    # padded_tensor = F.pad(input_tensor, (0, 0, 0, padding, 0, 0))
    return padded_tensor, torch.tensor([padded_tensor.size(1)])


def padding_feats(input_tensor, desired_shape):
    padding = desired_shape[1] - input_tensor.size(1)
    padded_tensor = F.pad(input_tensor, (0, 0, 0, padding, 0, 0))
    return padded_tensor, torch.tensor([padded_tensor.size(1)])
