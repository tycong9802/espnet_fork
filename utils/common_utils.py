import torch
import torch.nn.functional as F


def padding_audio_vanilla(speech, target_length):
    audio_input = torch.zeros([1, target_length])
    for i in range(len(speech[0])):
        audio_input[0,i] = speech[0,i]
    speech = audio_input
    return speech, torch.tensor([speech.size(1)])


def padding_audio_repeat_head(speech, target_length, repeat_head_length):
    audio_input = torch.zeros([1, target_length])
    for i in range(len(speech[0])):
        audio_input[0,i] = speech[0,i]

    # import math
    # torch.set_printoptions(threshold=math.inf)

    speech_head = torch.zeros([1, repeat_head_length])

    for i in range(repeat_head_length):
        speech_head[0, i] = speech[0,i]

    for i in range(len(speech[0]),target_length, repeat_head_length):
        for j in range(repeat_head_length):
            audio_input[0, i] = speech_head[0, j]

    speech = audio_input
    # print(f'DEBUG: speech = {speech}')
    return speech, torch.tensor([speech.size(1)])


def padding_audio_repeat_sentances(source_tensor, target_length):
    # import math
    # torch.set_printoptions(threshold=math.inf)
    target_data = torch.zeros(1, target_length)
    source_shape = source_tensor.shape
    target_data[:, :source_shape[1]] = source_tensor

    # source_tensor = source_tensor.to(target_data.device)
    target_tensor = target_data.to(source_tensor.device)

    # Updated version: Step 1: Reserving multiple 0s before padding the repeated sentences
    num_of_zeros = 100
    tensor_zeros = torch.zeros(1, num_of_zeros)
    concat_tensor = torch.cat((source_tensor, tensor_zeros) ,dim=1)

    # Step 2: Calculate the number of repetitions needed to fill the target tensor
    num_repetitions = target_tensor.size(1) // concat_tensor.size(1)

    # Step 3: Repeat the source tensor along the second dimension to match the target tensor's size
    repeated_source = concat_tensor.repeat(1, num_repetitions)

    # Step 4: Calculate the remaining elements that need to be filled in the target tensor
    remaining_elements = target_tensor.size(
        1) - repeated_source.size(1)

    # Step 5: Slice and assign the remaining elements from the repeated source to the target tensor
    target_tensor[:, :target_tensor.size(
        1) - remaining_elements] = repeated_source[:, :]

    remaining_tensor = torch.zeros(1, remaining_elements)
    target_tensor[:, target_tensor.size(
        1) - remaining_elements:] = remaining_tensor[:, :remaining_elements]

    return target_tensor, torch.tensor([target_tensor.size(1)])


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
