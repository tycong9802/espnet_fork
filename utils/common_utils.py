import torch
import torch.nn.functional as F
import logging

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

    # source_tensor = source_tensor.to(target_data.device)
    target_tensor = target_data.to(source_tensor.device)

    # Updated version: Step 1: Reserving multiple 0s before padding the repeated sentences
    reserved_pos = 2000

    # Step 2: Calculate the number of repetitions needed to fill the target tensor
    num_repetitions = (target_tensor.size(1) - reserved_pos) // source_tensor.size(1)

    # Step 3: Repeat the source tensor along the second dimension to match the target tensor's size
    repeated_source = source_tensor.repeat(1, num_repetitions)

    # Step 4: Calculate the remaining elements that need to be filled in the target tensor
    remaining_elements = target_tensor.size(1) - repeated_source.size(1) - reserved_pos

    # Step 5: Slice and assign the remaining elements from the repeated source to the target tensor
    target_tensor[:, reserved_pos:target_tensor.size(
        1) - remaining_elements] = repeated_source[:, :]
    target_tensor[:, target_tensor.size(
        1) - remaining_elements:] = repeated_source[:, :remaining_elements]

    return target_tensor, torch.tensor([target_tensor.size(1)])

# TODO: Create a new padding function (speech, speech#5, the number of 0s, target_length)
#       1. Add 0s after speech (#1, #2, #3, #4, #5)
#       2. Add speech #5 to the end of the step 1
#       3. Repeat step 1 and 2 until the length is equal to target_length
#       4. Return the padding speech (speech #3, 0...0, speech #5, 0...0, speech#5, 0...0)
def padding_audio_with_0s_and_speech5(source_tensor, target_length, repeat_tensor, reserved_pos):
    # import math
    # torch.set_printoptions(threshold=math.inf)
    target_data = torch.zeros(1, target_length)

    # source_tensor = source_tensor.to(target_data.device)
    target_tensor = target_data.to(source_tensor.device)

    # Updated version: Step 1: Reserving multiple 0s before padding the repeated sentences
    # reserved_pos = 0

    # Step 2: Calculate the number of repetitions needed to fill the target tensor
    num_repetitions = (target_tensor.size(1) - source_tensor.size(1) - reserved_pos) // repeat_tensor.size(1)
    # logging.info('target_tensor.size(1):  ' + str(target_tensor.size(1)))
    # logging.info('reserved_pos: ' + str(reserved_pos))
    # logging.info('repeat_tensor.size(1): ' + str(repeat_tensor.size(1)))

    # Step 3: Repeat the source tensor along the second dimension to match the target tensor's size
    repeated_source = repeat_tensor.repeat(1, num_repetitions)

    # logging.info('repeated_source.size():  ' + str(repeated_source.size()))
    # logging.info('repeated_source.size(0): ' + str(repeated_source.size(0)))
    # logging.info('repeated_source.size(1): ' + str(repeated_source.size(1)))

    # Step 4: Calculate the remaining elements that need to be filled in the target tensor
    remaining_elements = target_tensor.size(1) - source_tensor.size(1) - repeated_source.size(1) - reserved_pos
    # logging.info('remaining_elements:  ' + str(remaining_elements))

    # Step 5: Slice and assign the remaining elements from the repeated source to the target tensor
    # Put source tensor at the very beginning
    target_tensor[:, 0:source_tensor.size(1)] = source_tensor[:, :]
    # Put the repeated tensor after the source tensor and reserved tensor
    target_tensor[:, (source_tensor.size(1) + reserved_pos):(target_tensor.size(1) - remaining_elements)] = repeated_source[:, :]
    # Fill the rest of tensor
    target_tensor[:, (target_tensor.size(1) - remaining_elements):] = repeated_source[:, :remaining_elements]

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
