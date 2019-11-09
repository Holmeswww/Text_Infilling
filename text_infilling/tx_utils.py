"""
This script is adapted from the tensor2tensor repositor.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

__all__ = [
    "embedding_to_padding",
    "_bucket_boundaries",
    "_batching_scheme",
    "smoothing_cross_entropy",
    "prepare_template",
    "fill_template",
    "generate_prediction_offsets",
    "generate_prediction_segment_ids",
    "update_template_pack"
]

def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.
    We have hacked symbol_modality to return all-zero embeddings for padding.
    Args:
        emb: a Tensor with shape [..., depth].
    Returns:
        a float Tensor with shape [...].
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))
def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def _batching_scheme(batch_size,
                     max_length,
                     min_length_bucket,
                     length_bucket_step,
                     drop_long_sequences=False,
                     shard_multiplier=1,
                     length_multiplier=1,
                     min_length=0,
                     batch_relax=False):
  """A batching scheme based on model hyperparameters.
  Every batch containins a number of sequences divisible by `shard_multiplier`.
  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.
  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example
  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3

  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret

def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False,
                            zero_pad=True):
    """Cross entropy with label smoothing to limit over-confidence.
    Args:
        logits: Tensor of size [batch_size, ?, vocab_size]
        labels: Tensor of size [batch_size, ?]
        vocab_size: Tensor representing the size of the vocabulary.
        confidence: Used to determine on and off values for label smoothing.
            If `gaussian` is true, `confidence` is the variance to the gaussian
            distribution.
        gaussian: Uses a gaussian distribution for label smoothing
        zero_pad: use 0 as the probabitlity of the padding
            in the smoothed labels. By setting this, we replicate the
            numeric calculation of tensor2tensor, which doesn't set the
            <BOS> token in the vocabulary.
    Returns:
        the cross entropy loss.
    """
    with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
        # Low confidence is given to all non-true labels, uniformly.
        if zero_pad:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 2)
        else:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)

        if gaussian and confidence > 0.0:
            labels = tf.cast(labels, tf.float32)
            normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
            soft_targets = normal_dist.prob(
                tf.cast(tf.range(vocab_size), tf.float32)\
                    [:, None, None])
            # Reordering soft_targets from [vocab_size, batch_size, ?]
            # to match logits: [batch_size, ?, vocab_size]
            soft_targets = tf.transpose(soft_targets, perm=[1, 2, 0])
        else:
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence,
                dtype=logits.dtype)
        if zero_pad:
            soft_targets = tf.concat([tf.expand_dims(\
                tf.zeros_like(labels, dtype=tf.float32), 2),\
                soft_targets[:, :, 1:]], -1)

        if hasattr(tf.nn, 'softmax_cross_entropy_with_logits_v2'):
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        else:
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
        logits=logits, labels=soft_targets)


def parse_segment(lengths, masks):
    def _parse_segment(lengths, masks):
        """
        mask:        [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
        segment_ids: [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4],
                      [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4]] <- start from 0
        offsets:     [[0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0],
                      [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0]]
        :param masks:
        :return: segment_ids, offsets
        """
        segment_ids = np.full_like(masks, 0)
        offsets = np.full_like(masks, 0)
        batch_size = masks.shape[0]
        for i in range(batch_size):
            mask = masks[i]
            segment_ids[i][0] = 0
            for j in range(1, lengths[i]):
                if mask[j] == mask[j-1]:
                    segment_ids[i][j] = segment_ids[i][j-1]
                    offsets[i][j] = offsets[i][j-1] + 1
                else:
                    segment_ids[i][j] = segment_ids[i][j-1] + 1
                    offsets[i][j] = 0
        return segment_ids, offsets

    return tf.py_func(_parse_segment, [lengths, masks], [masks.dtype, masks.dtype])


def _pad_array_list(arrays, lens, pad_id):
    """
    :param ar: a list of 1-D array of different lengths, [batch_size, unfixed length]
    :return: a 2-D array, [batch_size, max_seq_len_in_original_list]
    """
    rst = []
    max_len = np.amax(lens)
    for idx, ar in enumerate(arrays):
        rst.append(np.pad(ar, (0, max_len - lens[idx]),
                          'constant', constant_values=pad_id))
    return np.array(rst), max_len


def _parse_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    """
    :param inputs:
    :param masks:
    :param start_positions: [batch_size, mask_num]
    :param end_positions:
    :param mask_id:
    :return:
    """
    inputs = inputs.tolist()
    masks = masks.tolist()
    l = len(inputs[0])
    rst, mask_rst, template_len = [], [], []
    for input, mask, start_pos_, end_pos_ in zip(inputs, masks, start_positions, end_positions):
        start_pos = [0]
        start_pos.extend(end_pos_.tolist())
        end_pos = start_pos_.tolist()
        end_pos.append(l)
        tmp_rst, tmp_mask = [], []
        for s, e in zip(start_pos, end_pos):
            tmp_rst.extend(input[s:e])
            tmp_rst.append(mask_id)
            tmp_mask.extend(mask[s:e])
            tmp_mask.append(1)
        tmp_rst.pop()  # delete the last mask_id
        tmp_mask.pop()
        rst.append(tmp_rst)
        mask_rst.append(tmp_mask)
        template_len.append(len(tmp_rst))
    rst, _ = _pad_array_list(rst, template_len, pad_id)
    mask_rst, _ = _pad_array_list(mask_rst, template_len, pad_id)
    return rst, mask_rst


def _prepare_squeezed_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    templates, template_masks = \
        tf.py_func(_parse_template,
                   [inputs, masks, start_positions, end_positions, mask_id, pad_id],
                   [tf.int64, tf.int64])
    batch_size = tf.shape(inputs)[0]
    templates = tf.reshape(templates, shape=tf.stack([batch_size, -1]))
    template_masks = tf.reshape(template_masks, shape=tf.stack([batch_size, -1]))
    return templates, template_masks


def generate_equal_length_mask(inputs, lengths, mask_num, mask_len, mask_id, eoa_id, pad_id):
    """
    inputs and lengths are numpy arrays!
    mask_num = 2, having two masked out segment
    mask_length = 2, min length of each masked out segment
    inputs:[[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
            [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]]
    mask:  [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
    """
    def _parse_answer(inputs, start_pos, end_pos, eoa_id):
        rst = None
        batch_size = inputs.shape[0]
        for i in range(batch_size):
            tmp_answer = np.append(inputs[i, start_pos[i]:end_pos[i]], eoa_id)[np.newaxis, :]
            rst = tmp_answer if rst is None else np.concatenate((rst, tmp_answer), axis=0)
        return rst

    # TODO(wanrong): OUT-OF-RANGE bound check, tf.argmin(length) >= masknum * (1 + mask_length)
    def _fill_mask(mask_not_generated, mask_length, prev_end_pos,
                   lengths, masks, start_positions, end_positions):
        """
        :param mask_not_generated: number of mask not generated(excluding this one)
        :param prev_end_pos: open range
        :param masks:
        :return: after loop start_positions will be of shape [max_num, batch_size]
        """
        cur_start_pos = np.full_like(prev_end_pos, 1)
        batch_size = cur_start_pos.shape[0]
        for i in range(batch_size):
            cur_start_pos[i] = \
                np.random.randint(prev_end_pos[i] + 1,
                                  lengths[i] - mask_not_generated * (1 + mask_length) + 1,
                                  size=1)
        cur_end_pos = cur_start_pos + mask_length
        if start_positions.size == 0:
            start_positions = cur_start_pos[np.newaxis, :]
            end_positions = cur_end_pos[np.newaxis, :]
        else:
            start_positions = \
                np.concatenate((start_positions, cur_start_pos[np.newaxis, :]), axis=0)
            end_positions = \
                np.concatenate((end_positions, cur_end_pos[np.newaxis, :]), axis=0)

        for i in range(batch_size):
            masks[i, cur_start_pos[i]: cur_end_pos[i]] = 1
        return mask_not_generated - 1, cur_end_pos, masks, start_positions, end_positions

    mask_id = tf.Variable(mask_id, dtype=tf.int64)
    pad_id = tf.Variable(pad_id, dtype=tf.int64)
    eoa_id = tf.Variable(eoa_id, dtype=tf.int64)
    mask_not_generated = tf.Variable(mask_num, dtype=tf.int64)
    mask_length = tf.Variable(mask_len, dtype=tf.int64)
    prev_end_pos = tf.ones_like(inputs)[:, 0]
    start_positions = tf.Variable([], dtype=tf.int64)
    end_positions = tf.Variable([], dtype=tf.int64)
    masks = tf.zeros_like(inputs)
    answers = []
    for i in range(mask_num):
        mask_not_generated, prev_end_pos, masks, start_positions, end_positions = \
            tf.py_func(_fill_mask,
                       [mask_not_generated, mask_length, prev_end_pos,
                        lengths, masks, start_positions, end_positions],
                       [tf.int64, tf.int64, prev_end_pos.dtype, tf.int64,
                        tf.int64])
        cur_answer = tf.py_func(_parse_answer,
                                [inputs, prev_end_pos - mask_length, prev_end_pos, eoa_id],
                                inputs.dtype)
        answers.append(cur_answer)

    templates, template_masks = \
        _prepare_squeezed_template(inputs, masks, tf.transpose(start_positions, perm=[1, 0]),
                                   tf.transpose(end_positions, perm=[1, 0]), mask_id, pad_id)
    return masks, answers, templates, template_masks


def generate_dynamic_mask(inputs, lengths, present_rate, mask_id, boa_id,
                          eoa_id, pad_id, partition_num):
    def _fill_mask(inputs, lengths, present_rate, eoa_id, pad_id, partition_num):
        """
        The input batch has the same mask pattern, randoms through max_seq_length in lengths.
        :param inputs:
        :param lengths:
        :param present_rate:
        :return: answers: a tensor of shape [batch_size, sum(unfixed_answer_len for each ans)]
        start_pos and end_pos marks out ranges for answers
        """
        def _fill_mask_py_func(inputs, lengths, present_rate, eoa_id, pad_id, partition_num):
            # TODO(wanrong): bound check
            def _get_split_pos(masked_num):
                # split masked_num into partition_num segments
                if masked_num <= 1:
                    return [1] * (partition_num - 1)

                splitted = np.array_split(range(masked_num), partition_num)
                split_positions = [a.size for a in splitted]
                for i in range(1, partition_num):
                    split_positions[i] += split_positions[i - 1]
                return np.insert(split_positions, 0, 0, axis=0)

            batch_size = inputs.shape[0]
            masked_nums = ((lengths - 2) * (1 - present_rate)).astype(np.int64)  # [batch_size]
            split_positions = \
                [_get_split_pos(masked_num) for masked_num in masked_nums]  # [batch_size, partition_num+1]

            # calculate the length of each mask segment
            mask_lengths = np.zeros(shape=(batch_size, partition_num), dtype=np.int64)
            left_len = np.zeros(shape=(batch_size, partition_num + 1), dtype=np.int64)  # add a -1 at the end
            for bid, split_position in enumerate(split_positions):
                for idx, (prev, cur) in enumerate(zip(split_position[:-1], split_position[1:])):
                    mask_lengths[bid][idx] = cur - prev
                left_len[bid][-1] = 0  # leave <EOS> unmasked
                for idx, cur_len in reversed(list(enumerate(mask_lengths[bid]))):
                    left_len[bid][idx] = left_len[bid][idx+1] + cur_len + 1
            left_len = left_len[:, :-1]  # remove last column

            # splitting
            start_positions = np.zeros(shape=(batch_size, 1))
            end_positions = np.zeros(shape=(batch_size, 1))
            answers = np.zeros((batch_size, 0))
            partitions = np.array([])
            masks = np.full_like(inputs, 0)
            after_pad_ans_lens = np.zeros(shape=partition_num)
            boa = np.full(shape=(batch_size, 1), fill_value=boa_id)
            for i in range(1, partition_num + 1):
                idx = i - 1  # ignore padding 0 in start/end_positions
                # get start and end position for current mask
                cur_start_pos = np.zeros(shape=(batch_size, 1), dtype=np.int64)
                cur_end_pos = np.zeros(shape=(batch_size, 1), dtype=np.int64)
                cur_answers = []
                for bid in range(batch_size):
                    s = end_positions[bid][idx] + 1
                    e = lengths[bid] - left_len[bid][idx] + 1
                    cur_start_pos[bid][0] = s + (e - s) / (partition_num + 1)
                    cur_end_pos[bid][0] = cur_start_pos[bid][0] + mask_lengths[bid][idx]
                    cur_answers.append(
                        np.append(inputs[bid][cur_start_pos[bid][0]:cur_end_pos[bid][0]], eoa_id))
                    # update mask
                    for j in range(cur_start_pos[bid][0], cur_end_pos[bid][0]):
                        masks[bid][j] = 1  # set masked element to 1
                start_positions = np.concatenate((start_positions, cur_start_pos), axis=1)
                end_positions = np.concatenate((end_positions, cur_end_pos), axis=1)

                # pad cur_answers to same length
                cur_padded_ans, cur_max_len = _pad_array_list(cur_answers, mask_lengths[:, idx], pad_id)
                cur_padded_ans = np.concatenate((boa, cur_padded_ans), axis=1)
                after_pad_ans_lens[idx] = cur_max_len
                answers = np.concatenate((answers, cur_padded_ans), axis=1)

                # generate current partition index
                cur_idx = np.full_like(cur_padded_ans[0], idx)
                partitions = np.concatenate((partitions, cur_idx), axis=0)

            return masks, start_positions[:, 1:].astype(np.int64),\
                   end_positions[:, 1:].astype(np.int64),\
                   answers.astype(np.int64), after_pad_ans_lens.astype(np.int64), \
                   mask_lengths.astype(np.int32), partitions.astype(np.int32)

        eoa_id = tf.Variable(eoa_id, dtype=tf.int64)
        present_rate = tf.Variable(present_rate, dtype=tf.float32)
        partition_num = tf.Variable(partition_num, dtype=tf.int64)
        return tf.py_func(_fill_mask_py_func,
                          [inputs, lengths, present_rate, eoa_id, pad_id, partition_num],
                          [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int32, tf.int32])

    masks, start_positions, end_positions, answers, after_pad_ans_lens, true_ans_lens, partitions = \
        _fill_mask(inputs, lengths, present_rate, eoa_id, pad_id, partition_num)
    answers = tf.dynamic_partition(data=tf.transpose(answers, perm=[1, 0]),  # [sum(lens), batch_size]
                                   partitions=partitions,
                                   num_partitions=partition_num)
    answers = [tf.transpose(ans, perm=[1, 0]) for ans in answers]
    mask_id = tf.Variable(mask_id, dtype=tf.int64)
    pad_id = tf.Variable(pad_id, dtype=tf.int64)
    templates, template_masks = \
        _prepare_squeezed_template(inputs, masks, start_positions, end_positions, mask_id, pad_id)

    return masks, answers, after_pad_ans_lens, true_ans_lens, templates, template_masks, \
           start_positions, end_positions


def generate_prediction_offsets(inputs, max_length):
    batch_size = tf.shape(inputs)[0]
    max_length = tf.cast(max_length, dtype=tf.int32)
    _, offsets = parse_segment(tf.fill([batch_size], max_length),
                               tf.fill([batch_size, max_length], 0))
    return tf.cast(offsets, dtype=tf.int64)


def generate_prediction_segment_ids(inputs, segment_id, max_length):
    batch_size = tf.shape(inputs)[0]
    return tf.cast(tf.fill([batch_size, tf.cast(max_length, dtype=tf.int32)], segment_id), dtype=tf.int64)


def _get_start_end_pos(mask_by_word, mask_id):
    def _get_start_end_pos_py_func(mask_by_word, mask_id):
        start_pos, end_pos = [[-2] for i in range(len(mask_by_word))], [[-2] for i in range(len(mask_by_word))]
        for idx, template in enumerate(mask_by_word):
            for i, word in enumerate(template):
                if word == mask_id:
                    if end_pos[idx][-1] == i:
                        end_pos[idx].pop()
                    else:
                        start_pos[idx].append(i)
                    end_pos[idx].append(i+1)
        return np.array(start_pos)[:, 1:].astype(np.int64), np.array(end_pos)[:, 1:].astype(np.int64)

    mask_id = tf.Variable(mask_id, dtype=tf.int64)
    return tf.py_func(_get_start_end_pos_py_func,
                      [mask_by_word, mask_id],
                      [tf.int64, tf.int64])


def prepare_template(data_batch, args, mask_id, boa_id, eoa_id, pad_id):
    """
    mask_id = 7
    pad_id = 6
    inputs:        [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]] <- a tensor
    mask:          [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
    masked_inputs: [[3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1], [2, 1, 4, 3, 7, 7, 5, 4, 7, 7, 5]]
    templates:     [[3, 5, 4, 7, 1, 3, 3, 7, 1], [2, 1, 4, 3, 7, 5, 4, 7, 5]]
    segment_ids:   [[1, 1, 1, 2, 3, 3, 3, 4, 5], [1, 1, 1, 1, 2, 3, 3, 4, 5]]
    answers:       [[[4, 2], [5, 1]],
                    [[2, 5], [3, 1]]] <- used as decode outputs(targets) in training
    :param masked_inputs:
    :param mask_id:
    :return: masked_inputs, segment_ids, answers
    """
    inputs = data_batch['text_ids']
    lengths = data_batch['length']
    masks, answers, after_pad_ans_lens, true_ans_lens, templates, template_masks,\
        start_positions, end_positions = \
        generate_dynamic_mask(inputs, lengths, args.present_rate, mask_id, boa_id,
                              eoa_id, pad_id, args.blank_num)

    template_lengths = tf.fill(tf.shape(lengths), tf.shape(templates)[1])
    template_segment_ids, template_offsets = \
        parse_segment(template_lengths, template_masks)
    all_masked_out = tf.cast(tf.fill(tf.shape(inputs), mask_id), dtype=tf.int64)
    masked_inputs = tf.where(tf.equal(masks, tf.ones_like(inputs)),
                             all_masked_out, inputs)
    template_pack = {
        'masks': masks,
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'templates': templates,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'template_lengths': template_lengths
    }

    answer_packs = []
    for idx, answer in enumerate(answers):
        mask_len = after_pad_ans_lens[idx] + 2  # has <eoa> and <boa>
        answer_segment_ids = generate_prediction_segment_ids(answer, idx * 2 + 1, mask_len)
        answer_offsets = generate_prediction_offsets(answer, mask_len)
        answer = tf.reshape(answer, shape=tf.stack([-1, mask_len]))
        lengths = tf.reshape(true_ans_lens[:, idx], shape=tf.stack([-1]))
        answer_packs.append({
            'text_ids': answer,
            'segment_ids': answer_segment_ids,
            'offsets': answer_offsets,
            'lengths': lengths
        })

    return template_pack, answer_packs


def _split_template(template, mask_start_positions, mask_end_positions):
    """
    template: [3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1]
    start_positions: [3, 8], starting positions of the masks
    end_positions: [5, 10], ending positions of the masks
    will be split into: [[3, 5, 4], [1, 3, 3], [1]]
    :param template: a list of numbers
    :return:
    """
    rst = []
    start_positions = [0] + mask_end_positions.tolist()
    end_positions = mask_start_positions.tolist() + [len(template)]
    for s, e in zip(start_positions, end_positions):
        rst.append(template[s: e])
    return rst


def _merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id):
    """
    template_segments: [[3, 5, 4], [1, 3, 3], [1]]
    fillings: [[4, 2], [2, 5]]
    rst: [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    :param template_segments:
    :param fillings:
    :return:
    """
    def _parse(id_list, eoa_id, pad_id, eos_id):
        rst = []
        for id in id_list:
            if id in [eoa_id, eos_id]:
                break
            elif id is not pad_id:
                rst.append(id)
        return rst

    template_segment_num = len(template_segments)
    filling_segment_num = len(fillings)
    assert template_segment_num == filling_segment_num or \
           template_segment_num == filling_segment_num + 1

    rst = []
    for i in range(filling_segment_num):
        rst.extend(template_segments[i])
        rst.extend(_parse(fillings[i], eoa_id, pad_id, eos_id))
    if template_segment_num > filling_segment_num:
        rst.extend(template_segments[-1])
    return rst


def fill_template(template_pack, predictions, eoa_id, pad_id, eos_id):
    """
    :param template: [batch_size, max_seq_len]
    :param mask: [batch_size, max_seq_len]
    :param predictions: a list of tensors
    :return:
    """
    def _transpose(a):
        """
        :param a: mask_num * batch_size * undefined_len
        :return: batch_size * mask_num * undefined_len
        """
        rst = []
        for _ in a[0]:
            rst.append([])
        for ar in a:
            for idx, sent in enumerate(ar):
                rst[idx].append(sent)
        return rst

    start_positions = template_pack['start_positions']
    end_positions = template_pack['end_positions']
    templates = template_pack['text_ids']
    templates = templates.tolist()
    predictions = [prediction.tolist() for prediction in predictions]  # mask_num * batch_size * undefined_len
    predictions = _transpose(predictions)
    rst = []
    for template, start_pos, end_pos, fillings in zip(templates, start_positions, end_positions, predictions):
        template_segments = _split_template(template, start_pos, end_pos)
        rst.append(_merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id))
    return rst


def update_template_pack(template_pack, filling, mask_id, eoa_id, pad_id):
    def _fill_segment(masked_by_word_template, filling, start_pos, end_pos, eoa_id, pad_id):
        def _fill_segment_py_func(masked_by_word_templates, fillings, start_pos, end_pos, eoa_id, pad_id):
            masked_by_word_templates = masked_by_word_templates.tolist()
            fillings = fillings.tolist()
            start_pos = start_pos.tolist()
            end_pos = end_pos.tolist()
            rst, length = [], []
            for template, filling, s, e in zip(masked_by_word_templates, fillings, start_pos, end_pos):
                try:
                    end_pos = filling.index(eoa_id)
                    filling = filling[:end_pos]
                except ValueError:
                    pass
                cur_rst = template[:s] + filling + template[e:]
                length.append(len(cur_rst))
                rst.append(cur_rst)
            rst, _ = _pad_array_list(rst, length, pad_id)
            return rst
        return tf.py_func(_fill_segment_py_func,
                          [masked_by_word_template, filling, start_pos, end_pos, eoa_id, pad_id],
                          tf.int64)

    eoa_id = tf.Variable(eoa_id, dtype=tf.int64)
    pad_id = tf.Variable(pad_id, dtype=tf.int64)
    masked_inputs = _fill_segment(template_pack['text_ids'], filling,
                                  template_pack['start_positions'][:, 0],
                                  template_pack['end_positions'][:, 0], eoa_id, pad_id)
    masks = tf.where(tf.equal(masked_inputs, mask_id * tf.ones_like(masked_inputs)),
                     tf.ones_like(masked_inputs), tf.zeros_like(masked_inputs))
    start_positions, end_positions = _get_start_end_pos(masked_inputs, mask_id)
    templates, template_masks = \
        _prepare_squeezed_template(masked_inputs, masks, start_positions, end_positions, mask_id, pad_id)
    template_lengths = tf.fill(tf.shape(template_pack['template_lengths']), tf.shape(templates)[1])
    template_segment_ids, template_offsets = \
        parse_segment(template_lengths, template_masks)
    return_pack = {
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'templates': templates,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'masks': masks,
        'template_lengths': template_lengths
    }
    return return_pack