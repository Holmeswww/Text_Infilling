import tensorflow as tf
import texar.tf as tx
import os


FLAGS = tf.app.flags.FLAGS


def transform_input_with_is_missing_token(inputs, targets_present, mask_id):
    """
    Transforms the inputs to have missing tokens when it's masked out.  The
    mask is for the targets, so therefore, to determine if an input at time t is
    masked, we have to check if the target at time t - 1 is masked out.
    e.g.
    inputs = [a, b, c, d]
    targets = [b, c, d, e]
    targets_present = [1, 0, 1, 0]
    then,
    transformed_input = [a, b, <missing>, d]
    Args:
    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens
      up to, but not including, vocab_size.
    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with
      True representing the presence of the word.
    Returns:
    transformed_input:  tf.int32 Tensor of shape [batch_size, sequence_length]
      which takes on value of inputs when the input is present and takes on
      value=vocab_size to indicate a missing token.
    """
    # To fill in if the input is missing.
    input_missing = tf.fill(tf.shape(inputs), mask_id)

    # The 0th input will always be present.
    zeroth_input_present = tf.zeros_like(inputs)[:, 0][:, tf.newaxis]
    print(zeroth_input_present.shape)
    print(targets_present.shape)

    # Input present mask.
    inputs_present = tf.concat(
        [zeroth_input_present, targets_present[:, :-1]], axis=1)

    transformed_input = tf.where(tf.equal(inputs_present, tf.ones_like(inputs)),
                                 inputs, input_missing)
    return transformed_input


def prepare_data():
    """Downloads the PTB or COCO dataset
    """
    yelp_url = 'https://VegB.github.io/downloads/yelp.tgz'
    data_path = 'yelp_data/'

    if not tf.gfile.Exists(data_path):
        tx.data.maybe_download(yelp_url, './', extract=True)
        os.remove('yelp.tgz')


if __name__ == '__main__':
    prepare_data()
