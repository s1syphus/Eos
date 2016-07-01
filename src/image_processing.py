"""
    Based on:

    https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py

    Read and preprocess image data.

     Image processing occurs on a single image at a time. Image are read and
     preprocessed in parallel across multiple threads. The resulting images
     are concatenated together to form a single batch for training or evaluation.

    -- Provide processed image data for a network:
        inputs: Construct batches of evaluation examples of images.
        distorted_inputs: Construct batches of training examples of images.
        batch_inputs: Construct batches of training or evaluation examples of images.
    -- Data processing:
        parse_example_proto: Parses an Example proto containing a training example of an image.
    -- Image decoding:
        decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.
    -- Image preprocessing:
        image_preprocessing: Decode and preprocess one image for evaluation or training
        distort_image: Distort one image for training a network.
        eval_image: Prepare one image for evaluation.
        distort_color: Distort the color in one image for training.
"""

# This prevents tensorflow from freaking out
import numpy as np

import  tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('image_size', 299,
                            """Provide square images of this size.""")

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")

tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")

# Images are preprocessed asynchronously using multiple threads specified by
# --num_preprocss_threads and the resulting processed images are stored in a
# random shuffling queue. The shuffling queue dequeues --batch_size images
# for processing on a given Inception tower. A larger shuffling queue guarantees
# better mixing across examples within a batch and results in slightly higher
# predictive performance in a trained model. Empirically,
# --input_queue_memory_factor=16 works well. A value of 16 implies a queue size
# of 1024*16 images. Assuming RGB 299x299 images, this implies a queue size of
# 16GB. If the machine is memory limited, then decrease this factor to
# decrease the CPU memory footprint, accordingly.
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """
                            """Default is ideal but try smaller values, e.g. """
                            """4, 2 or 1, if host memory is constrained. See """
                            """comments in code for more details.""")


def inputs(dataset, batch_size=FLAGS.batch_size, num_preprocess_threads=None):
    """
       Generate batches of ImageNet images for evaluation.

       Args:
           dataset: instance of Dataset class specifying the dataset.
           batch_size: integer, number of examples in batch
           num_preprocess_threads: integer, total number of preprocessing threads but
            None defaults to FLAGS.num_preprocess_threads.

        Returns:
            images: Images. 4D tensor of size [batch_size, FLAGS.image_size, image_size, 3].
            labels: 1-D integer Tensor of [FLAGS.batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset, batch_size, train=False,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=1)
    return images, labels


def distorted_inputs(dataset, batch_size=FLAGS.batch_size, num_preprocess_threads=None):
    """
        Generate batches of distorted versions of ImageNet images.

        Use this function as the inputs for training a network.
        Distorting images provides a useful technique for augmenting the data
        set during training in order to make the network invariant to aspects
        of the image that do not effect the label.
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset, batch_size, train=True,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=FLAGS.num_readers)
    return images, labels


def decode_jpeg(image_buffer, scope=None):
    """
        Decode a JPEG string into one 3-D float image Tensor.

        Args:
            image_buffer: scalar string Tensor.
            scope: Optional scope for op_scope.

        Returns:
            3-D float Tensor with values ranging from [0, 1).
    """
    with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """
    Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for op_scope.

    Returns:
        color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, thread_id=0, scope=None):
    """
        Distort one image for training a network.

        Distorting images provides a useful technique for augmenting the dataset
        during training in order to make the network invariant to aspects
        of the image that do not effect the label.

        Args:
            image: 3-D float Tensor of image
            height: integer
            width: integer
            thread_id: integer indicating the preprocessing thread.
            scope: Optional scope for op_scope.

        Returns:
            3-D float Tensor of distorted image used for training.
    """
    with tf.op_scope([image, height, width], scope, 'distort_image'):
        # This resizing operation may distort the images because the aspect
        # ratio is not respected. We select a resize method in a round robin
        # fashion based on the thread number.
        # Note that ResizeMethod contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        distorted_image = tf.image.resize_images(image, height, width, resize_method)

        # This comment kept in case something breaks
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        distorted_image.set_shape([height, width, 3])
        if not thread_id:
            tf.image_summary('cropped_resized_image',
                             tf.expand_dims(distorted_image, 0))

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Randomly distort the colors.
        distorted_image = distort_color(distorted_image, thread_id)

        if not thread_id:
            tf.image_summary('final_distorted_image',
                             tf.expand_dims(distorted_image, 0))
        return distorted_image


def eval_image(image, height, width, scope=None):
    """
        Prepare one image for evaluation.

        Args:
            image: 3-D float Tensor
            height: integer
            width: integer
            scope: Optional scope for op_scope.

        Returns:
            3-D float Tensor of prepared image.
    """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image_buffer, train, thread_id=0):
    """
        Decode and preprocess one image for evaluation or training.
    """
    image = decode_jpeg(image_buffer)
    height = FLAGS.image_size
    width = FLAGS.image_size

    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    """
         Parses an Example proto containing a training example of an image.
    """
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.string,
                                                default_value='')
        }
    features = tf.parse_single_example(example_serialized, feature_map)

    return features['image/encoded'], features['image/class/label']


def batch_inputs(dataset, batch_size, train, num_preprocess_threads=None, num_readers=1):
    """
        I got lazy, write this at some point
    """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=12)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

        if num_preprocess_threads is None:
            num_preprocess_threads = FLAGS.num_preprocess_threads

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = FLAGS.num_readers

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * FLAGS.input_queue_memory_factor
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples,
                dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(
                capacity=examples_per_shard + 3 * batch_size,
                dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image_buffer, label = parse_example_proto(example_serialized)
                image = image_preprocessing(image_buffer, train, thread_id)
                images_and_labels.append([image, label])

            images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)
            # Reshape images into these desired dimensions.
            height = FLAGS.image_size
            width = FLAGS.image_size
            depth = 3

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[batch_size, height, width, depth])

            # Display the training images in the visualizer.
            tf.image_summary('images', images)

            return images, tf.reshape(label_index_batch, [batch_size])
