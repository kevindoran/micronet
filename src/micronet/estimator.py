import tensorflow as tf

# Number of iterations to run on the TPU workers before returning control to the
# master (not sure if the terminology is correct here).
iterations_between_model_update = 50
checkpoints_max = 0


def create_estimator(gcloud_settings, model_dir, model_fn, batch_size):
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            # In the future, the tpu parameter might support lists.
            tpu=gcloud_settings.tpu_name,
            zone=gcloud_settings.tpu_zone,
            project=gcloud_settings.project_name)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        # Directory to save model parameters, graph etc. Also used as a source
        # directory when loading checkpoints.
        model_dir=model_dir,
        keep_checkpoint_max=checkpoints_max,
        session_config=tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(
            # The number of train steps running in TPU system before returning
            # to CPU host for each `Session.run`. This means that global step is
            # increased `iterations_per_loop` times in one `Session.run`. It is
            # recommended to be set as number of global steps between each
            # checkpoint.
            iterations_per_loop=iterations_between_model_update,
            # Deprecated: num_shards,
            # num_cores_per_replica:  Useful? Used for model parallelism.
            # per_host_input_for_training: No idea what this is.
            # initial_infeed_sleep_secs: delay for infeed thread. Useful to
            #     avoid issues if the model requires a long compilation time.
            # input_partition_dims: A nested list to describe the partition
            #       dims for all the tensors from input_fn(). The structure of
            #       input_partition_dims must match the structure of `features`
            #       and `labels` from input_fn(). The total number of partitions
            #       must match `num_cores_per_replica`. For example, if
            #       input_fn() returns two tensors: images with shape [N, H, W,
            #       C] and labels [N].  input_partition_dims = [[1, 2, 2, 1],
            #       None] will split the images to 4 pieces and feed into 4 TPU
            #       cores. labels tensor are directly broadcasted to all the TPU
            #       cores since the partition dims is `None`. Current
            #       limitations: This feature is only supported with the
            #       PER_HOST_V2 input mode.
            # eval_training_input_configuration: If `SLICED`, `input_fn` is only
            #       invoked once on host 0 and the tensors are broadcasted to
            #       all other replicas. Unlike
            #       per_host_input_for_training=BROADCAST, each replica will
            #       only get a slice of the data instead of a whole copy. If
            #       `PER_HOST_V1`, the behaviour is determined by
            #       per_host_input_for_training.
        )
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        # A function that  returns an EstimatorSpec or TPUEstimatorSpec.
        model_fn=model_fn,
        use_tpu=True,
        config=run_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        # model_dir=None, # Inherited from runConfig.
        # predict_batch_size=FLAGS.batch_size, possibly needed. Put if I use
        # params={'key':'value'}. Optional. Don't need it yet.
        # prediction on the CPU, then we don't need TPU prediction.
        # export_to_cpu=True, I might need this option for prediction.
        # batch_axis: not sure how to use this.
    )
    return estimator

