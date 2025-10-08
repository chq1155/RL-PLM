import webdataset as wds
import functools
import math
import webdataset as wds
from tools.data_utils import *


def get_pifold_dataset(args, tokenizer, epoch=0, floor=False):
    """
    param: args
    param: tokenizer: ProGen Tokenizer
    """

    input_shards = args.pifold_shards
    # if args.rank == 0:
    #     print("input_shards", input_shards)
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    n_shards = len(input_shards)

    num_samples, num_shards = get_dataset_size(input_shards[0])
    # num_samples = num_samples * n_shards
    num_shards = num_shards * n_shards
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples * n_shards
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_image_fn = preprocess_pifold_emb
    preprocess_text_fn = functools.partial(preprocess_pifold_seq, tokenizer=tokenizer, lm_type="progen")

    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=2000,
                    initial=500,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                # wds.split_by_node,
                # wds.split_by_worker,
            ]
        )
        
    pipeline.extend(
        [
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=2000,
                initial=500,
                rng=random.Random(args.seed)
                                  
            ),
        ]
    )

    def preprocess(sample):
        sample = sample[0] #TODO: embeding here!
        # print(sample['coords'].shape[0] == len(sample['seq']))
        return sample['pifold_emb'], sample['seq'], sample['coords'], sample['seq'] #TODO: change embedding here!

    pipeline.extend(
        [
            wds.decode(handler=log_and_continue),
            wds.to_tuple("pyd", handler=log_and_continue),
            wds.map(preprocess, handler=log_and_continue),
            wds.batched(args.batch_size_pifold,
                        partial=False,
                        collation_fn=functools.partial(wds.filters.default_collation_fn, combine_tensors=False)),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)

    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_pifold * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    # dataset = dataset.with_epoch(num_worker_batches)
    return dataset
    # dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

# class args:
#     rank = 0
#     pifold_shards = ["/root/llava_pro/esmfold_shards/shard-000%.03d.tar" % i for i in range(0,100)]
#     train_num_samples = 1000
#     lm_path = "progen"
#     seed = 2023
#     batch_size_pifold = 16
#     workers = 1
#     world_size = 1

# from transformers import PreTrainedTokenizerFast
# from tokenizers.processors import TemplateProcessing
# from tokenizers import Tokenizer
# def create_tokenizer_custom(file):
#     with open(file, 'r') as f:
#         return Tokenizer.from_str(f.read())
# progen_tokenizer = '/root/progen2-base/tokenizer.json'
# progen_tokenizer = create_tokenizer_custom(progen_tokenizer)
# progen_tokenizer.post_processor = TemplateProcessing(
#     single="1 $A 2",
#     special_tokens=[("1", 3), ("2", 4)],
# ) 

# progen_tokenizer = PreTrainedTokenizerFast(tokenizer_object=progen_tokenizer)
# progen_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
# progen_tokenizer.add_special_tokens({'bos_token': '<|bos|>'})        
# progen_tokenizer.add_special_tokens({'eos_token': '<|eos|>'}) 
# progen_tokenizer.add_special_tokens({'bos_token': '1'})        
# progen_tokenizer.add_special_tokens({'eos_token': '2'}) 

# get_pifold_dataset(args, progen_tokenizer)
