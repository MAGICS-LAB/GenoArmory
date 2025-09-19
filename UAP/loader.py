import torch
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchvision import transforms, datasets


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # Define label list manually
    label_list = list(range(0,args.num_label))
    
    
    # Construct cache filename
    cached_features_file = os.path.join(
        args.data_dir,
        f"cached_{'test' if evaluate else 'train'}_{os.path.basename(args.model_name_or_path)}_{args.max_seq_length}_{task}"
    )
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = torch.load(cached_features_file)
    else:
        
        # Manually load dataset
        file_path = os.path.join(args.data_dir, "test.csv" if evaluate else "train.csv")
        examples = []
        lines = open(file_path, 'r', encoding='utf-8').readlines()[1:]
        for i, line in enumerate(lines):
            split = line.strip('\n').split(',')
            label = int(split[-1])
            seq = split[0]

            examples.append([seq, label])
                
        # Convert examples to features
        features = []
        
        for text_a, label in examples:
            inputs = tokenizer(
                text_a,
                max_length=args.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            if args.model_type=='bert':
                features.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "token_type_ids": inputs.get("token_type_ids", torch.zeros_like(inputs["input_ids"])),
                    "label": label
                })
            elif args.model_type == 'hyena': 
                features.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "label": label
                })
            else:
                features.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "label": label
                })


        
        torch.save(features, cached_features_file)
    
    # Convert to Tensors and build dataset
    if args.model_type=='bert':
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_attention_mask = torch.stack([f["attention_mask"] for f in features])
        all_token_type_ids = torch.stack([f["token_type_ids"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    elif args.model_type=='hyena':
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,  all_labels)
    else:
        all_input_ids = torch.stack([f["input_ids"] for f in features])
        all_attention_mask = torch.stack([f["attention_mask"] for f in features])
        all_labels = torch.tensor([label_list.index(f["label"]) for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
        
    return dataset


class DataSet(object):
    '''
    Base class for representing a dataset. Meant to be subclassed, with
    subclasses implementing the `get_model` function. 
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        Args:
            ds_name (str) : string identifier for the dataset
            data_path (str) : path to the dataset 
            num_classes (int) : *required kwarg*, the number of classes in
                the dataset
            mean (ch.tensor) : *required kwarg*, the mean to normalize the
                dataset with (e.g.  :samp:`ch.tensor([0.4914, 0.4822,
                0.4465])` for CIFAR-10)
            std (ch.tensor) : *required kwarg*, the standard deviation to
                normalize the dataset with (e.g. :samp:`ch.tensor([0.2023,
                0.1994, 0.2010])` for CIFAR-10)
            custom_class (type) : *required kwarg*, a
                :samp:`torchvision.models` class corresponding to the
                dataset, if it exists (otherwise :samp:`None`)
            label_mapping (dict[int,str]) : *required kwarg*, a dictionary
                mapping from class numbers to human-interpretable class
                names (can be :samp:`None`)
            transform_train (torchvision.transforms) : *required kwarg*, 
                transforms to apply to the training images from the
                dataset
            transform_test (torchvision.transforms) : *required kwarg*,
                transforms to apply to the validation images from the
                dataset
        """
        required_args = ['num_classes', 'mean', 'std', 
                        ]
        optional_args = ['custom_class', 'label_mapping', 'custom_class_args', 'transform_train', 'transform_test']

        missing_args = set(required_args) - set(kwargs.keys())
        if len(missing_args) > 0:
            raise ValueError("Missing required args %s" % missing_args)

        extra_args = set(kwargs.keys()) - set(required_args + optional_args)
        if len(extra_args) > 0:
            raise ValueError("Got unrecognized args %s" % extra_args)
        final_kwargs = {k: kwargs.get(k, None) for k in required_args + optional_args} 

        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(final_kwargs)
    
    def override_args(self, default_args, kwargs):
        '''
        Convenience method for overriding arguments. (Internal)
        '''
        for k in kwargs:
            if not (k in default_args): continue
            req_type = type(default_args[k])
            no_nones = (default_args[k] is not None) and (kwargs[k] is not None)
            if no_nones and (not isinstance(kwargs[k], req_type)):
                raise ValueError(f"Argument {k} should have type {req_type}")
        return {**default_args, **kwargs}

    def get_model(self, arch, pretrained):
        '''
        Should be overriden by subclasses. Also, you will probably never
        need to call this function, and should instead by using
        `model_utils.make_and_restore_model </source/robustness.model_utils.html>`_.

        Args:
            arch (str) : name of architecture 
            pretrained (bool): whether to try to load torchvision 
                pretrained checkpoint

        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError

    def make_loaders(self, workers, batch_size, data_aug=True, subset=None, 
                    subset_start=0, subset_type='rand', val_batch_size=None,
                    only_val=False, shuffle_train=True, shuffle_val=True, subset_seed=None):
        '''
        Args:
            workers (int) : number of workers for data fetching (*required*).
                batch_size (int) : batch size for the data loaders (*required*).
            data_aug (bool) : whether or not to do train data augmentation.
            subset (None|int) : if given, the returned training data loader
                will only use a subset of the training data; this should be a
                number specifying the number of training data points to use.
            subset_start (int) : only used if `subset` is not None; this specifies the
                starting index of the subset.
            subset_type ("rand"|"first"|"last") : only used if `subset is
                not `None`; "rand" selects the subset randomly, "first"
                uses the first `subset` images of the training data, and
                "last" uses the last `subset` images of the training data.
            seed (int) : only used if `subset == "rand"`; allows one to fix
                the random seed used to generate the subset (defaults to 1).
            val_batch_size (None|int) : if not `None`, specifies a
                different batch size for the validation set loader.
            only_val (bool) : If `True`, returns `None` in place of the
                training data loader
            shuffle_train (bool) : Whether or not to shuffle the training data
                in the returned DataLoader.
            shuffle_val (bool) : Whether or not to shuffle the test data in the
                returned DataLoader.

        Returns:
            A training loader and validation loader according to the
            parameters given. These are standard PyTorch data loaders, and
            thus can just be used via:

            >>> train_loader, val_loader = ds.make_loaders(workers=8, batch_size=128) 
            >>> for im, lab in train_loader:
            >>>     # Do stuff...
        '''
        transforms = (self.transform_train, self.transform_test)
        return loaders.make_loaders(workers=workers,
                                    batch_size=batch_size,
                                    transforms=transforms,
                                    data_path=self.data_path,
                                    data_aug=data_aug,
                                    dataset=self.ds_name,
                                    label_mapping=self.label_mapping,
                                    custom_class=self.custom_class,
                                    val_batch_size=val_batch_size,
                                    subset=subset,
                                    subset_start=subset_start,
                                    subset_type=subset_type,
                                    only_val=only_val,
                                    seed=subset_seed,
                                    shuffle_train=shuffle_train,
                                    shuffle_val=shuffle_val,
                                    custom_class_args=self.custom_class_args)


class DNA(DataSet):
    def __init__(self, args, **kwargs):
        """
        """
        ds_kwargs = {
            'num_classes': args.num_label,
            'mean': torch.tensor([0.485, 0.456, 0.406]),
            'std': torch.tensor([0.229, 0.224, 0.225]),
            'custom_class': None,
            'label_mapping': None,
        }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(DNA, self).__init__('dna', args, **ds_kwargs)

    def get_model(self, arch):
        """
        """
        return arch

    def make_loaders(self, args, tokenizer, workers, batch_size, data_aug=True, subset=None, 
                    subset_start=0, subset_type='rand', val_batch_size=None,
                    only_val=False, shuffle_train=True, shuffle_val=True, subset_seed=None):
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader
