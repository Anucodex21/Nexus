from torch.utils.data import DataLoader

class LLMDataLoader:
    """Custom data loader for LLM training."""

    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    @staticmethod
    def collate_fn(batch, tokenizer):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [tokenizer.pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask
        }
