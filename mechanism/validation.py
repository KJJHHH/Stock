def walk_forward_splits(n_samples: int, train_size: int, test_size: int, embargo: int = 0):
    """Yield (train_idx, test_idx) for purged walk-forward validation."""
    start = 0
    while True:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        train_idx = range(start, train_end)
        test_idx = range(test_start, test_end)
        yield train_idx, test_idx
        start += test_size
