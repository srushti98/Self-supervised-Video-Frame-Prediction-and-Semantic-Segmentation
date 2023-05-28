from .moving_objects import load_moving_object

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    return load_moving_object(batch_size, val_batch_size, data_root, num_workers)
