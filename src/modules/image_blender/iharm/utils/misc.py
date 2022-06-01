import torch


def load_weights(model, path_to_weights):

    current_state_dict = model.state_dict()
    new_state_dict = torch.load(str(path_to_weights), map_location='cpu')
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict)
