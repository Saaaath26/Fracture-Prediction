"""Training acknowledgement shared by YOLO, relationship and neural entry points."""


def require_training(authorized):
    if authorized is not True:
        raise PermissionError(
            "Training disabled: explicit --authorize-training is required"
        )
