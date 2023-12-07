def calculate_phc(batch, y_pred):
    """

    :param batch: y_true : B x M x 6
    :param y_pred: List[B] -> K x 7
    :return:
    """
    _, points, y_true, transforms = batch