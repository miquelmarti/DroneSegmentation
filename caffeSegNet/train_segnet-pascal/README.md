Many tests here :

camvid_arch_weighting : test with the same train_val and deploy as the one for camvid, and with weighting the classes (get them with get_class_weighting)
    -> the loss decreases a little, but is not stabilized at all

camvid_arch_no_weighting : test with the same train_val and deploy as the one for camvid, and without weighting the classes
    -> the loss decreases (but don't stabilize yet). The training is unfinished, but good

dropout_weighting : use the segnet architecture provided for pascal, with the weighting per classes (get them with get_class_weighting)
    -> maybe the worst, the loss seems to decrease but nothing sure, and the results are quite bad

dropout_no_weighting : use the segnet architecture provided for pascal, without the weighting per classes
    -> the loss decreases REALLY slowly... some results, nothing perfect


