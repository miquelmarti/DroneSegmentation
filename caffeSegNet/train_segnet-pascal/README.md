Many tests here :

camvid_arch_no_weighting : test with the same train_val and deploy as the one for camvid, and without weighting the classes
    -> 

camvid_arch_weighting : test with the same train_val and deploy as the one for camvid, and with weighting the classes (get them with get_class_weighting)
    -> the loss doesn't decrease at all, but the training seems to work. Still not perfect thought

dropout_weighting_wrong_colors : use the segnet architecture provided for camvid, with the weighting per classes (get them with get_class_weighting), and with pascal_21_colors.png (see in VOC2012/colors/README why it can be wrong)
    -> The shape of the class is built correctly, but no colors... weird

