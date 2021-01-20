Code for GIDGAN.


Prepare data :
Run "main.m" (the first function) to convert source images from RGB color space to YCbCr.

To train :
Put training image pairs (Y channel) in the "Train_near" and "Train_far" folders, and run "python main.py" to train the network.

To test :
Put test image pairs (Y channel) in the "Test_near" and "Test_far" folders, and run "python test.py" to test the trained model. You can also directly use the trained model we provide.

Restore the output of networks to RGB space :
Run "main.m" (the second function) to restore the output of networks to RGB color space.
