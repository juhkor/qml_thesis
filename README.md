# qml_thesis
Source codes and extra material for masters thesis

NOTE: because of the 100 MB file size limit imposed by GitHub we were unable to push datasets that had full sized Mel spectrograms, so some of the codes will fail to run.

# QML_codes.ipynb
Following is a short description of the structure of the notebook.
1. Sources of the original.
2. Installation of preliminary dependencies for data fetching and preparing.
3. Function to prepare dataset.
4. Function to changing the sampling rate. This was not used in finished work.
5. Functions to encode and decode the numerical value to true label and vice versa.
6. Installation of QCNN dependencies.
7. Limited labels, which were used in the article that our work bases.
8. Mounting of Google Drive.
9. LSTM model which was used in all our experiments. Reducing of model parameters was implemented by adding a divider for the node counts.
10. Mel spectrogram function. Finished work used sampling rate of 16 000, but we did some tests with limited halved sampling rate.
11. Functions to generate quantum circuits, perform quanvolution with those circuits and save the results to datasets. Note that all of the datasets are not mentioned here and that all of the mentioned datasets are not available, because of GH limitations.
12. Draws a circuit image.
13. Draws alternative circuit image.
14. Functions to perform classic convolution. Same dataset availability note applies here as well.
15. Functions to split and alter the dataset. Generate Mel spectrograms, perform quanvolution or convolution and split to training and validation datasets. Availability limits apply here.
16. Loader functions for pre-made datasets. Availability limits apply here.
17. Function that reduces number of commands.
18. Prints label distribution.
19. Call to reduce function that will equalize the number of samples per command. Not really useful here, because reduce function is much lower in the code.
20. Training function and some checkpoint objects.
21. Seed experiments along with maximum validation accuracies for without / with quanvolution.
22. Training function for the models. Take a note of the history object names.
23. Alternative training function that was used to obtain timed evaluations.
24. Functions to create and plot confusion matrices. Also includes removing of command with smallest error rate and saving the reduced dataset. Last part saves the relevant statistics to be used later.
25. Plotting function that was used at first, but later implemented in another file that was ran locally.
26. Prints some statistics.
27. Datapoints reduce function that was used in part 19.
28. Making sure that data arrays are of equal length.
29. Training function for data reduced models.
30. Plotting function for data reduced models. Later implemented locally.
31. Training function for parameter reduced models.
32. Plotting function for parameter reduction. Later implemented locally.
33. Statistics printing for data and parameter reductions.
34. Helper function to perform quanvolution of binary images.
35. Binary images and optimal parameter search experiments.

# avg.py
Prints statistics of the seed tests.

# confusion.py
This does not work, because it tries to load full datasets, which could not be provided. With datasets available this would make predictions by using saved models and save plots of the resulting confusion matrices.

# confusion_plotter.py
This contains arrays of all the label reduction experiments confusion matrices. This can be used to create confusion plots and to print some statistical plots of the results.

# data.py
Prints statistics from the label reduction experiments, but file name have to be changed manually.

# example.py
Creates example plots that were in Chapter 2 of thesis.

# label_reduce.py
Original implementation of label reducing functionality, included in Jupyter notebook as well. Calculates the error rates from both models and removes the command that had smallest error rate.

# plotter.py
This creates all the plots from the training histories of every model that we experimented with. Also prints numerical statistics from the training histories. To print plots uncomment the relevant lines at the end of the file.

# plotting.py
Creates plot from the binary image experiment.

# test.py
Original implementation of the confusion matrix creation function. Creates small confusion matrix as a proof of concept.



