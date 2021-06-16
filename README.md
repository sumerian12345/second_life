# SECOND LIFE
-----

[Second life]("Project website") // [Clip]("Clip website?") // Article?

- Image - ? Logo?

[Sumer Matharu](), [Aleksandra Jastrzębska](), [Jonathan Hernández López]() and [Jaime Cordero Cerrillo]()
IAAC, 2021.

## Abstract
-----

This research focuses on one of the main problems in the AEC industry: the huge amount of waste it produces. As a contribution in the way to solve it, this tool classifies recycled elements which can have a second life, by learning visual concepts to finally match them with the search criteria that the user specifies.

## Datasets
-----

The datasets that were used containing elements scraped from Google Images are the following:

 - `Doors`: Trained for condition, color, style, opening and structure.
 - `Flooring`: Trained for material, condition and color.
 - `Windows`: Trained for style, transparency and condition.
 - `Used lumber`: Trained for use.

## Training
-----

This project uses two models:

 - Clip AI: for classifying uploaded images considering the features and assign tags to them.
 - Minisom: for mapping the images from the dataset and offering the user a range of images based on the one he previously selected.

## Training
-----

The tool uses the pre-trained model from Clip AI to classify the images. Each of the datasets were trained on the aformentioned features, so that it is able to assign a tag to each image for each of the trained features.

<img src="imgs/training.jpg" width="900px"/>