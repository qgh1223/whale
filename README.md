# EAD2019 challenge 

## Video endoscopy image data and labels

[1] First training set data 

        Release date: 15th November 2018

- Number of video frames/images = 887
    - Data institutions: JRC, UK/ICL, France/ APHP, France/ UHV, Switzerland/Botkin Clinical City Hospital, Russia
    - Data modalities: White light/Fluorescence/Narrow-band imaging
    - Data resolutions: Standard HD/Full HD/high magnification and others
    - Tissue types: Oesophagus, stomach, pylorus, rectum, anus , colon, bladder...


[2] Second training set data

```TO BE DECLARED!!!```


[3] Test data release

``` TO BE DECLARED ``` (Tentative in Febraury)


See for updated at [EAD2019](https://ead2019.grand-challenge.org/Data/) 

## Annotation

- Annotation provided are for 7 classes and consists of intotal 9352 bounding boxes (bbox)

- Distribution of class specific bboxes in the provided dataset is presented in the bar graph

    ![image](challenge_phase_I_data_classes.png)

- Bounding box labels has been provided in the below format:
    
        <object-class> <x> <y> <width> <height> : Ideally, <0> <0> <1> <1>

        <object-class> integer values from 0 to 6
        <x> = <abs x>/<image_width>, <y> = <abs y>/<image_height> they correspond to annotators box's centres
        <width>: annotator box normalised width 
        <height> annotator box normalised height

- Annotation converter to corresponding non-normalized format as csv file is provided as software tools

- class labels --> 0: specularity ..... 6:instrument

## Software tools

- Please visit [github-repos](https://github.com/sharibox/EAD2019) for some help



## Terms and conditions of data usage:

[1] All data provided in this challenge is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)

[2] Image data can be used for research purposes only. Utilization of any data for a product or (self) publication without consent and proper legal agreement from the EAD2019 organisers is forbidden!!!

[3] See [EAD2019 website](https://ead2019.grand-challenge.org/Rules/) for challenge participation rules
