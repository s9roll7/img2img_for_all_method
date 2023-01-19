# img2img_for_all_method

## Overview
#### AUTOMATIC1111 UI extension to find the best settings for running img2img.

## Example

#### from Realistic to Animated (Euler a)
![from Realistic to Animated sample](sample/case1/0000_Euler_a.png "from Realistic to Animated sample")

#### from Animated to Realistic (DPM++ SDE Karras)
![from Animated to Realistic sample](sample/case2/cyber_DPM++_SDE_Karras.png "from Animated to Realistic sample")

## Installation
- Use the Extensions tab of the webui to [Install from URL]

## Usage
- Go to img2img and load your base image
- Choose "img2img for all method" from the scripts select
- Create an empty directory for input and Fill in the [Input directory] field
- Create an empty directory for output and Fill in the [Output directory] field
- Put the original image in [Input directory]. Multiple images are allowed. (Start with just a couple of images at first.)
- Generate
(In the current latest webui, it seems to cause an error if you do not drop the image on the main screen of img2img.  
Please drop the image as it does not affect the result.)

## Options
- "Sampling method" ... The method used by img2img. If "All" is selected, all sampling methods are used.
- "[Sampling Steps, CFG Scale] List" ... Combination of Sampling Steps, CFG Scale. Set up in the format [Sampling Steps A, CFG Scale B], [Sampling Steps C, CFG Scale D] ...
- "Denoising Strength List" ... List of Denoising Strength. Set up in the format Denoising Strength A, Denoising Strength B, Denoising Strength C ...
- "Img2Img Repeat Count" ... This is the number of times img2img is run again with the exact same settings, using the output of img2img as input.
- "Add N to seed when repeating" ... Value to be added to seed when repeating by setting repeat count. There seems to be a big difference in results between 0 and other values.

## Warning
Depending on the configuration, it may take a long time to execute a very large number of img2img's.  
3 input images  
Sampling method = All  
[Sampling Steps, CFG Scale] List = [20,7],[50,20]  
Denoising Strength List = 0.1, 0.25, 0.35  
Img2Img Repeat Count = 3  
As an example, the above configuration will execute img2img 972 times  
3 x 18 x 2 x 3 x 3 = 972


