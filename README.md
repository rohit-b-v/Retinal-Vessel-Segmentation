# Retinal-Vessel-Segmentation
Segmentation of retinal scans using GANs

Project:

Retinal vessel segmentation

This project presents a deep learning model for retinal vessel segmentation using a task-driven Generative Adversarial Network (GAN). Retinal vessels provide crucial diagnostic information for various diseases, but manual segmentation is labor-intensive and often affected by image noise and low contrast. To address these issues, the proposed model automates the segmentation process using a GAN framework trained with perceptual loss to closely mimic expert-annotated segmentations.

The model architecture consists of a U-Net-based generator and three discriminators with varying receptive fields: a global discriminator and two multi-scale discriminators. These discriminators evaluate the quality of the generated segmentations by comparing them to manual annotations. The use of perceptual loss, computed from features extracted by a pre-trained LeNet model, ensures that the generated images preserve high-level structural similarity to the reference images. This combination allows the model to effectively highlight even the faintest retinal vessels.

Before training, input images undergo preprocessing steps including grayscale conversion, histogram equalization, and gamma adjustment, which improve image clarity and accelerate training. The model is trained and evaluated on two widely-used datasets: DRIVE, containing images from diabetic retinopathy screening, and STARE, which includes images with various lesions. These datasets allow the model to generalize across different retinal image conditions.

Results demonstrate that the proposed method surpasses traditional U-Net and GAN approaches in terms of vessel visibility and segmentation accuracy. It performs particularly well in challenging scenarios where vessels are thin or low in contrast. Although pixel-to-pixel accuracy is commonly used to evaluate segmentation models, the project highlights its limitations and calls for the development of better performance metrics that account for structural correctness.

Future directions include creating models that can classify retinal images as healthy or defective, and refining evaluation methods to better compare segmentation architectures. Overall, this task-driven GAN approach proves to be a powerful tool for retinal image analysis, combining precision with automation in a clinically relevant setting.