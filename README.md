# Vision Transformers in Medical Imaging: Brain Tumor Classification

In this project, we will investigate the effectiveness of Vision Transformers (ViT) in tackling challenging problem of medical imaging using the Brain Tumor dataset. While Convolutional Neural Networks (CNNs) have long been the standard for computer vision tasks, ViT offers a novel approach by leveraging self-attention mechanisms derived from the Transformer architecture, which was initially developed for natural language processing.

<div align="center">
  <img src="./brain_tumor.jpg" alt="Brain Tumor Introduction" width=700>
</div>

# Vision Transformer (ViT)

A Vision Transformer (ViT) is an advanced deep learning model that adapts the Transformer architecture—originally developed for processing sequential data such as text—to analyze images. Instead of relying on traditional convolutional operations, ViT segments an image into uniform patches and converts them into numerical representations. These representations are then processed as sequential data through Transformer layers, allowing the model to effectively learn intricate patterns and relationships at both local and global levels within the image.

<div align="center">
  <img src="./ViT Architecture.png" alt="ViT Architecture">
</div>

In this project, we have developed a specialized Vision Transformer (ViT) model for classifying brain tumors. The architecture begins with a patch embedding layer that segments the input images into fixed-size patches, followed by a series of Transformer encoder layers. By leveraging self-attention mechanisms, the model effectively captures complex spatial interactions across different regions of brain scans, enhancing its ability to distinguish between tumor types with high accuracy.


# Data Exploration and Visualization

The Brain Tumor dataset is a collection of brain MRI images from patients both with and without brain tumors. Each image poses a distinct challenge because of differences in size, resolution, and contrast. Our objective is to utilize the ViT architecture to create a strong classification model capable of accurately detecting brain tumors in these MRI scans. 
Our dataset consists of 7022 images, which are split as follows:

```python
Found 5712 files belonging to 4 classes.
Using 4570 files for training.
Found 5712 files belonging to 4 classes.
Using 1142 files for validation.
Found 1311 files belonging to 4 classes.
```
```
Sample Image Shape: (512, 512, 3)
```
## Sample labeled images:

<div align="center">
  <img src="./samples.png" alt="MRI Samples">
</div>


# ViT Model

<table style="width: 90%; table-layout: fixed;">
  <tr>
    <td align="center">
      <img src="./before.png" alt="Before" width="478.5">
    </td>
    <td align="center">
      <img src="./after.png" alt="After" width="478.5">
    </td>
  </tr>
  <tr>
    <td align="center" style="background-color: #333; color: white; padding: 10px; border-radius: 5px;">
      <strong>Before Patching</strong>
    </td>
    <td align="center" style="background-color: #333; color: white; padding: 10px; border-radius: 5px;">
      <strong>After Patching</strong> 
    </td>
  </tr>
</table>




## Test Predictions

We use the Adam optimizer with a learning rate of 0.0001 and weight decay of 0.0001. The model is trained with a batch size of 32 for 100 epochs. During training, a periodic evaluation on a test set is performed to track performance metrics including loss, accuracy, and confusion matrix. These metrics are closely monitored to assess the model’s performance and ensure accurate classification across the four classes: glioma, meningioma, notumor, and pituitary.

<div align="center">
  <img src="./predictions.png" alt="Predictions" width=600>
</div>

## Accuracy and Loss Curves
<div align="center">
  <img src="./accuracy.png" alt="Accuracy" width="45%">
  <img src="./loss.png" alt="Loss" width="45%">
</div>



## Confusion Matrix

<div align="center">
  <img src="./confusion_matrix.png" alt="Confusion Matrix" width=500>
</div>


