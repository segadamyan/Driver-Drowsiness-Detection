#Driver Drowsiness Detection
Sergey Adamyan student at YSU CS

Introduction

Background

Driver drowsiness poses a significant threat on road safety, contributing to numerous accidents and fatalities worldwide. The impaired cognitive and motor functions associated with drowsy driving can result in delayed reaction times, decreased vigilance, and an increased likelihood of accidents. According to [insert relevant statistics or studies], drowsy driving is a pervasive issue that demands effective and innovative solutions.

Objectives

In response to the critical need for enhanced road safety, this project aims to develop a robust and real-time driver drowsiness detection system. Leveraging the power of deep learning and machine learning techniques, our approach utilizes TensorFlow and Keras to create a sophisticated model capable of accurately identifying signs of driver drowsiness.

By focusing on the development of an advanced drowsiness detection system, we intend to contribute to the reduction of road accidents caused by driver fatigue. The implementation of this system holds the potential to not only enhance safety but also revolutionize the way we approach and mitigate the risks associated with drowsy driving.

This documentation serves as a comprehensive guide to the problem at hand, the methods employed for its resolution, the results obtained, and instructions for the practical implementation of the developed drowsiness detection system. As we delve into the intricacies of our methodology and the outcomes achieved, we invite the reader to explore the innovative intersection of technology and road safety that this project represents.

Problem Statement

Driver drowsiness is a critical factor contributing to road accidents and poses a serious threat to public safety. The impairment of cognitive and motor functions due to fatigue increases the risk of delayed reactions and decreased alertness while driving. This project addresses the urgent need for an effective solution to detect and mitigate the risks associated with drowsy driving. By leveraging TensorFlow and Keras, we aim to develop a real-time driver drowsiness detection system that can identify signs of driver fatigue and alert the driver, thus enhancing road safety and reducing the incidence of accidents caused by drowsiness.





Methodology

Data Collection

To train our drowsiness detection model, we gathered a dataset containing images of both opened and closed eyes. The dataset encompasses a diverse range of individuals and lighting conditions to enhance the model's robustness.

Model Architecture

Our model focuses on binary classification, distinguishing between opened and closed eyes. The convolutional neural network (CNN) architecture, implemented using TensorFlow and Keras, has been fine-tuned to effectively capture the subtle features indicative of drowsiness.

Training

For the training phase, we harnessed the computational power of the MSI RTX 3050ti GPU to expedite the model training process. Leveraging the parallel processing capabilities of a dedicated GPU significantly reduced the training time, allowing for quicker experimentation and model refinement.

To facilitate seamless accessibility and cost-effective computation, we conducted the training sessions on Google Colab. Google Colab provides free access to Graphics Processing Units (GPUs), including the NVIDIA Tesla K80, and can be easily configured to use the MSI RTX 3050ti GPU. This cloud-based platform enables collaborative and resource-efficient model development, overcoming potential hardware limitations.

Throughout the training sessions, we monitored key performance metrics, such as accuracy and loss, to ensure the model's convergence and effectiveness. The combination of the MSI RTX 3050ti GPU and Google Colab's infrastructure significantly enhanced the efficiency of our training pipeline, making the development of the drowsiness detection model both practical and accessible.

Results

In real-time application, we extract the driver's eye region from a captured image. This region is then fed into the trained model for prediction. The model outputs a probability indicating the likelihood of the eye being closed or opened. Based on a predefined threshold, we classify the driver's eye state.

By concentrating on eye state prediction, our methodology provides a lightweight and efficient solution for real-time drowsiness detection, making it suitable for various applications in driver assistance systems.
![image](https://github.com/segadamyan/Driver-Drowsiness-Detection/assets/29497688/fd1ca91f-2aa8-4d90-93d4-804a3f5c7b7d)
